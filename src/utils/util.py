import os
import numpy as np
import torch
import random


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=0):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape)
    if only_generate_missing == 1:
        z = audio * mask.float() + z * (1 - mask).float()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    print(f"transormed X: {transformed_X}")
    epsilon_theta = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta
    print(f"eps theta: {epsilon_theta}")
    if only_generate_missing == 1:
        return loss_fn(epsilon_theta * loss_mask, z * loss_mask)
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z)


def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))  # lenght of series indexes
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        idx = perm[0:k]
        mask[:, channel][idx] = 0

    return mask

def partial_bm(sample, selected_features, length_range, n_chunks):
    length = np.random.randint(length_range[0], length_range[1] + 1)
    k = length
    # mask = ~np.isnan(sample) * 1.0
    length_index = torch.tensor(range(sample.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = np.random.choice(list_of_segments_index, n_chunks, replace=False)
    gt_intact = sample.copy()
    # print(f"feats: {mask[selected_features]}")
    # print(f"snan: {s_nan}")
    # print(f"mask: {mask[selected_features][s_nan[0]:(s_nan[-1] + 1)]}")
    for chunk in range(n_chunks):
        # mask[selected_features][s_nan[chunk][0]:s_nan[chunk][-1] + 1] = 0
        gt_intact[s_nan[chunk][0]:s_nan[chunk][-1] + 1, selected_features] = np.nan
        # print(f"gt: {gt_intact}\ngt_snan: {gt_intact[s_nan[chunk][0]:s_nan[chunk][-1] + 1, selected_features]}")
    obs_data = np.nan_to_num(sample, copy=True)
    mask = ~np.isnan(gt_intact) * 1.0
    # print(f"mask 1: {mask}")
    return obs_data, mask, gt_intact

def parse_data(sample, rate=0.2, is_test=False, length=100, include_features=None, forward_trial=-1, lte_idx=None, random_trial=False, pattern=None,  partial_bm_config=None):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()

    obs_mask = ~np.isnan(sample)
    if not is_test:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        indices = np.where(~np.isnan(evals))[0].tolist()
        indices = np.random.choice(indices, int(len(indices) * rate))
        values = evals.copy()
        values[indices] = np.nan
        mask = ~np.isnan(values)
        mask = mask.reshape(shp)
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
    elif random_trial:
        evals = sample.copy()
        values = evals.copy()
        for i in range(evals.shape[1]):
            indices = np.where(~np.isnan(evals[:, i]))[0].tolist()
            indices = np.random.choice(indices, int(len(indices) * rate))
            values[indices, i] = np.nan
        mask = ~np.isnan(values)
        obs_data = np.nan_to_num(evals, copy=True)
    elif forward_trial != -1:
        indices = np.where(~np.isnan(sample[:, lte_idx]))[0].tolist()
        start = indices[forward_trial]
        obs_data = np.nan_to_num(sample, copy=True)
        gt_intact = sample.copy()
        gt_intact[start:, :] = np.nan
        mask = ~np.isnan(gt_intact)
    elif partial_bm_config is not None:
        total_features = np.arange(sample.shape[1])
        features = np.random.choice(total_features, partial_bm_config['features'])
        obs_data, mask, gt_intact = partial_bm(sample, features, partial_bm_config['length_range'], partial_bm_config['n_chunks'])
    else:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        a = np.arange(sample.shape[0] - length)
        # print(f"a: {a}\nsample: {sample.shape}")
        start_idx = np.random.choice(a)
        # print(f"random choice: {start_idx}")
        end_idx = start_idx + length
        obs_data_intact = sample.copy()
        if include_features is None or len(include_features) == 0:
            obs_data_intact[start_idx:end_idx, :] = np.nan
        else:
            obs_data_intact[start_idx:end_idx, include_features] = np.nan
        mask = ~np.isnan(obs_data_intact)
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
        # obs_intact = np.nan_to_num(obs_intact, copy=True)
    mask = mask.astype(float)
    obs_mask = obs_mask.astype(float)
    target_mask = obs_mask - mask
    # print(f"obs: {obs_mask.shape}, mask: {mask.shape}, target: {mask.shape}")
    return obs_data, obs_mask, mask, target_mask.astype(bool)

def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask
