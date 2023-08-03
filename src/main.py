import numpy as np
import pandas as pd
from process_data import *
import os
import json
import torch
import torch.nn as nn

from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, parse_data

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    if use_model == 0:
        net = DiffWaveImputer(**model_config).cuda()
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).cuda()
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).cuda()
    else:
        print('Model chosen not available.')
    print_size(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

        
        
    
    ### Custom data loading and reshaping ###
        
        

    training_data = np.load(trainset_config['train_data_path'])
    training_data = np.array_split(training_data, 2, axis=0)
    training_data = np.array(training_data)
    # training_data = torch.from_numpy(training_data).float().cuda()
    print('Data loaded')

    
    
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch in training_data:

            # if masking == 'rm':
            #     transposed_mask = get_mask_rm(batch[0], missing_k)
            # elif masking == 'mnr':
            #     transposed_mask = get_mask_mnr(batch[0], missing_k)
            # elif masking == 'bm':
            #     transposed_mask = get_mask_bm(batch[0], missing_k)
            batch, obs_mask, mask, loss_mask = parse_data(batch)
            batch = torch.from_numpy(batch).float().cuda()
            mask = torch.from_numpy(mask).float().cuda()
            loss_mask = torch.from_numpy(loss_mask).float().cuda()
            batch = batch.permute(0, 2, 1)
            mask = mask.permute(0, 2, 1)
            loss_mask = loss_mask.permute(0, 2, 1)
            # mask = transposed_mask.permute(1, 0)
            # mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            # loss_mask = ~mask.bool()
            # batch = batch.permute(0, 2, 1)

            assert batch.size() == mask.size() == loss_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)

            loss.backward()
            optimizer.step()
            print(f"iteration: {n_iter} \tloss: {loss.item()}")
            # if n_iter % iters_per_logging == 0:
            #     print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1

# filename = "./src/ColdHardiness_Grape_Merlot_2.csv"
# df = pd.read_csv(filename)

# modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)
# season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)

# train_season_df = season_df.drop(season_array[-1], axis=0)
# train_season_df = train_season_df.drop(season_array[-2], axis=0)
# mean, std = get_mean_std(train_season_df, features)
# X, Y = split_XY(season_df, max_length, season_array, features)

# train_X = X[:-2]
# test_X = X[-2:]

# input_folder = './datasets/agaid'
# if not os.path.isdir(input_folder):
#     os.makedirs(input_folder)

# np.save(f"{input_folder}/train_agaid.npy", train_X)
# np.save(f"{input_folder}/test_agaid.npy", test_X)

config_path = "./src/config/config_SSSDS4.json"

with open(config_path) as f:
    config = json.load(f)
    print(config)

train_config = config["train_config"]  # training parameters

trainset_config = config["trainset_config"]  # to load trainset

diffusion_config = config["diffusion_config"]  # basic hyperparameters

diffusion_hyperparams = calc_diffusion_hyperparams(
    **diffusion_config)  # dictionary of all diffusion hyperparameters

# For SSSDS4
model_config = config['wavenet_config']

train(**train_config)
