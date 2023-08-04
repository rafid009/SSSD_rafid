import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, parse_data

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer


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
        
        
    num_split = 2
    training_data = np.load(trainset_config['train_data_path'])
    print(f"training data: {training_data.shape}")

    transposed_mask = np.zeros_like(training_data)
    loss_mask = np.zeros_like(training_data)
    train_X = np.zeros_like(training_data)
    for i in range(training_data.shape[0]):
        train_X[i], _, transposed_mask[i], loss_mask[i] = parse_data(training_data[i])

    train_X = np.array_split(train_X, num_split, 0)
    transposed_mask = np.array_split(transposed_mask, num_split, 0)
    loss_mask = np.array_split(loss_mask, num_split, 0)
    # training_data = np.array(training_data, dtype=np.float32)
    # training_data = torch.from_numpy(training_data).float().cuda()
    print('Data loaded')
    # print(f"X: {train_X[-1]}\nmask: {transposed_mask[-1]}\nloss: {loss_mask[-1]}")
    
    
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        i = 0
        for batch in train_X:
            
            # if masking == 'rm':
            #     transposed_mask = get_mask_rm(batch[0], missing_k)
            # elif masking == 'mnr':
            #     transposed_mask = get_mask_mnr(batch[0], missing_k)
            # elif masking == 'bm':
            #     transposed_mask = get_mask_bm(batch[0], missing_k)
            
            batch = torch.from_numpy(batch).float().cuda()
            mask = torch.from_numpy(transposed_mask[i]).float().cuda()
            mask = mask.permute(0, 2, 1)
            
            target_mask = torch.from_numpy(loss_mask[i]).float().cuda()
            target_mask = target_mask.permute(0, 2, 1)
            # mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            # loss_mask = ~mask.bool()
            batch = batch.permute(0, 2, 1)
            # print(f"batch: {batch.shape}, mask: {mask.shape}, target: {target_mask.shape}")
            assert batch.size() == mask.size() == target_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, target_mask
            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)
            i += 1
            n_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./src/config/config_SSSDS4.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    train(**train_config)
