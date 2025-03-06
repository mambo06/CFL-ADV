# https://github.com/AstraZeneca/SubTab


import copy
import time
from tqdm import tqdm
import gc
import itertools

import mlflow
import yaml

import eval
from src.model import SubTab
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims
import numpy as np

import torch


# def train(config, data_loader, save_weights=True):


#     """Utility function for training and saving the model.
#     Args:
#         config (dict): Dictionary containing options and arguments.
#         data_loader (IterableDataset): Pytorch data loader.
#         save_weights (bool): Saves model if True.

#     """
#     # # Instantiate model
#     # model = SubTab(config)
#     # initiaate model list
#     model_list = [SubTab(config) for i in range( config["fl_cluster"])] # model each federated cluster
    
#     # print("print me", model_list) 

#     #set shuffle column
#     fShuffle = True

       
#     train_loader = data_loader.train_loader
#     total_batches = len(train_loader)
    
#     # print(total_batches)
#     shuffle_list = config["shuffle_list"]

#     # train_tqdm = tqdm(enumerate(train_loader), total=total_batches, leave=True)
#     print(f"preparing shuffle and Pearson Reaordering with {config['fl_cluster']} clients")
#     train_tqdm = tqdm(enumerate(train_loader), total=total_batches, leave=True)
#     for i, (x, _) in train_tqdm:
#         # total_loss, contrastive_loss, recon_loss, zrecon_loss = [], [], [], []

#         if fShuffle: 
#             np.random.seed(10) # make sure similar permutation accros client test and validate
#             featShuffle = np.random.permutation(x.shape[1])
#             fShuffle = False

#         start = time.process_time()

#         for client in range(config["fl_cluster"]):
#             model = model_list[client]            
            
#             # pearson reordering
#             if (len(shuffle_list[client]) == 0 ):
#                 x = x[:,featShuffle]
            
#                 client_dataset = x[:,int(client * (x.shape[1]/config["fl_cluster"])):int((client+1)*(x.shape[1]/config["fl_cluster"]))]
#                 y = (torch.corrcoef(client_dataset.T)[:1,:]).sort()[1]
#                 shuffle_list[client] = y[0]

#         if (np.sum([len(x) for x in shuffle_list]) ==  config["fl_cluster"]): break # Preparation done

#     print(f"preparing shuffle and Pearson Reaordering ..... Done")





# def main(config):
#     """Main wrapper function for training routine.

#     Args:
#         config (dict): Dictionary containing options and arguments.

#     """
#     # Set directories (or create if they don't exist)
#     set_dirs(config)
#     # Get data loader for first dataset.
#     ds_loader = Loader(config, dataset_name=config["dataset"])
#     # Add the number of features in a dataset as the first dimension of the model
#     config = update_config_with_model_dims(ds_loader, config)
#     # Start training and save model weights at the end
#     train(config, ds_loader, save_weights=True)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Overwrite the parent folder name for saving results
    config["framework"] = config["dataset"]
    # Get a copy of autoencoder dimensions
    dims = copy.deepcopy(config["dims"])
    # Summarize config and arguments on the screen as a sanity check
    # config["shuffle_list"] = [[] for i in range( config["fl_cluster"])] # ordered shuffle each client / federated cluster
    print_config_summary(config, args)
    
    
    #----- Moving to evaluation stage
    # Reset the autoencoder dimension since it was changed in train.py
    config["dims"] = dims
    # Disable adding noise since we are in evaluation mode
    config["add_noise"] = False
    # Turn off valiation
    config["validate"] = False
    # Get all of available training set for evaluation (i.e. no need for validation set)
    # config["training_data_ratio"] = 0.1

    for client in range(config["fl_cluster"]):

        eval.main(config, client, [client])

