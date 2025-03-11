import copy
import time
from tqdm import tqdm
import gc

import mlflow
import yaml

import eval
from src.model import SubTab
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims
import numpy as np

import torch

from torch.multiprocessing import Process
import os
# import torch.distributed as dist
import datetime
from itertools import islice
import pandas as pd


def run(config, save_weights=True):
    """Utility function for training and saving the model.
    Args:
        config (dict): Dictionary containing options and arguments.
        data_loader (IterableDataset): Pytorch data loader.
        save_weights (bool): Saves model if True.

    """
    # # Instantiate model
    models = []
    ds_loaders = []
    datas = []
    for client in range(config["fl_cluster"]):
        # models.append( SubTab(config))
        datas.append(Loader(config, dataset_name=config["dataset"], client = client).trainFL_loader)


    loss = {"tloss_b": [], "tloss_e": [], "vloss_e": [],
                     "closs_b": [], "rloss_b": [], "zloss_b": []}
       
    # train_loader = data_loader.trainFL_loader
    train_loader = datas[0]
    total_batches = len(datas[0])
    # validation_loader = data_loader.validation_loader

    
    # print(total_batches)
    # shuffle_list = config["shuffle_list"]
    # client = dist.get_rank()
    

    
    for epoch in range(config["epochs"]):
        if epoch==0: 

            xDict = {}
            yDict = {}
            for client in range(config["fl_cluster"]):
                xDict[client]=[]
                yDict[client]=[]
        epoch_loss = 0.0
        start = time.process_time()
        for i in tqdm(range(total_batches)):
            params = []

            for client in range(config["fl_cluster"]):
                model = models[client]
                train_loader = datas[client]

                # syncFed = True
                x,y = next(islice(train_loader, i, None))

                # skipping            
                if (client < int(config["fl_cluster"] * config["client_drop_rate"])) and \
                    (i < int(total_batches * config["data_drop_rate"])) :
                    y[y!=200] = 200 
                    # continue

                    # syncFed = False
                np.random.seed(epoch)
                idx = np.random.permutation(x.shape[0])
                x = x[idx]
                y = y[idx]

                ## class imbalance
                classes = [np.nan]
                if (
                    int(config["fl_cluster"] * config["client_drop_rate"]) <= 
                    client < 
                    ( 
                        int(config["fl_cluster"] * config["client_drop_rate"]) + int(config["fl_cluster"] * config["client_imbalance_rate"])
                        ) 
                    ) and (i < int(total_batches * config['class_imbalance']) ) : # cut half to make imbalance class
                    # print(x.shape,y)
                    np.random.seed(client)
                    classes = np.random.choice(config["n_classes"], 
                        config["n_classes"] - int(config["class_imbalance"] * config['n_classes']), 
                        replace = False )

                    x[np.in1d(y,classes)] = 0
                    y[np.in1d(y,classes)] = 300

                if epoch==0:
                    yDict[client].append(y.tolist())
                    xDict[client].append(x.tolist())

    for k,v in yDict.items():
        yDict[k] = [ x for xs in v for x in xs ]
    for k,v in xDict.items():
        xDict[k] = [ x for xs in v for x in xs ]
    # print(dataDict)
    pd.DataFrame(yDict).to_csv('y_'+config['dataset']+'_'+str(config['client_drop_rate'])+'_'+str(config['data_drop_rate'])+'_'+str(config['class_imbalance'])+'.csv', index=False)
    pd.DataFrame(xDict).to_csv('x_'+config['dataset']+'_'+str(config['client_drop_rate'])+'_'+str(config['data_drop_rate'])+'_'+str(config['class_imbalance'])+'.csv', index=False)



        

def main(config):
    """Main wrapper function for training routine.

    Args:
        config (dict): Dictionary containing options and arguments.

    """
    # Set directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"], client = 0)
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start training and save model weights at the end
    run(config, save_weights=True)



if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Overwrite the parent folder name for saving results
    config["framework"] = config["dataset"]
    # Get a copy of autoencoder dimensions
    dims = copy.deepcopy(config["dims"])
    cfg = copy.deepcopy(config)
    main(config)
    # Summarize config and arguments on the screen as a sanity check
    # config["shuffle_list"] = [[] for i in range( config["fl_cluster"])] # ordered shuffle each client / federated cluster

    

    
    
    

