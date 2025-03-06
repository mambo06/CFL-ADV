"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Wrapper function for training routine.
"""

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
import torch.distributed as dist
import datetime


def run(config, datas, save_weights=True):
    """Utility function for training and saving the model.
    Args:
        config (dict): Dictionary containing options and arguments.
        data_loader (IterableDataset): Pytorch data loader.
        save_weights (bool): Saves model if True.

    """
    # # Instantiate model
    model = SubTab(config)
    # initiaate model list
    # model_list = [SubTab(config) for i in range( config["fl_cluster"])] # model each federated cluster
    
    # print("print me", model_list) 

    #set shuffle column
    # fShuffle = True

    loss = {"tloss_b": [], "tloss_e": [], "vloss_e": [],
                     "closs_b": [], "rloss_b": [], "zloss_b": []}
       
    # train_loader = data_loader.trainFL_loader
    train_loader = datas
    total_batches = len(train_loader)
    # validation_loader = data_loader.validation_loader

    
    # print(total_batches)
    # shuffle_list = config["shuffle_list"]
    client = dist.get_rank()
    

    
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        start = time.process_time()
                
        # train_tqdm = tqdm(enumerate(train_loader), total=total_batches, leave=True)
        # print(f"Start Training epoch {epoch} with {config['fl_cluster']} clients")
        # train_tqdm = tqdm(enumerate(train_loader), total=total_batches, leave=True)
        total_batches = len(train_loader)
        for i, (x, y) in enumerate(train_loader):  # y is only for non iid data filter
            syncFed = True

            # skipping            
            if (client < int(config["fl_cluster"] * config["client_drop_rate"])) and \
                (i < int(total_batches * config["data_drop_rate"])) :
                if config['local'] : continue
                x = torch.zeros(x.shape)
                syncFed = False

            ## class imbalance
            if (
                int(config["fl_cluster"] * config["client_drop_rate"]) <= 
                client < 
                ( 
                    int(config["fl_cluster"] * config["client_drop_rate"]) + int(config["fl_cluster"] * config["client_imbalance_rate"])
                    ) 
                ) and (i < int(config['n_classes'] * config['class_imbalance']) ) : # cut half to make imbalance class
                # print(x.shape,y)
                np.random.seed(client)
                classes = np.random.choice(config["n_classes"], 
                    config["n_classes"] - int(config["class_imbalance"] * config['n_classes']), 
                    replace = False )

                x[np.in1d(y,classes)] = 0

            total_loss, contrastive_loss, recon_loss, zrecon_loss = [], [], [], []
            model.optimizer_ae.zero_grad()

            tloss, closs, rloss, zloss = model.fit(x)

            model.loss["tloss_o"].append(tloss.item())
            model.loss["tloss_b"].append(tloss.item())
            model.loss["closs_b"].append(closs.item())
            model.loss["rloss_b"].append(rloss.item())
            model.loss["zloss_b"].append(zloss.item())

            epoch_loss += tloss.item()
            tloss.backward()


            if config['local'] :
                model.optimizer_ae.step()
                continue

            # reduce aggregation size hancle clients drop
            # skipping this step caused memory leak on client dropping
            if (config["client_drop_rate"] > 0 ) and \
                (i < int(total_batches * config["data_drop_rate"])) :
                average_gradients(model.encoder, float(config["client_drop_rate"]))
            else : 
                average_gradients(model.encoder)

            # prevent update for skipped clients due to data drop
            if syncFed : model.optimizer_ae.step()

            
        
            model.loss["tloss_e"].append(sum(model.loss["tloss_b"][-total_batches:-1]) / total_batches)

        if (epoch + 1) == config["epochs"]:
            model.saveTrainParams(config['rank'])

            # Save the model for future use
            _ = model.save_weights(config['rank']) if save_weights else None

            # Save the config file to keep a record of the settings
            prefix = "Client-" + str(config['rank']) + "-" + str(config['epochs']) + "e-" + str(config["fl_cluster"]) + "c-"  \
            + str(config["client_drop_rate"]) + "cd-" + str(config["data_drop_rate"])\
            + "dd-" + str(config["client_imbalance_rate"]) + "nc-" + str(config["class_imbalance"]) \
            + "ci-" + str(config["dataset"]) + "-"
            if config["local"] : prefix += "local"
            else : prefix += "FL"

            with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
                yaml.dump(config, config_file, default_flow_style=False)


        training_time = time.process_time() - start

        # Report the training time
        # print(f"{dist.get_rank()} Training time epoch {epoch} :  {training_time // 60} minutes, {training_time % 60} seconds")
        print('Client ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / total_batches)

def average_gradients(model, world_cut=1):
    """ Gradient averaging. """
    size = float(dist.get_world_size() / world_cut)

    for i,param in enumerate(model.parameters()):

        # print(type(param), ": isinya : ",param)
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        # datetime.timedelta(seconds=1)
        param.grad.data /= size
        # if type(param) is torch.Tensor:
        #     print("masuk : ",i)
        #     dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        #     param.grad.data /= size    

def init_processes(rank, size, config, datas, fn, backend='gloo'):

    # Set directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    # ds_loader = Loader(config, dataset_name=config["dataset"])
    # Add the number of features in a dataset as the first dimension of the model
    

    """ Initialize the distributed environment. """
    os.environ["GLOO_SOCKET_IFNAME"]="lo0"
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(config, datas)


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
    # Summarize config and arguments on the screen as a sanity check
    # config["shuffle_list"] = [[] for i in range( config["fl_cluster"])] # ordered shuffle each client / federated cluster

    

    
    
    size = config["fl_cluster"]
    processes = []
    print(f"Start Training with {config['fl_cluster']} clients")
    for rank in range(size):
        # config["dims"] = dims
        config = copy.deepcopy(cfg)
        ds_loader = Loader(config, dataset_name=config["dataset"], client = rank)
        config = update_config_with_model_dims(ds_loader, config)
        config['rank'] = rank
        # print_config_summary(config, args)
        datas = ds_loader.trainFL_loader
        p = Process(target=init_processes, args=(rank, size, config, datas, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
   # run_with_profiler(main, config) if config["profile"] else main(config)
    
    #----- Moving to evaluation stage
    # Reset the autoencoder dimension since it was changed in train.py
    config["dims"] = dims
    # Disable adding noise since we are in evaluation mode
    config["add_noise"] = False
    # Turn off valiation
    config["validate"] = False
    # Get all of available training set for evaluation (i.e. no need for validation set)
    # config["training_data_ratio"] = 1.0
    # if (len(config["shuffle_list"][0]) == 0) : exit()
    # for client in range(config["fl_cluster"]):
    #     # Run Evaluation
    #     eval.main(config, client)
