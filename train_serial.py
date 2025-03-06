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
from src.modelV1 import SubTab
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
        models.append( SubTab(config))
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
                    model.loss["tloss_o"].append(np.nan)
                    model.loss["tloss_b"].append(np.nan)
                    model.loss["closs_b"].append(np.nan)
                    model.loss["rloss_b"].append(np.nan)
                    model.loss["zloss_b"].append(np.nan)
                    
                    continue
                    # syncFed = False
                np.random.seed(epoch)
                idx = np.random.permutation(x.shape[0])
                x = x[idx]
                y = y[idx]

                ## class imbalance
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
                params.append(model.encoder)


            if config['local'] :
                model.optimizer_ae.step()
                continue

            
            
            beta = 1/len(params)

            for idx in range(len(params)):
                params[idx] = params[idx].state_dict()

            paramNames = [k for k,v in params[-1].items()]
            # print(paramNames)

            for paramName in paramNames:
                # print(paramName)
                for idx, each in enumerate(params):
                    if idx ==0 : cumulator = each[paramName].data * beta
                    else : cumulator += each[paramName].data * beta

                params[0][paramName].data.copy_(
                    cumulator
                    )

            for client in range(config["fl_cluster"]):
                if (client < int(config["fl_cluster"] * config["client_drop_rate"])) and \
                    (i < int(total_batches * config["data_drop_rate"])) :
                    
                    continue

                # print(client)
                model = models[client]

                model.encoder.load_state_dict(params[0])

                model.optimizer_ae.step()
                model.loss["tloss_e"].append(sum(model.loss["tloss_b"][-total_batches:-1]) / total_batches)


        training_time = time.process_time() - start

        if (epoch + 1) == config["epochs"]:
            for client in range(config["fl_cluster"]):
                model = models[client]

                model.saveTrainParams(client)

                # Save the model for future use
                _ = model.save_weights(client) if save_weights else None

                # Save the config file to keep a record of the settings
                prefix = "Client-" + str(client) + "-" + str(config['epochs']) + "e-" + str(config["fl_cluster"]) + "c-"  \
                + str(config["client_drop_rate"]) + "cd-" + str(config["data_drop_rate"])\
                + "dd-" + str(config["client_imbalance_rate"]) + "nc-" + str(config["class_imbalance"]) \
                + "ci-" + str(config["dataset"]) + "-"
                if config["local"] : prefix += "local"
                else : prefix += "FL"

                with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
                    yaml.dump(config, config_file, default_flow_style=False)

        print('Client ',
            client, ', epoch ', epoch, ': ',
            np.mean(model.loss["tloss_b"][ int(config["epochs"] * epoch) : int(config["epochs"] * (epoch+1)) ]))



        
        print(f"Training time epoch {epoch} :  {training_time // 60} minutes, {int(training_time % 60)} seconds")

# def average_gradients(model, world_cut=1):
#     """ Gradient averaging. """
#     size = float(dist.get_world_size() / world_cut)

#     for i,param in enumerate(model.parameters()):

#         # print(type(param), ": isinya : ",param)
#         dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
#         # datetime.timedelta(seconds=1)
#         param.grad.data /= size
#         # if type(param) is torch.Tensor:
#         #     print("masuk : ",i)
#         #     dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
#         #     param.grad.data /= size    

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

    

    
    
    

