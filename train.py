import copy
import time
from tqdm import tqdm
import gc

import mlflow
import yaml

import eval_training as eval
from src.model import CFL
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims, set_seed

import numpy as np

import torch
import json
from pathlib import Path

from torch.multiprocessing import Process
import os

import datetime
from itertools import islice

class Server:
    def __init__(self, model):
        self.global_model = model
        
    def aggregate_models(self, client_models):
        # FedAvg aggregation
        global_dict = self.global_model.encoder.state_dict()
        
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client.get_model_params()[k].float() for client in client_models if client.tloss != None]).mean(0)
            
        self.global_model.encoder.load_state_dict(global_dict)
        
    def distribute_model(self):
        return copy.deepcopy(self.global_model.encoder)


class Client:
    def __init__(self, model, dataloader, client_number):
        self.model = copy.deepcopy(model)
        self.dataloader = dataloader
        self.client_number = client_number
        self.tloss = None
        
    def train(self, batch_number):
        model = self.model
        train_loader = self.dataloader
        client = self.client_number
        i = batch_number

        # syncFed = True
        x,y = next(islice(train_loader, i, None))

        # np.random.seed(epoch)
        idx = np.random.permutation(x.shape[0])
        x = x[idx]
        y = y[idx]

        # skipping            
        if (client < int(config["fl_cluster"] * config["client_drop_rate"])) and \
            (i < int(total_batches * config["data_drop_rate"])) :
            model.loss["tloss_o"].append(np.nan)
            model.loss["tloss_b"].append(np.nan)
            model.loss["closs_b"].append(np.nan)
            model.loss["rloss_b"].append(np.nan)
            model.loss["zloss_b"].append(np.nan)
            self.tloss = None
            return
            # syncFed = False
        
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

        # total_loss, contrastive_loss, recon_loss, zrecon_loss = [], [], [], []
        model.optimizer_ae.zero_grad()

        tloss, closs, rloss, zloss = model.fit(x)

        model.loss["tloss_o"].append(tloss.item())
        model.loss["tloss_b"].append(tloss.item())
        model.loss["closs_b"].append(closs.item())
        model.loss["rloss_b"].append(rloss.item())
        model.loss["zloss_b"].append(zloss.item())

        # epoch_loss += tloss.item()
        tloss.backward()
        self.tloss = tloss
        return tloss
    
    def poison_model(self):
        # Scale up the weights significantly to affect the global model
        for param in self.model.parameters():
            param.data = param.data * 10  # Scaling attack
    
    def get_model_params(self):
        return copy.deepcopy(self.model.encoder.state_dict())

    def step(self):
        self.model.optimizer_ae.step()


def run(config, save_weights):
    set_dirs(config)
    set_seed(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"], client = 0)
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    global_model = CFL(config)
    server = Server(global_model)
    clients = []
    for client in range(config["fl_cluster"]):
        loader = Loader(config, dataset_name=config["dataset"], client = client).trainFL_loader
        client = Client(global_model, loader, client)
        clients.append(client)

    total_batches = len(loader)
    for epoch in range(config["epochs"]):
        for i in tqdm(range(total_batches)):
            for client in clients:
                client.train(i)
            
            server.aggregate_models(clients)

            for client in clients:
                model = client.model
                model.encoder = server.distribute_model()
                client.step()
                model.loss["tloss_e"].append(sum(model.loss["tloss_b"][-total_batches:-1]) / total_batches)

    for n,client in enumerate(clients):
        model = client.model

        model.saveTrainParams(n)

        # Save the model for future use
        _ = model.save_weights(n) if save_weights else None

        # Save the config file to keep a record of the settings
        prefix = "Client-" + str(n) + "-" + str(config['epochs']) + "e-" + str(config["fl_cluster"]) + "c-"  \
        + str(config["client_drop_rate"]) + "cd-" + str(config["data_drop_rate"])\
        + "dd-" + str(config["client_imbalance_rate"]) + "nc-" + str(config["class_imbalance"]) \
        + "ci-" + str(config["dataset"]) + "-"
        if config["local"] : prefix += "local"
        else : prefix += "FL"

        with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)



def main(config):
    config["framework"] = config["dataset"]
    config['task_type'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['task_type']
    config['cat_policy'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['cat_policy']
    config['norm'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['norm']
    config['learning_rate_reducer'] = config['learning_rate']
    # Get a copy of autoencoder dimensions
    dims = copy.deepcopy(config["dims"])
    cfg = copy.deepcopy(config)
    run(config,save_weights=True)
    eval.main(copy.deepcopy(cfg))
        



if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    main(config)
    
    

    
    
    

