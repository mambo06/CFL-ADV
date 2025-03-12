import copy
import time
from tqdm import tqdm
import gc

import mlflow
import yaml

import evaluation as eval
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
import random

class Server:
    def __init__(self, model):
        self.global_model = model
        self.global_dict = self.global_model.encoder.state_dict()
        
    def aggregate_models(self, client_models, rnd):
        # random  = [random.randint(0, 1) if random else None]
        
        for k in self.global_dict.keys():
            paramStack = [client.get_model_params()[k].float() for client in client_models]
            # random by 50%
            paramStack = random.sample(paramStack, int(len(client_models) * config['randomLevel'])) if rnd else paramStack
            self.global_dict[k] = torch.stack(paramStack).mean(0)
        
    def distribute_model(self):
        return self.global_dict


class Client:
    def __init__(self, model, dataloader, client_number):
        self.model = copy.deepcopy(model)
        self.dataloader = copy.deepcopy(dataloader)
        self.client_number = copy.deepcopy(client_number)
        self.slice = islice(self.dataloader, 0, None)
        # self.tloss = None
        
    def train(self):
        # model = self.model
        # train_loader = self.dataloader
        # client = self.client_number
        # i = batch_number

        # syncFed = True
        x,_ = next(iter(self.dataloader))

        # np.random.seed(epoch)
        idx = np.random.permutation(x.shape[0])
        x = x[idx]
        # y = y[idx]
        
        self.model.optimizer_ae.zero_grad()
        # if client ==1 :
        #     print(x)
        tloss, closs, rloss, zloss = self.model.fit(x)

        self.model.loss["tloss_o"].append(tloss.item())
        self.model.loss["tloss_b"].append(tloss.item())
        self.model.loss["closs_b"].append(closs.item())
        self.model.loss["rloss_b"].append(rloss.item())
        self.model.loss["zloss_b"].append(zloss.item())

        # epoch_loss += tloss.item()
        tloss.backward()
        # self.model.optimizer_ae.step()
        self.tloss = tloss
        return tloss
    
    def poison_model(self, scale):
        # Scale up the weights significantly to affect the global model
        for param in self.model.encoder.parameters():
            param.data = param.data * scale  # Scaling attack
    
    def get_model_params(self):
        return copy.deepcopy(self.model.encoder.state_dict())

    def step(self):
        self.model.optimizer_ae.step()

    def set_model(self, params):
        self.model.encoder.load_state_dict(params)


def run(config, save_weights, poison):
    # set_dirs(config)
    # set_seed(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"], client = 0)
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    global_model = CFL(config)
    server = Server(global_model)
    clients = []
    poisonClients = random.sample(range(config["fl_cluster"]), int(config["fl_cluster"] * config['poisonClient'])) if poison else [None]
    print('Warning posining applied to client :', poisonClients)
    for clt in range(config["fl_cluster"]):
        loader = Loader(config, dataset_name=config["dataset"], client = clt).trainFL_loader
        client = Client(global_model, loader, clt)
        client.poison = True if clt in poisonClients else False
        clients.append(client)

    total_batches = len(loader)
    for epoch in range(config["epochs"]):
        tloss = 0
        for i in tqdm(range(total_batches)):
            for n, client in enumerate(clients):
                tloss += client.train().item()
                # client.step()
                _ = client.poison_model(config['poisonLevel']) if client.poison else None
            
            server.aggregate_models(clients, rnd=config['randomClient'])

            for client in clients:
                client.set_model(server.distribute_model())
                client.step() 
                client.model.loss["tloss_e"].append(sum(client.model.loss["tloss_b"][-total_batches:-1]) / total_batches)
        print('epochs ', epoch,'loss : ', str(tloss/(config['fl_cluster']*total_batches)))

    for n,client in enumerate(clients):
        model = client.model

        model.saveTrainParams(n)

        # Save the model for future use
        _ = model.save_weights(n) if save_weights else None

        # Save the config file to keep a record of the settings
        prefix = "Client-" + str(client) + "-" + str(config['epochs']) + "e-" + str(config["fl_cluster"]) + "fl-"  \
        + str(config["poisonClient"]) + "pc-" + str(config["poisonLevel"]) +  "pl-" \
        + str(config["randomLevel"]) + "rl-" + str(config["dataset"]) 

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
    run(config,save_weights=True, poison = config['poison'])
    eval.main(copy.deepcopy(cfg))
        



if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    main(config)
    
    

    
    
    

