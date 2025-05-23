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

import torch
import json
from pathlib import Path

from torch.multiprocessing import Process
import os

import datetime
from itertools import islice
import random


class Server:
    def __init__(self, model,config):
        self.global_model = model
        self.global_dict = self.global_model.encoder.state_dict()
        self.config = config

    def aggregate_models(self, client_models):
        aggregated_params = {}
        for k in self.global_dict.keys():
            param_stack = [client.get_model_params()[k].float() for client in client_models]
            
            # Check if parameters are identical
            # are_identical = True
            # first_params = param_stack[0]
            # for i in range(1, len(param_stack)):
            #     if not torch.all(torch.eq(param_stack[i], first_params)):
            #         are_identical = False
            #         break
                    
            # if are_identical:
            #     print(f"\nParameters are identical across clients for layer {k}")
            #     # Print some sample values and their gradients
            #     print(f"Sample values from {k}:")
            #     print(param_stack[i], first_params)
     
            aggregated_params[k] = torch.stack(param_stack).mean(0)
        
        self.global_dict = aggregated_params
        return aggregated_params



    def distribute_model(self):
        return self.global_dict


class Client:
    def __init__(self, model, dataloader, client_number):
        self.model = copy.deepcopy(model)
        if id(self.model) == id(model):
            print(f"Warning: Client {client_number} model is not a deep copy!")
        
        # Verify encoder parameters are different
        for p1, p2 in zip(self.model.encoder.parameters(), model.encoder.parameters()):
            if id(p1) == id(p2):
                print(f"Warning: Client {client_number} shares parameters with global model!")

        self.dataloader = copy.deepcopy(dataloader)
        self.client_number = copy.deepcopy(client_number)
        self.slice = islice(self.dataloader, 0, None)
        self.data_iter = iter(dataloader)  # Create the iterator once

    def train(self):
        try:
            x, _ = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            x, _ = next(self.data_iter)
            
        # Store parameters before training
        old_params = {k: v.clone() for k, v in self.model.encoder.state_dict().items()}
        
        idx = torch.randperm(x.shape[0])
        x = x[idx].to(self.model.device)

        self.model.optimizer_ae.zero_grad()
        tloss, closs, rloss, zloss = self.model.fit(x)

        # Update losses
        self.model.loss["tloss_o"].append(tloss.item())
        self.model.loss["tloss_b"].append(tloss.item())
        self.model.loss["closs_b"].append(closs.item())
        self.model.loss["rloss_b"].append(rloss.item())
        self.model.loss["zloss_b"].append(zloss.item())

        tloss.backward()
        self.model.optimizer_ae.step()

        # Verify parameters changed
        changed = False
        for k, old_v in old_params.items():
            new_v = self.model.encoder.state_dict()[k]
            if not torch.all(torch.eq(old_v, new_v)):
                changed = True
                break
        if not changed:
            print(f"Warning: Training did not change parameters for client {self.client_number}")

        self.tloss = tloss
        return tloss


    def poison_model(self, scale):
        # Scale up the weights significantly to affect the global model
        for param in self.model.encoder.parameters():
            param.data.mul_(scale)  # In-place scaling for efficiency

    def get_model_params(self):
        return copy.deepcopy(self.model.encoder.state_dict())

    # def step(self):
    #     self.model.optimizer_ae.zero_grad()
    #     self.tloss.backward()
    #     self.model.optimizer_ae.step()

    def set_model(self, params):
        old_params = {k: v.clone() for k, v in self.model.encoder.state_dict().items()}
        self.model.encoder.load_state_dict(params)
        
        # Verify parameters were updated
        changed = False
        for k, old_v in old_params.items():
            new_v = self.model.encoder.state_dict()[k]
            if not torch.all(torch.eq(old_v, new_v)):
                changed = True
                break
        if not changed:
            print(f"Warning: Model parameters were not updated for client {self.client_number}")



def run(config, save_weights, poison):
    config = copy.deepcopy(config)
    ds_loader = Loader(config, dataset_name=config["dataset"], client=0)
    config = update_config_with_model_dims(ds_loader, config)
    global_model = CFL(config)
    server = Server(global_model,config)
    clients = []

    poison_clients = random.sample(
        range(config["fl_cluster"]), 
        int(config["fl_cluster"] * config['malClient'])
    ) if config['malClient'] > 0.0 else []
    print('Warning: Poisoning applied to clients:', poison_clients)

    for clt in range(config["fl_cluster"]):
        prefix = (f"Cl-{clt}-{config['epochs']}e-{config['fl_cluster']}fl-"
                 f"{config['malClient']}mc-{config['attack_type']}_at-"
                 f"{config['randomLevel']}rl-{config['dataset']}")
        config.update({"prefix":prefix})

        loader = Loader(config, dataset_name=config["dataset"], client=clt).trainFL_loader
        client = Client(global_model, loader, clt)
        client.poison = clt in poison_clients
        clients.append(client)

    total_batches = len(loader)
    for epoch in range(config["epochs"]):
        tloss = 0
        for i in tqdm(range(total_batches)):
            for client in clients:
                tloss += client.train().item()
                if client.poison:
                    client.poison_model(config['attack_scale'])

            server.aggregate_models(clients)

            for client in clients:
                client.set_model(server.distribute_model())
                # client.step()
                client.model.loss["tloss_e"].append(sum(client.model.loss["tloss_b"][-total_batches:]) / total_batches)

        print(f'Epoch {epoch}, Loss: {tloss / (config["fl_cluster"] * total_batches):.4f}')

    for n, client in enumerate(clients):
        model = client.model
        model.saveTrainParams(n)

        if save_weights:
            model.save_weights(n)

        prefix = config['prefix']
        with open(model._results_path + f"/config_{prefix}.yml", 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)


def main(config):
    config["framework"] = config["dataset"]
    info_path = Path(f'data/{config["dataset"]}/info.json')
    info = json.loads(info_path.read_text())
    config.update({
        'task_type': info['task_type'],
        'cat_policy': info['cat_policy'],
        'norm': info['norm'],
        'learning_rate_reducer': config['learning_rate']
    })

    run(config, save_weights=True, poison=config['poison'])
    eval.main(copy.deepcopy(config))


if __name__ == "__main__":
    args = get_arguments()
    config = get_config(args)
    main(config)
