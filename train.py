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
    def __init__(self, model):
        self.global_model = model
        self.global_dict = self.global_model.encoder.state_dict()

    def aggregate_models(self, client_models, rnd):
        for k in self.global_dict.keys():
            param_stack = [client.get_model_params()[k].float() for client in client_models]
            # Randomly sample parameters based on config['randomLevel']
            param_stack = random.sample(param_stack, int(len(client_models) * config['randomLevel'])) if rnd else param_stack
            self.global_dict[k] = torch.stack(param_stack).mean(0)

    def distribute_model(self):
        return self.global_dict


class Client:
    def __init__(self, model, dataloader, client_number):
        self.model = copy.deepcopy(model)
        self.dataloader = copy.deepcopy(dataloader)
        self.client_number = copy.deepcopy(client_number)
        self.slice = islice(self.dataloader, 0, None)
        self.data_iter = iter(dataloader)  # Create the iterator once

    def train(self):
        try:
            x, _ = next(self.data_iter)  # Fetch the next batch
        except StopIteration:
            # If the iterator is exhausted, reinitialize it
            self.data_iter = iter(self.dataloader)
            x, _ = next(self.data_iter)
            
        idx = torch.randperm(x.shape[0])  # Replace np.random.permutation with torch.randperm
        x = x[idx].to(self.model.device)  # Ensure data is on the correct device

        self.model.optimizer_ae.zero_grad()
        tloss, closs, rloss, zloss = self.model.fit(x)

        # Append losses to the model's loss dictionary
        self.model.loss["tloss_o"].append(tloss.item())
        self.model.loss["tloss_b"].append(tloss.item())
        self.model.loss["closs_b"].append(closs.item())
        self.model.loss["rloss_b"].append(rloss.item())
        self.model.loss["zloss_b"].append(zloss.item())

        tloss.backward()
        self.tloss = tloss
        return tloss

    def poison_model(self, scale):
        # Scale up the weights significantly to affect the global model
        for param in self.model.encoder.parameters():
            param.data.mul_(scale)  # In-place scaling for efficiency

    def get_model_params(self):
        return copy.deepcopy(self.model.encoder.state_dict())

    def step(self):
        self.model.optimizer_ae.step()

    def set_model(self, params):
        self.model.encoder.load_state_dict(params)


def run(config, save_weights, poison):
    ds_loader = Loader(config, dataset_name=config["dataset"], client=0)
    config = update_config_with_model_dims(ds_loader, config)
    global_model = CFL(config)
    server = Server(global_model)
    clients = []

    poison_clients = random.sample(range(config["fl_cluster"]), int(config["fl_cluster"] * config['poisonClient'])) if poison else []
    print('Warning: Poisoning applied to clients:', poison_clients)

    for clt in range(config["fl_cluster"]):
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
                    client.poison_model(config['poisonLevel'])

            server.aggregate_models(clients, rnd=config['randomClient'])

            for client in clients:
                client.set_model(server.distribute_model())
                client.step()
                client.model.loss["tloss_e"].append(sum(client.model.loss["tloss_b"][-total_batches:]) / total_batches)

        print(f'Epoch {epoch}, Loss: {tloss / (config["fl_cluster"] * total_batches):.4f}')

    for n, client in enumerate(clients):
        model = client.model
        model.saveTrainParams(n)

        if save_weights:
            model.save_weights(n)

        prefix = f"Client-{n}-{config['epochs']}e-{config['fl_cluster']}fl-{config['poisonClient']}pc-" \
                 f"{config['poisonLevel']}pl-{config['randomLevel']}rl-{config['dataset']}"
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
