# import torch.nn.functional as F
# from enum import Enum
# import numpy as np
# from train import Server, Client
import yaml

from utils.arguments import get_arguments, get_config, print_config_summary
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims, set_seed
import json
from pathlib import Path
from utils.load_data import Loader
from src.model import CFL
import random
from tqdm import tqdm
# import torch
import evaluation as eval
import copy

from attacks.attackmanager import AttackManager
from clients.maliciousclient import MaliciousClient
from clients.secureclient import SecureClient
from servers.robustserver import RobustServer


def run(config, save_weights):
    config = copy.deepcopy(config)
    ds_loader = Loader(config, dataset_name=config["dataset"], client=0)
    config = update_config_with_model_dims(ds_loader, config)
    global_model = CFL(config)
    server = RobustServer(
        model = global_model, 
        config= config
        )
    clients = []
    
    # Initialize attack manager
    attack_manager = AttackManager(config)

    poison_clients = random.sample(
        range(config["fl_cluster"]), 
        int(config["fl_cluster"] * config['malClient'])
    ) if config['malClient'] > 0.0 else []

    print(f'Warning: Poisoning applied to clients: {poison_clients}')
    print(f'Attack type: {attack_manager.attack_type.value}')
    print(f'Defense type: {config["defense_type"]}')

    for clt in range(config["fl_cluster"]):
        prefix = (f"Cl-{clt}-{config['epochs']}e-{config['fl_cluster']}fl-"
                 f"{config['malClient']}mc-{config['attack_type']}_at-"
                 f"{config['defense_type']}_dt"
                 f"{config['randomLevel']}rl-{config['dataset']}"
                 )
        config.update({"prefix":prefix})
        loader = Loader(config, dataset_name=config["dataset"], client=clt).trainFL_loader
        # Use MaliciousClient for poisoned clients
        if clt in poison_clients:
            client = MaliciousClient(
                model = global_model, 
                dataloader = loader, 
                client_number = clt, 
                config = config, 
                attack_manager = attack_manager
                )
        else:
            client = SecureClient(
                 model = global_model, 
                 dataloader = loader, 
                 client_number = clt, 
                 config = config
                 )
        client.poison = clt in poison_clients
        clients.append(client)

    # Training loop
    total_batches = len(loader)
    for epoch in range(config["epochs"]):
        tloss = 0
        for i in tqdm(range(total_batches)):
            for client in clients:
                tloss += client.train().item()

            server.aggregate(client_models = clients)

            for client in clients:
                client.set_model(server.distribute_model())
                # client.step()
                client.model.loss["tloss_e"].append(
                    sum(client.model.loss["tloss_b"][-total_batches:]) / total_batches
                )

        avg_loss = tloss / (config["fl_cluster"] * total_batches)
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

    # Save results
    for n, client in enumerate(clients):
        model = client.model
        model.saveTrainParams(n)

        if save_weights:
            model.save_weights(n)

        prefix = (f"Cl-{clt}-{config['epochs']}e-{config['fl_cluster']}fl-"
                 f"{config['malClient']}mc-{config['attack_type']}_at-"
                 f"{config['defense_type']}_dt"
                 f"{config['randomLevel']}rl-{config['dataset']}"
                 )
        
        with open(model._results_path + f"/config_{prefix}.yml", 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

def main(config):
    config["framework"] = config["dataset"]
    info_path = Path(f'data/{config["dataset"]}/info.json')
    info = json.loads(info_path.read_text())
    
    # attack
    config.update({
        'task_type': info['task_type'],
        'cat_policy': info['cat_policy'],
        'norm': info['norm'],
        'learning_rate_reducer': config['learning_rate'],
        "attack_probability": 1.0,
        "target_layer": "encoder.layer1",
        "noise_std": 0.1,
        # "attack_type": ['scale', 'model_replacement','direction', 'gradient_ascent', 'targeted'][0],
        # "malClient": 0.5
    })

    # defense
    config.update({
        # 'defense_type': 
        #     [
        #     "multi_krum", 
        #     "geometric_median", 
        #     "foolsgold", 
        #     "trimmed_mean", 
        #     "momentum", 
        #     "random", 
        #     "robust"
        #     ][-2],
        'trim_ratio': 0.1,
        'random_level': 0.8,
        'history_size': 10,
        'num_groups': 5,
        'eps': 0.5,
        'min_samples': 3,
        'num_subsets': 5,
        'subset_size': 0.8,
        'window_size': 10,
        'detection_threshold': 2.0
    })

    run(copy.deepcopy(config), save_weights=True)
    # eval.main(copy.deepcopy(config))


if __name__ == "__main__":
    args = get_arguments()
    config = get_config(args)
    # print_config_summary(config)
    main(config)

