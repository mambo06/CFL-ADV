# https://github.com/AstraZeneca/SubTab


import copy
import time
from tqdm import tqdm
import gc
import itertools

import mlflow
import yaml

import _eval as eval
from src.model import CFL
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims
import numpy as np

import torch
from pathlib import Path
import json


def main(config):
    
    
    # Disable adding noise since we are in evaluation mode
    config["add_noise"] = False
    # Turn off valiation
    config["validate"] = False
    # Get all of available training set for evaluation (i.e. no need for validation set)
    # config["training_data_ratio"] = 0.1

    for client in range(config["fl_cluster"]):

        eval.main(config, client)


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
    #----- Moving to evaluation stage
    # Reset the autoencoder dimension since it was changed in train.py
    config["dims"] = dims
    config["framework"] = config["dataset"]
    config['task_type'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['task_type']
    config['cat_policy'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['cat_policy']
    config['norm'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['norm']
    config['learning_rate_reducer'] = config['learning_rate']
    print_config_summary(config, args)
    main(config)
    

