# https://github.com/AstraZeneca/SubTab


import mlflow
import torch as th
import torch.utils.data
from tqdm import tqdm
import numpy as np
import copy

from src.modelV1 import SubTab
from utils.arguments import get_arguments, get_config
from utils.arguments import print_config_summary
from utils.eval_utils import linear_model_eval, plot_clusters, append_tensors_to_lists, concatenate_lists, aggregate
from utils.load_data import Loader
# from utils.load_data_new import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims
import pandas as pd
import pickle


torch.manual_seed(1)

# shuffle_list = None

def eval(data_loader, config, client, nData):
    """Wrapper function for evaluation.

    Args:
        data_loader (IterableDataset): Pytorch data loader.
        config (dict): Dictionary containing options and arguments.

    """
    # Instantiate Autoencoder model
    # model = SubTab(config)
    
    # Evaluate Autoencoder
    with th.no_grad():        
        # Get the joint embeddings and class labels of training set
        z_train, y_train = evalulate_original(data_loader, config, client, plot_suffix="training", mode="train")
        
        dataDict = evalulate_original(data_loader, config, client, plot_suffix="test", mode="test", z_train=z_train, y_train=y_train)
    return dataDict

        
    


def evalulate_original(data_loader, config, client, plot_suffix="_Test", mode='train', z_train=None, y_train=None, nData=None):
    
    z_l, clabels_l = [], []   

    # data_loader_tr_or_te = data_loader.trainFl_loader if mode == 'train' else data_loader.validationFl_loader
    data_loader_tr_or_te = data_loader.trainFL_loader if mode == 'train' else data_loader.test_loader

    # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
    train_tqdm = tqdm(enumerate(data_loader_tr_or_te), total=len(data_loader_tr_or_te), leave=True)

    

    # Go through batches
    total_batches = len(data_loader_tr_or_te)
    for i, (x, label) in train_tqdm:
        z_l, clabels_l = append_tensors_to_lists([z_l, clabels_l],
                                                 [x, label.int()])

    
    z = concatenate_lists([z_l])
  
    clabels = concatenate_lists([clabels_l])
 
    if mode == 'test':
        dataDict = {}
        dataDict['X_train'] = [z_train]
        dataDict['Y_train'] = [y_train]
        dataDict['X_test'] = [z]
        dataDict['Y_test'] = [clabels]
        return dataDict
        # for k,v in dataDict.items():
        #     dataDict[k] = [ x for xs in v for x in xs ]
        
        # pd.DataFrame(dataDict).to_csv('dataset_'+str(client)+'_'+config['dataset']+'_'+str(config['client_drop_rate'])+'_'+str(config['data_drop_rate'])+'_'+str(config['class_imbalance'])+'.csv', index=False)

       
    else:
        return z, clabels


def main(config, nData=None):
    """Main function for evaluation

    Args:
        config (dict): Dictionary containing options and arguments.

    """
    # Set directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    
    # Start evaluation
    cfg = copy.deepcopy(config)
    dataset = {}
    
    for client in range(config['fl_cluster']):
        config = copy.deepcopy(cfg)
        ds_loader = Loader(config, dataset_name=config["dataset"], client = client)
        config = update_config_with_model_dims(ds_loader, config)
        dataDict = eval(ds_loader, config, client, nData)
        dataset[client] = dataDict

    dbfile = open('dataset_'+config['dataset']+'_'+str(config['client_drop_rate'])+'_'+str(config['data_drop_rate'])+'_'+str(config['class_imbalance'])+'.pckl', 'ab')
    pickle.dump(dataset,dbfile)
    dbfile.close()

if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Overwrite the parent folder name for saving results
    config["framework"] = config["dataset"]
    # Turn off valiation
    config["validate"] = False
    # Get all of available training set for evaluation (i.e. no need for validation set)
    # config["training_data_ratio"] = 1.0
    # Turn off noise when evaluating the performance
    config["add_noise"] = False
    # Summarize config and arguments on the screen as a sanity check
    # print_config_summary(config, args)
    # --If True, start of MLFlow for experiment tracking:
    if config["mlflow"]:
        # Experiment name
        experiment_name = "Give_Your_Experiment_A_Name"
        mlflow.set_experiment(experiment_name=experiment_name + "_" + str(args.experiment))
        # Start a new mlflow run
        with mlflow.start_run():
            # Run the main with or without profiler
            run_with_profiler(main, config) if config["profile"] else main(config)
    else:
        # Run the main with or without profiler
        run_with_profiler(main, config) if config["profile"] else main(config)
