# https://github.com/AstraZeneca/SubTab
import copy
import time
from tqdm import tqdm
import gc

import mlflow
import yaml

import evalV1 as eval
from src.modelV11 import SubTab
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.load_data_new import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims
import numpy as np
from utils.eval_utils_validation import linear_model_eval, aggregate, append_tensors_to_lists, concatenate_lists

import torch

from torch.multiprocessing import Process
import os
# import torch.distributed as dist
import datetime
from itertools import chain as ch

### chain from itertool adjust with torch
def chain(epochs, *iterables):
    # chain('ABC', 'DEF') --> A B C D E F
    for i in range(len(iterables) * epochs):
        # if (i%len(iterables)) == 0 : print('New begining')
        it = iterables[i%len(iterables)]
    # for it in iterables:
        for element in range(len(it)):
            yield next(iter(it))

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
    validDatas = []
    validScores = []
    validModels = []
    validepocs = []
    total_batches = 0
    lenTraining = 0    
    for client in range(config["fl_cluster"]):
        models.append( SubTab(config))
        if total_batches == 0 :
            total_batches = len(Loader(config, dataset_name=config["dataset"], client = client).trainFl_loader) +\
            len(Loader(config, dataset_name=config["dataset"], client = client).validationFl_loader) +\
            len(Loader(config, dataset_name=config["dataset"], client = client).test_loader)

        if lenTraining == 0 :
            lenTraining = len(Loader(config, dataset_name=config["dataset"], client = client).trainFl_loader)

        data = chain(config['epochs'],
            Loader(config, dataset_name=config["dataset"], client = client).trainFl_loader,
            Loader(config, dataset_name=config["dataset"], client = client).validationFl_loader,
            Loader(config, dataset_name=config["dataset"], client = client).test_loader
            )
        datas.append(data)
        valid = [
        Loader(config, dataset_name=config["dataset"], client = client).trainFl_loader,
        Loader(config, dataset_name=config["dataset"], client = client).test_loader
        ]
        validDatas.append(valid)
        validScores.append(0)
        validModels.append(np.nan)
        validepocs.append(np.nan)
    
    print('Batches',total_batches)


    loss = {"tloss_b": [], "tloss_e": [], "vloss_e": [],
                     "closs_b": [], "rloss_b": [], "zloss_b": []}
       
    # train_loader = data_loader.trainFL_loader
    # train_loader = datas[0]
    
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
                

                syncFed = True
                # x,y = next(islice(train_loader, i, None))
                x,_ = next(train_loader)
                # if(client==1 and i ==180): print(x)

                    
                # print(x.shape)
                # skipping was done in loader           
                # if (client < (config["fl_cluster"] * config["client_drop_rate"])) and \
                #     (i < (total_batches * config["data_drop_rate"] * config['training_data_ratio'])):
                #     syncFed = False
                # elif (client < int(config["fl_cluster"] * config["client_drop_rate"])) and \
                #     (lenTraining < i < lenTraining + (total_batches * config["data_drop_rate"] * (1-config['training_data_ratio']))):
                #     syncFed = False
                model.syncFed = syncFed
                #     model.loss["tloss_o"].append(np.nan)
                #     model.loss["tloss_b"].append(np.nan)
                #     model.loss["closs_b"].append(np.nan)
                #     model.loss["rloss_b"].append(np.nan)
                #     model.loss["zloss_b"].append(np.nan)
                    
                #     continue
                    # syncFed = False
                # np.random.seed(epoch)
                # idx = np.random.permutation(x.shape[0])
                # x = x[idx]
                # y = y[idx]

                ## class imbalance was done in loader
                # if (
                #     int(config["fl_cluster"] * config["client_drop_rate"]) <= 
                #     client < 
                #     ( 
                #         int(config["fl_cluster"] * config["client_drop_rate"]) + int(config["fl_cluster"] * config["client_imbalance_rate"])
                #         ) 
                #     ) and (i < int(total_batches * config['class_imbalance']) )   and imbalance : # cut half to make imbalance class
                #     # print(x.shape,y)
                #     np.random.seed(client)
                #     classes = np.random.choice(config["n_classes"], 
                #         config["n_classes"] - int(config["class_imbalance"] * config['n_classes']), 
                #         replace = False )

                #     x[np.in1d(y,classes)] = 0

                # total_loss, contrastive_loss, recon_loss, zrecon_loss = [], [], [], []
                

                
                if syncFed :
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
                # replaced in v1, syn everyone
                # if (client < int(config["fl_cluster"] * config["client_drop_rate"])) and \
                #     (i < int(total_batches * config["data_drop_rate"])) :
                    
                #     continue

                # print(client)
                model = models[client]

                if model.syncFed :
                    model.encoder.load_state_dict(params[0])

                    model.optimizer_ae.step()

                    model.loss["tloss_e"].append(sum(model.loss["tloss_b"][-total_batches:-1]) / total_batches)


        training_time = time.process_time() - start

        if (epoch + 1) == config["epochs"]:
            for client in range(config["fl_cluster"]):
                model = models[client]

                model.saveTrainParams(client)

                if not config['validate']:
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

        if config['validate']:
            for client in range(config["fl_cluster"]):
                model = models[client]
                data_loader_tr_or_te = validDatas[client][0]
                z_l, clabels_l = [], []
                total_batches = len(data_loader_tr_or_te)
                for i, (x, label) in enumerate(data_loader_tr_or_te):
                    x_tilde_list = model.subset_generator(x)

                    latent_list = []

                    # Extract embeddings (i.e. latent) for each subset
                    for xi in x_tilde_list:
                        # Turn xi to tensor, and move it to the device
                        Xbatch = model._tensor(xi)
                        # Extract latent
                        _, latent, _ = model.encoder(Xbatch) # decoded
                        # Collect latent
                        latent_list.append(latent)

                        
                    # Aggregation of latent representations
                    latent = aggregate(latent_list, config)

                        
                    # Append tensors to the corresponding lists as numpy arrays
                    z_l, clabels_l = append_tensors_to_lists([z_l, clabels_l],
                                                             [latent, label.int()])

                
                x_train, y_train = concatenate_lists([z_l]), concatenate_lists([clabels_l])

                data_loader_tr_or_te = validDatas[client][1]
                z_l, clabels_l = [], []
                total_batches = len(data_loader_tr_or_te)
                for i, (x, label) in enumerate(data_loader_tr_or_te):
                    x_tilde_list = model.subset_generator(x)

                    latent_list = []

                    # Extract embeddings (i.e. latent) for each subset
                    for xi in x_tilde_list:
                        # Turn xi to tensor, and move it to the device
                        Xbatch = model._tensor(xi)
                        # Extract latent
                        _, latent, _ = model.encoder(Xbatch) # decoded
                        # Collect latent
                        latent_list.append(latent)

                        
                    # Aggregation of latent representations
                    latent = aggregate(latent_list, config)
                        
                    # Append tensors to the corresponding lists as numpy arrays
                    z_l, clabels_l = append_tensors_to_lists([z_l, clabels_l],
                                                             [latent, label.int()])
                    
                x_test, y_test = concatenate_lists([z_l]), concatenate_lists([clabels_l])
                # print(x_train[0].shape,len(y_train),len(x_test),len(y_test))
                score = linear_model_eval(x_train,y_train,x_test,y_test)
                if validScores[client] < score :
                    validScores[client] = score
                    validModels[client] = model
                    validepocs[client] = epoch


        print(f"Training time epoch {epoch} :  {training_time // 60} minutes, {int(training_time % 60)} seconds")
        for client in range(config["fl_cluster"]):
            model = models[client]

            _ = model.scheduler.step() if model.options["scheduler"] else None


    print(f'Best Score : {validScores} with epochs {validepocs}')
    if config['validate']:
        for client in range(config["fl_cluster"]):
            model = validModels[client]


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
    if config['local'] == False:
        config['modeFL'] = True 
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

    config["validate"] = False
    # Get all of available training set for evaluation (i.e. no need for validation set)
    # config["training_data_ratio"] = 1.0
    # Turn off noise when evaluating the performance
    config["add_noise"] = False
    config['modeFL'] = False
    # eval.main(config)


    

    
    
    

