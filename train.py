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


def train(config, data_loader, save_weights=True):
    """Utility function for training and saving the model.
    Args:
        config (dict): Dictionary containing options and arguments.
        data_loader (IterableDataset): Pytorch data loader.
        save_weights (bool): Saves model if True.

    """
    # # Instantiate model
    # model = SubTab(config)
    # initiaate model list
    model_list = [SubTab(config) for i in range( config["fl_cluster"])] # model each federated cluster
    
    # print("print me", model_list) 

    #set shuffle column
    fShuffle = True

    loss = {"tloss_b": [], "tloss_e": [], "vloss_e": [],
                     "closs_b": [], "rloss_b": [], "zloss_b": []}
       
    train_loader = data_loader.trainFL_loader[:,:10]
    total_batches = len(train_loader)
    # validation_loader = data_loader.validation_loader

    
    # print(total_batches)
    shuffle_list = config["shuffle_list"]
    for epoch in range(config["epochs"]):
                
        # train_tqdm = tqdm(enumerate(train_loader), total=total_batches, leave=True)
        print(f"Start Training epoch {epoch} with {config['fl_cluster']} clients")
        train_tqdm = tqdm(enumerate(train_loader), total=total_batches, leave=True)
        for i, (x, y) in train_tqdm: # y is only for non iid data filter
            total_loss, contrastive_loss, recon_loss, zrecon_loss = [], [], [], []

            if fShuffle: 
                np.random.seed(10) # make sure similar permutation accros client test and validate
                featShuffle = np.random.permutation(x.shape[1])
                fShuffle = False

            start = time.process_time()

            for client in range(config["fl_cluster"]):
                model = model_list[client]

                #drop client's to limits data
                if (client < int(config["fl_cluster"] * config["client_drop_rate"])) and \
                (i < int(total_batches * config["data_drop_rate"])) : 
                    # print("skip")
                    if len(model.loss["tloss_o"]) > 1 : model.loss["tloss_o"].append(model.loss["tloss_o"][-1])
                    else : model.loss["tloss_o"].append(np.nan)
                    continue
                # else:
                    # print("not skip")

                
                # drop data to simulate imbalance non iid data
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
                    x[np.in1d(y,classes)] = 0 # drop random classes by fll it with 0
                    # print(f"enter imbalance data of client {client} filtered to {classes}")
                    # print(x)


                # else:
                #     print("not skip")

                
                # print(f"Training epoch {epoch} of Client {client}")
                
                x = x[:,featShuffle]
                
                client_dataset = x[:,int(client * (x.shape[1]/config["fl_cluster"])):int((client+1)*(x.shape[1]/config["fl_cluster"]))]

                # pearson reordering
                if (len(shuffle_list[client]) == 0 ):
                    z = (torch.corrcoef(client_dataset.T)[:1,:]).sort()[1]
                    # print(z[0])
                    shuffle_list[client] = z[0]

                client_dataset = client_dataset[:,shuffle_list[client]]


                # Fit the model to the data
                tloss, closs, rloss, zloss = model.fit(client_dataset)

                if config['local'] :
                    model.update_autoencoder(tloss)

                    model.loss["tloss_o"].append(tloss.item())
                    model.loss["tloss_b"].append(tloss.item())
                    model.loss["closs_b"].append(closs.item())
                    model.loss["rloss_b"].append(rloss.item())
                    model.loss["zloss_b"].append(zloss.item())
                    # print(f"skipping FL")

                    continue
                

                 # accoumeulted per steps
                total_loss.append(tloss)
                contrastive_loss.append(closs)
                recon_loss.append(rloss)
                zrecon_loss.append(zloss)

                model.loss["tloss_o"].append(tloss.item())
                    
                if (client == config["fl_cluster"]-1): # update model end of last clients inthe server
                    # do this in the server
                    # Compute the average of losses 

                    # n = len(total_loss)
                    n = len(total_loss) #pinalty push bigger learning 
                    # print(n)
                    total_loss = sum(total_loss) / n
                    # print(total_loss.item())
                    contrastive_loss = sum(contrastive_loss) / n
                    recon_loss = sum(recon_loss) / n
                    zrecon_loss = sum(zrecon_loss) / n

                    for client in range(config["fl_cluster"]):
                        model = model_list[client]
                        # # Record reconstruction loss
                        model.loss["tloss_b"].append(total_loss.item())
                        model.loss["closs_b"].append(contrastive_loss.item())
                        model.loss["rloss_b"].append(recon_loss.item())
                        model.loss["zloss_b"].append(zrecon_loss.item())

                        model.optimizer_ae.zero_grad()

                    total_loss.backward(retain_graph=True) # backward for all model, specific to pytorch

                    for client in range(config["fl_cluster"]):
            
                        model = model_list[client]
                        
                        model.optimizer_ae.step()

                    # total_loss, contrastive_loss, recon_loss, zrecon_loss = [], [], [], []

                    del contrastive_loss, recon_loss, zrecon_loss, tloss, closs, rloss, zloss
                    gc.collect()


       

        # # validate cilents
        
        # for client in range(config["fl_cluster"]):
        #     vloss = 0
        #     model = model_list[client]
        #     if epoch % model.options["nth_epoch"] == 0  and config["validate"]:
        #         # validation
        #         # print(f"Start Validating epoch {epoch} of Client {client}")
        #         total_batches = len(validation_loader)
        #         print(f"Computing validation loss of client {client}. #Batches: {epoch}")
                
        #         # print(total_batches,validation_loader)
        #         model.train_tqdm = tqdm(enumerate(validation_loader), total=total_batches, leave=True)

        #     # if (epoch>0) : 
        #         val_loss = []
        #         for i, (x, _) in model.train_tqdm:
        #             x = x[:,featShuffle]
        #             # if (i==0) : print("Size data : ", x.shape)
        #             client_dataset = x[:,int(client * (x.shape[1]/config["fl_cluster"])):int((client+1)*(x.shape[1]/config["fl_cluster"]))]
                    
        #             client_dataset = client_dataset[:,shuffle_list[client]]
        #             # x = client_dataset
        #             # y = torch.cat((torch.corrcoef(x.T)[:1,:],x),0)
        #             # client_dataset = y[:,y[0,:].sort()[1]]
        #             # client_dataset = client_dataset[1:,:]

        #             # print(x.shape,client_dataset.shape)                    

        #             # Fit the model to the data
        #             val_loss_s = model.validate_train(client,epoch, total_batches ,client_dataset)
        #             # print(val_loss_s.shape)
        #             val_loss.append(val_loss_s)
        #             # Delete the loss
        #             del val_loss_s

                    

        #         # Compute the validation loss for this batch
        #         val_loss = sum(val_loss) / len(val_loss)
        #         vloss = vloss + val_loss.item()
        #         # Clean up to avoid memory issues
        #         del val_loss
        #         gc.collect()

        #         # # Turn on training mode
        #         # self.set_mode(mode="training")
        #         # Compute mean validation loss
        #         vloss = vloss / total_batches
        #         # Record the loss
        #         model.loss["vloss_e"].append(vloss)
                # # Return mean validation loss
                # return vloss
        # print("Update log message using epoch and batch numbers", client)
        # 1 - Update log message using epoch and batch numbers
        for client in range(config["fl_cluster"]):
            model = model_list[client]
            model.loss["tloss_e"].append(sum(model.loss["tloss_b"][-total_batches:-1]) / total_batches)

            # Change learning rate if scheduler==True
            _ = model.scheduler.step() if model.options["scheduler"] else None
            
                # print (description)
                # Update the displayed message
            # model.train_tqdm.set_description(description)
        
            # Delete loss and associated graph for efficient memory usage
        # del total_loss, contrastive_loss, recon_loss, zrecon_loss, tloss, closs, rloss, zloss
        # gc.collect()


    training_time = time.process_time() - start

    # Report the training time
    print(f"Training time epoch {epoch} :  {training_time // 60} minutes, {training_time % 60} seconds")

    for i,model in enumerate(model_list) :
        # save pram training
        model.saveTrainParams(i)

        # Save the model for future use
        _ = model.save_weights(i) if save_weights else None

        # Save the config file to keep a record of the settings
        prefix = "Client-" + str(i) + "-" + str(config['epochs']) + "e-" + str(config["fl_cluster"]) + "c-"  \
        + str(config["client_drop_rate"]) + "cd-" + str(config["data_drop_rate"])\
        + "dd-" + str(config["client_imbalance_rate"]) + "nc-" + str(config["class_imbalance"]) \
        + "ci-" + str(config["dataset"]) + "-"
        if config["local"] : prefix += "local"
        else : prefix += "FL"

        with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)
        print("Done with training...")

        # Track results
        if config["mlflow"]:
            # Log config with mlflow
            mlflow.log_artifacts("./config", "config")
            # Log model and results with mlflow
            mlflow.log_artifacts(model._results_path + "/training/" + config["model_mode"] + "/plots", "training_results")
            # log model
            # mlflow.pytorch.log_model(model, "models")


def main(config):
    """Main wrapper function for training routine.

    Args:
        config (dict): Dictionary containing options and arguments.

    """
    # Set directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"])
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start training and save model weights at the end
    train(config, ds_loader, save_weights=True)


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
    config["shuffle_list"] = [[] for i in range( config["fl_cluster"])] # ordered shuffle each client / federated cluster
    print_config_summary(config, args)
    
    
    
    #----- If True, start of MLFlow for experiment tracking:
    if config["mlflow"]:
        # Experiment name
        experiment_name = "Give_Your_Experiment_A_Name"
        # Set the experiment
        mlflow.set_experiment(experiment_name=experiment_name + "_" + str(args.experiment))
        # Start a new mlflow run
        with mlflow.start_run():
            # Run the main with or without profiler
            run_with_profiler(main, config) if config["profile"] else main(config)
    else:
        #----- Run Training - with or without profiler
        run_with_profiler(main, config) if config["profile"] else main(config)
    
        #----- Moving to evaluation stage
        # Reset the autoencoder dimension since it was changed in train.py
        config["dims"] = dims
        # Disable adding noise since we are in evaluation mode
        config["add_noise"] = False
        # Turn off valiation
        config["validate"] = False
        # Get all of available training set for evaluation (i.e. no need for validation set)
        # config["training_data_ratio"] = 1.0
        if (len(config["shuffle_list"][0]) == 0) : exit()
        for client in range(config["fl_cluster"]):
            # Run Evaluation
            eval.main(config, client)
