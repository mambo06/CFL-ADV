

import sys
# import mlflow
import torch as th
import torch.utils.data
from tqdm import tqdm
import numpy as np

from src.model import CFL
from utils.arguments import get_arguments, get_config
from utils.arguments import print_config_summary
from utils.eval_utils_retrain import linear_model_eval, plot_clusters, append_tensors_to_lists, concatenate_lists, aggregate
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims




def eval(data_loader, config):
    torch.manual_seed(config['random'])
    model = CFL(config)
    # Load the model
    model.load_models(client=config['client'])
    # model.encoder.eval()
    model.set_mode('test')
    
    # Evaluate Autoencoder
    with th.no_grad():
        print(f" Evaluate embeddings dataset")      
        model.options['add_noise'] = False 
        z_train,  y_train = evalulate_models(data_loader, model, config, suffix="ClNoRetrain", mode="train", z_train=None, y_train=None)
        evalulate_models(data_loader, model, config, suffix="ClNoRetrain", mode="test", z_train=z_train, y_train=y_train)
        
        # End of the run
        print(f"Evaluation results are saved under ./results/{config['framework']}/evaluation/\n")
        print(f"{100 * '='}\n")

    # Retrain model
    ds_retrain = Loader(config, dataset_name=config["dataset"], drop_last=True)
    model = retrain_models(model,ds_retrain)

    with th.no_grad():
        print(f" Evaluate embeddings with retrain")
        model.options['add_noise'] = False
        model.encoder.eval()
        z_train,  y_train = evalulate_models(data_loader, model, config, suffix="ClRetrain", mode="train", z_train=None, y_train=None)
        model.options["add_noise"] = False

        results = evalulate_models(data_loader, model, config, suffix="ClRetrain", mode="test", z_train=z_train, y_train=y_train)
        
        # End of the run
        print(f"Evaluation results are saved under ./results/{config['framework']}/evaluation/\n")
        print(f"{100 * '='}\n")

    return results


def retrain_models(model,data_loader):
    # model.encoder.train()
    model.set_mode('training')
    model.options["add_noise"] = True
    data_loader_tr_or_te = data_loader.train_loader 
    
    # retrain
    print('Prepare for retrain, calcultae fisher matrix.......!')
    model.on_task_update(data_loader_tr_or_te)
    model.options['ewc'] = True

    # data_loader_tr_or_te = data_loader.test_loader
    data_loader_tr_or_te = data_loader.testUpper_loader
    # data_loader_ve = data_loader.validation_loader

    # parameters = [model.parameters() for _, model in model.model_dict.items()]
    # model.optimizer_ae = model._adam(parameters, lr=model.options["learning_rate"])
    model.set_autoencoder()
    # print('Opt before', model.optimizer_ae.state_dict())

    # print('Retrain with Validation.........!')
    # # retrain model
    # for g in range(0):
    #     train_tqdm = tqdm(enumerate(data_loader_ve), total=len(data_loader_ve), leave=True)
    #     for i, (x, _) in train_tqdm:
    #         model.fit(x)
    # print('Done retrain.........!')
    print('Retrain with New Case.........!', model.encoder.training)
    # retrain model
    for g in range(25):
        train_tqdm = tqdm(enumerate(data_loader_tr_or_te), total=len(data_loader_tr_or_te), leave=True)
        for i, (x, _) in train_tqdm:
            model.fit(x)
    print('Done retrain.........!')
    model.options['ewc'] = False
    # print('Opt after', model.optimizer_ae.state_dict())
    # sys.exit(0)
    return model
 

def evalulate_models(data_loader, model, config, suffix="_Test", mode='train', z_train=None, y_train=None, nData=None):
    break_line = lambda sym: f"{100 * sym}\n{100 * sym}\n"
    
    # Print whether we are evaluating training set, or test set
    decription = break_line('#') + f"Getting the joint embeddings of {suffix} set...\n" + \
                 break_line('=') + f"Dataset used: {config['dataset']}\n" + break_line('=')
    
    # Print the message         
    print(decription)
    if mode =='train':
        data_loader_tr = data_loader.train_loader

        data_loader_tr_retrain = data_loader.testUpper_loader

        z , clabels  = generateEncoded(data_loader_tr, model, config)
        z_, clabels_ = generateEncoded(data_loader_tr_retrain, model, config)

        return [z , clabels], [z_, clabels_]

    if mode == 'test':
        [z_train,y_train], [xFromTest,yFromTest] = z_train, y_train

        # data_loader_tr = data_loader.train_loader
        data_loader_te = data_loader.testUpper_loader
        data_loader_val = data_loader.testLower_loader
    
        # z , clabels  = generateEncoded(data_loader_tr, model, config)
        z_, clabels_ = generateEncoded(data_loader_te, model, config)
        x_, y_      = generateEncoded(data_loader_val, model, config)

        # Title of the section to print 
        print(20 * "*" + " Running evaluation trained on the joint embeddings" \
                       + " of training set and tested on that of test set" + 20 * "*")
        # Description of the task (Classification scores using Logistic Regression) to print on the command line
        description = "Sweeping models with arguments:"
        # Evaluate the embeddings
        suffixd= f"-{suffix}-classicalNoRetrain-"
        modelClasical = linear_model_eval(config, z_train, y_train, suffixd, 
        z_test=z_, y_test=clabels_, 
        z_val=x_, y_val=y_,
        description=description,x_=None,y_=None)
        
        suffixd = f"-{suffix}-classicalRetrain-"
        
        return linear_model_eval(config, z_train, y_train, suffixd, 
        # z_test=z_, y_test=clabels_,  _retrain
        z_test=z_, y_test=clabels_, 
        z_val=x_, y_val=y_,
        description=description,
        models=modelClasical,x_=xFromTest,y_=yFromTest)

def generateEncoded(dataLoader,model, config):
   
    model.encoder.to(config["device"])
    # Set the model to evaluation mode

    model.encoder.eval()
    total_batches = len(dataLoader)
    train_tqdm = tqdm(enumerate(dataLoader), total=total_batches, leave=True)
    z_val, clabels_val = [], []    
    for i, (x, label) in train_tqdm:
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
        if config['task_type'] == 'regression':
            label = label
        else : label = label.int()
        z_val, clabels_val = append_tensors_to_lists([z_val, clabels_val],
                                                 [latent, label])

    z = concatenate_lists([z_val])
    clabels = concatenate_lists([clabels_val])
    return z, clabels

def main(config):
    """Main function for evaluation

    Args:
        config (dict): Dictionary containing options and arguments.

    """
    # Set directories (or create if they don't exist)
    # prefix = (f"Cl-{client}-{config['epochs']}e-{config['fl_cluster']}fl-"
    #              f"{config['malClient']}mc-{config['attack_type']}_at-"
    #              f"{config['defense_type']}_dt-"
    #              f"{config['randomLevel']}rl-{config['dataset']}"
    #              )
    # config.update({"prefix":prefix})
    config['random'] = 95   
    set_dirs(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"], drop_last=False)
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start evaluation
    eval(ds_loader, config)


