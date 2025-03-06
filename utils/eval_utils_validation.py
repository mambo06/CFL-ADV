# https://github.com/AstraZeneca/SubTab


import csv
import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from utils.utils import tsne
from utils.colors import get_color_list
import torch as th
from sklearn.svm import SVC
import pickle


def linear_model_eval(z_train, y_train, z_test, y_test, description="Logistic Reg."):
    # Sweep regularization parameter to see what works best for logistic regression

    c = 0.001 # overide all
    clf = LogisticRegression(max_iter=1200, solver='lbfgs',class_weight = 'balanced', C=c, multi_class='multinomial')
    # clf = SVC()
    # Fit model to the data
    clf.fit(z_train, y_train)
    # Score for training set
    tr_acc = clf.score(z_train, y_train)
    # Score for test set
    te_acc = clf.score(z_test, y_test)
    # Print results
    # print("Training score:", tr_acc)
    print("Test score:", te_acc)
    return te_acc

def aggregate(latent_list, config):
    """Aggregates the latent representations of subsets to obtain joint representation

    Args:
        latent_list (list[torch.FloatTensor]): List of latent variables, one for each subset
        config (dict): Dictionary holding the configuration

    Returns:
        (torch.FloatTensor): Joint representation

    """
    # Initialize the joint representation
    latent = None
    
    # Aggregation of latent representations
    if config["aggregation"]=="mean":
        latent = sum(latent_list)/len(latent_list)
    elif config["aggregation"]=="sum":
        latent = sum(latent_list)
    elif config["aggregation"]=="concat":
        latent = th.cat(latent_list, dim=-1)
    elif config["aggregation"]=="max":
        latent = functools.reduce(th.max, latent_list)
    elif config["aggregation"]=="min":
        latent = functools.reduce(th.min, latent_list)
    else:
        print("Proper aggregation option is not provided. Please check the config file.")
        exit()
        
    return latent

def append_tensors_to_lists(list_of_lists, list_of_tensors):
    """Appends tensors in a list to a list after converting tensors to numpy arrays

    Args:
        list_of_lists (list[lists]): List of lists, each of which holds arrays
        list_of_tensors (list[torch.tensorFloat]): List of Pytorch tensors

    Returns:
        list_of_lists (list[lists]): List of lists, each of which holds arrays

    """
    # Go through each tensor and corresponding list
    for i in range(len(list_of_tensors)):
        # Convert tensor to numpy and append it to the corresponding list
        list_of_lists[i] += [list_of_tensors[i].detach().numpy()]
    # Return the lists
    return list_of_lists

def concatenate_lists(list_of_lists):
    """Concatenates each list with the main list to a numpy array

    Args:
        list_of_lists (list[lists]): List of lists, each of which holds arrays

    Returns:
        (list[numpy.ndarray]): List containing numpy arrays

    """
    list_of_np_arrs = []
    # Pick a list of numpy arrays ([np_arr1, np_arr2, ...]), concatenate numpy arrs to a single one (np_arr_big),
    # and append it back to the list ([np_arr_big1, np_arr_big2, ...])
    for list_ in list_of_lists:
        list_of_np_arrs.append(np.concatenate(list_))
    # Return numpy arrays
    return list_of_np_arrs[0] if len(list_of_np_arrs) == 1 else list_of_np_arrs
        



