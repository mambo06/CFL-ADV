# https://github.com/AstraZeneca/SubTab


import os

import datatable as dt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets as dts
import torchvision.transforms as transforms
import h5py
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import normalize, OneHotEncoder

from pathlib import Path
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import OrdinalEncoder


class Loader(object):
    """ Data loader """

    def __init__(self, config, dataset_name, client, drop_last=True, kwargs={}):
        """Pytorch data loader

        Args:
            config (dict): Dictionary containing options and arguments.
            dataset_name (str): Name of the dataset to load
            drop_last (bool): True in training mode, False in evaluation.
            kwargs (dict): Dictionary for additional parameters if needed

        """
        self.client = client
        # Get batch size
        batch_size = config["batch_size"]
        # Get config
        self.config = config
        # Set the paths
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name)
        # Get the datasets
        # train_dataset, test_dataset, validation_dataset, trainFl_dataset = self.get_dataset(dataset_name, file_path)
        testNS_dataset, trainNS_dataset, test_dataset, trainFl_dataset = self.get_dataset(dataset_name, file_path)
        # Set the loader for training set
        self.trainFL_loader = DataLoader(trainFl_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, **kwargs)
        # Set the loader for training set
        # self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, **kwargs)
        # Set the loader for test set
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        # Set the loader for validation set
        # self.validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, **kwargs)

        self.testNS_loader = DataLoader(testNS_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.trainNS_loader = DataLoader(trainNS_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        

    def get_dataset(self, dataset_name, file_path):
        
        """Returns training, validation, and test datasets"""
        # Create dictionary for loading functions of datasets.
        # If you add a new dataset, add its corresponding dataset class here in the form 'dataset_name': ClassName
        loader_map = {'default_loader': TabularDataset}
        # Get dataset. Check if the dataset has a custom class. 
        # If not, then assume a tabular data with labels in the first column
        dataset = loader_map[dataset_name] if dataset_name in loader_map.keys() else loader_map['default_loader']
        # Training and Validation datasets
        # train_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode='train', client = self.client)
        # Training and Validation datasets fort FL
        trainFl_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode='train_fl', client = self.client)
        # Test dataset
        test_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode='test', client = self.client)

        trainNS_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode='train_fl', client = self.client, NS=True)
        # Test dataset
        testNS_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode='test', client = self.client, NS=True)
        # validation dataset
        # validation_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode="validation", client = self.client)
        # Return
        # return train_dataset, test_dataset, validation_dataset, trainFl_dataset
        return testNS_dataset, trainNS_dataset, test_dataset, trainFl_dataset


class ToTensorNormalize(object):
    """Convert ndarrays to Tensors."""
    def __call__(self, sample):
        # Assumes that min-max scaling is done when pre-processing the data
        return torch.from_numpy(sample).float()


class TabularDataset(Dataset):
    def __init__(self, config, datadir, dataset_name, mode='train', client = 1, transform=ToTensorNormalize(), NS=False):
        """Dataset class for tabular data format.

        Args:
            config (dict): Dictionary containing options and arguments.
            datadir (str): The path to the data directory
            dataset_name (str): Name of the dataset to load
            mode (bool): Defines whether the data is for Train, Validation, or Test mode
            transform (func): Transformation function for data
            client : the client / silo number
            
        """
        self.client = client
        self.config = config
        self.baseGlobal = config['baseGlobal'] #if true then all data generated for base supervised model
        self.mode = mode
        self.paths = config["paths"]
        self.dataset_name = dataset_name
        self.data_path = os.path.join(self.paths["data"], dataset_name)
        self.ns = NS
        self.data, self.labels = self._load_data()
        self.transform = transform
        

    def __len__(self):
        """Returns number of samples in the data"""
        return len(self.data)
        # return self.data.shape[0]

    def __getitem__(self, idx):
        """Returns batch"""
        sample = self.data[idx]
        cluster = int(self.labels[idx])
        
        return sample, cluster
        

    def _load_data(self):
        dirs = [
             'adult',
             'aloi',
             'california_housing',
             'covtype',
             'epsilon',
             'helena',
             'higgs_small',
             'jannis',
             'microsoft',
             'yahoo',
             'year'
            ]
        # dirs = []
        dataset_name = self.config['dataset']

        dir_ = 'data/'+ dataset_name + '/normal' 
        if self.config['dataset'] in dirs : # if from other paper, datra not normalized
            N_train, N_test,N_val, y_train, y_test,y_val = self.joinData(cat_policy=self.config['cat_policy'],normalization=True, norm=self.config['norm'])
           
        else : # else data has been normalized

            N_train = np.load('data/'+dataset_name + '/normal'+'/N_train.npy')
            N_test = np.load('data/'+dataset_name + '/normal'+'/N_test.npy')
            N_val = np.load('data/'+dataset_name + '/normal'+'/N_val.npy')

            y_train = np.load('data/'+dataset_name + '/normal'+'/y_train.npy')
            y_test = np.load('data/'+dataset_name + '/normal'+'/y_test.npy')
            y_val = np.load('data/'+dataset_name + '/normal'+'/y_val.npy')

        x_train, y_train  = N_train[:,:-(N_train.shape[1] % self.config['fl_cluster'])], y_train, 
        x_test, y_test, = N_test[:,:-(N_train.shape[1] % self.config['fl_cluster'])], y_test, 
        x_val, y_val = N_val[:,:-(N_train.shape[1] % self.config['fl_cluster'])], y_val
        
        x_train = np.vstack((x_train, x_val))
        y_train = np.hstack((y_train, y_val))

        # print(f"dataset {self.config['dataset']} size {x_train.shape}")

        # Define the ratio of training-validation split, e.g. 0.8
        training_data_ratio = self.config["training_data_ratio"]
        
        # If validation is on, and trainin_data_ratio==1, stop and warn
        if self.config["validate"] and training_data_ratio >= 1.0:
            print(f"training_data_ratio must be < 1.0 if you want to run validation during training.")
            exit()

        #shuffle and cut data
        np.random.seed(0) # make sure similar permutation accros client test and validate
        data = x_train
        featShuffle = np.random.permutation(data.shape[1])
        min_lim = int(data.shape[1]/self.config['fl_cluster'] * self.client) 
        max_lim = int(data.shape[1]/self.config['fl_cluster'] * (self.client+1) )
        # print(featShuffle,min_lim,max_lim)
        
        x_train = x_train[:,featShuffle]
        if (self.baseGlobal==False) : x_train = x_train[:,min_lim : max_lim]

        x_test = x_test[:,featShuffle]
        if (self.baseGlobal==False) : x_test = x_test[:,min_lim : max_lim]
            

        # Shuffle indexes of samples to randomize training-validation split
        # np.random.seed(np.random.randint(1000)) 
        # idx = np.random.permutation(x_train.shape[0])
        idx = np.arange(x_train.shape[0])
        # print(idx)

        # Divide training and validation data : 
        # validation data = training_data_ratio:(1-training_data_ratio)
        tr_idx = idx[:int(len(idx) * training_data_ratio)]
        val_idx = idx[int(len(idx) * training_data_ratio):]

        # Validation data
        x_val = x_train[val_idx, :]
        y_val = y_train[val_idx]
        # print(" validation data : ",np.unique(y_val), y_val.shape)
        
        # Training data FL decode all 
        x_train_fl = x_train.copy()
        y_train_fl = y_train.copy()

        # Training data
        x_train = x_train[tr_idx, :]
        y_train = y_train[tr_idx]

        # Update number of classes in the config file in case that it is not correct.
        n_classes = len(list(set(y_train.reshape(-1, ).tolist())))
        if self.config["n_classes"] != n_classes:
            self.config["n_classes"] = n_classes
            print(f"{50 * '>'} Number of classes changed "
                  f"from {self.config['n_classes']} to {n_classes} {50 * '<'}")

        # Check if the values of features are small enough to work well for neural network
        if np.max(np.abs(x_train)) > 20:
            print(f"Pre-processing of data does not seem to be correct. "
                  f"Max value found in features is {np.max(np.abs(x_train))}\n"
                  f"Please check the values of features...")
            exit()
        # Select features and labels, based on the mode
        if self.mode == "train":
            data = x_train
            labels = y_train
        elif self.mode == "train_fl":
            data = x_train_fl
            labels = y_train_fl
        elif self.mode == "validation":
            data = x_val
            labels = y_val
        elif self.mode == "test":
            data = x_test
            labels = y_test
        else:
            print(f"Something is wrong with the data mode. "
                  f"Use one of three options: train, validation, and test.")
            exit()
        
        # if pearson reordering requirements

        if self.ns == False:
            sampling = x_train_fl[:] # data already suffled

            z = self.calculatePearson(sampling)
            data = data[:,z]
    
        # Return features, and labels
        return data, labels

    def calculatePearson(self, data):
        # Pearson Reordering
        return np.argsort(np.corrcoef(data.T)[0])
        # return (torch.corrcoef(data.T)[:1,:]).sort()[1][0]

    def rgbToGrey(self,rgb):
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).flatten()

    def joinData(self, cat_policy='ohe',seed=int(9),normalization=False, norm="l1"):
        dataset_name = self.config['dataset']
        dir_ = Path('data/'+ dataset_name + '/normal' )
        y_train = np.load(dir_.joinpath('y_train.npy'))
        y_test = np.load(dir_.joinpath('y_test.npy'))
        y_val = np.load(dir_.joinpath('y_val.npy'))
        # y = np.concatenate((y_train,y_test,y_val), axis=0)
        y = [y_train,y_test,y_val]
        result = []
        
        if dir_.joinpath('C_train.npy').exists():
            C_train = np.load(dir_.joinpath('C_train.npy'))
            C_test = np.load(dir_.joinpath('C_test.npy'))
            C_val = np.load(dir_.joinpath('C_val.npy'))
            # C = np.concatenate((C_train,C_test,C_val), axis=0)
            
            ord = OrdinalEncoder()
            C_train = ord.fit_transform(C_train)
            C_test = ord.transform(C_test)
            C_val = ord.transform(C_val)
            C = [C_train,C_test,C_val]
            
            
            if cat_policy == 'indices':
                C = C
            elif cat_policy == 'ohe':
                ohe = OneHotEncoder(
                    handle_unknown='ignore', sparse=False, dtype='float32'  # type: ignore[code]
                )
                ohe.fit(C[0])
                C[0] = ohe.transform(C[0])
                C[1] = ohe.transform(C[1])
                C[2] = ohe.transform(C[2])
            elif cat_policy == 'counter':
                assert seed is not None
                loo = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
                loo.fit(C[0], y[0])
                C[0] = loo.transform(C[0])  # type: ignore[code]
                C[1] = loo.transform(C[1])
                C[2] = loo.transform(C[2])
            result = C
                    
        if dir_.joinpath('N_train.npy').exists():
            N_train = np.load(dir_.joinpath('N_train.npy'))
            N_test = np.load(dir_.joinpath('N_test.npy'))
            N_val = np.load(dir_.joinpath('N_val.npy'))
            # N = np.concatenate((N_train,N_test,N_val), axis=0)
            N = [N_train,N_test,N_val]
            # print('size :',N_train.shape,N_test.shape, N_val.shape)
            result = N
            
        if ('N' in locals()) and ('C' in locals()):
            result[0] = np.hstack((C[0],N[0]))
            result[1] = np.hstack((C[1],N[1]))
            result[2] = np.hstack((C[2],N[2]))
        #dropna
        a = ~np.isnan(result[0]).any(axis=1)
        result[0] = result[0][a]
        y[0] = y[0][a]
        a = ~np.isnan(result[1]).any(axis=1)
        result[1] = result[1][a]
        y[1] = y[1][a]
        a = ~np.isnan(result[2]).any(axis=1)
        result[2] = result[2][a]
        y[2] = y[2][a]
        if normalization:
            mmx = MinMaxScaler()
            result[0] = mmx.fit_transform(result[0])
            result[2] = mmx.transform(result[2])

            result[1] = mmx.transform(result[1])
            
            # result[0] = normalize(result[0], norm=norm)
            # result[1] = normalize(result[1], norm=norm)
            # result[2] = normalize(result[2], norm=norm)  # type: ignore[code]
        if self.config['sampling'] < 1 :
            print('Warning : Sampling being applied ! ')
            idx = np.random.choice(range(result[0].shape[0]),int(result[0].shape[0]*self.config['sampling']), replace = False)
            result[0] = result[0][idx]
            y[0] = y[0][idx]
        return result[0],result[1],result[2], y[0],y[1],y[2]


