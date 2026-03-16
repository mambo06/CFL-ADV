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

    def __init__(self, config, dataset_name, drop_last=True, kwargs={}):
        """Pytorch data loader"""
        self.client = config['client']
        # Get batch size
        batch_size = config["batch_size"]
        # Get config
        self.config = config
        # Set the paths
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name)
        
        # Get the datasets (now including upper and lower test datasets)
        testNS_dataset, trainNS_dataset, train_dataset, test_dataset, trainFl_dataset, testUpper_dataset, testLower_dataset = self.get_dataset(dataset_name, file_path)
        
        # Set the loaders
        self.trainFL_loader = DataLoader(trainFl_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, **kwargs)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.testNS_loader = DataLoader(testNS_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.trainNS_loader = DataLoader(trainNS_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        
        # Add loaders for upper and lower test datasets
        self.testUpper_loader = DataLoader(testUpper_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.testLower_loader = DataLoader(testLower_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        
    def get_dataset(self, dataset_name, file_path):
        """Returns training, validation, and test datasets"""
        # Create dictionary for loading functions of datasets
        loader_map = {'default_loader': TabularDataset}
        # Get dataset class
        dataset_class = loader_map[dataset_name] if dataset_name in loader_map.keys() else loader_map['default_loader']
        
        # Preload the data once (this will cache it for subsequent uses)
        dataset_class.load_base_data(self.config, dataset_name)
        
        # Create dataset instances (they will use the cached data)
        trainFl_dataset = dataset_class(self.config, datadir=file_path, dataset_name=dataset_name, mode='train_fl', client=self.client)
        train_dataset = dataset_class(self.config, datadir=file_path, dataset_name=dataset_name, mode='train', client=self.client)
        test_dataset = dataset_class(self.config, datadir=file_path, dataset_name=dataset_name, mode='test', client=self.client)
        trainNS_dataset = dataset_class(self.config, datadir=file_path, dataset_name=dataset_name, mode='train_fl', client=self.client, NS=True)
        testNS_dataset = dataset_class(self.config, datadir=file_path, dataset_name=dataset_name, mode='test', client=self.client, NS=True)
        
        # Create upper and lower half test datasets
        testUpper_dataset = dataset_class(self.config, datadir=file_path, dataset_name=dataset_name, mode='test_upper', client=self.client)
        testLower_dataset = dataset_class(self.config, datadir=file_path, dataset_name=dataset_name, mode='test_lower', client=self.client)
        
        return testNS_dataset, trainNS_dataset, train_dataset, test_dataset, trainFl_dataset, testUpper_dataset, testLower_dataset


class ToTensorNormalize(object):
    """Convert ndarrays to Tensors."""
    def __call__(self, sample):
        # Assumes that min-max scaling is done when pre-processing the data
        return torch.from_numpy(sample).float()


class TabularDataset(Dataset):
    # Class variable to store loaded data
    _cached_data = {}
    
    @classmethod
    def load_base_data(cls, config, dataset_name):
        """Load data once and cache it for reuse"""
        cache_key = dataset_name
        
        if cache_key not in cls._cached_data:
            dirs = [
                'adult', 'aloi', 'california_housing', 'covtype',
                'epsilon', 'helena', 'higgs_small', 'jannis',
                'microsoft', 'yahoo', 'year'
            ]
            
            dir_ = 'data/'+ dataset_name + '/normal'
            dir_ = 'data/'+ dataset_name 
            if dataset_name in dirs:  # if from paper, data not normalized
                N_train, N_test, N_val, y_train, y_test, y_val = cls._join_data(
                    config, dataset_name, cat_policy=config['cat_policy'], 
                    normalization=True, norm=config['norm']
                )
            else:  # else data has been normalized
                N_train = np.load('data/'+dataset_name + '/normal'+'/N_train.npy')
                N_test = np.load('data/'+dataset_name + '/normal'+'/N_test.npy')
                N_val = np.load('data/'+dataset_name + '/normal'+'/N_val.npy')
                
                y_train = np.load('data/'+dataset_name + '/normal'+'/y_train.npy')
                y_test = np.load('data/'+dataset_name + '/normal'+'/y_test.npy')
                y_val = np.load('data/'+dataset_name + '/normal'+'/y_val.npy')
            
            # Store all the loaded data in the cache
            cls._cached_data[cache_key] = {
                'N_train': N_train, 'N_test': N_test, 'N_val': N_val,
                'y_train': y_train, 'y_test': y_test, 'y_val': y_val,
            }
        
        return cls._cached_data[cache_key]
    
    @classmethod
    def _join_data(cls, config, dataset_name, cat_policy='ohe', seed=int(9), normalization=False, norm="l1"):
        """Extract from the original joinData method to make it a class method"""
        dir_ = Path('data/'+ dataset_name + '/normal')
        y_train = np.load(dir_.joinpath('y_train.npy'))
        y_test = np.load(dir_.joinpath('y_test.npy'))
        y_val = np.load(dir_.joinpath('y_val.npy'))
        y = [y_train, y_test, y_val]
        result = []
        
        if dir_.joinpath('C_train.npy').exists():
            C_train = np.load(dir_.joinpath('C_train.npy'))
            C_test = np.load(dir_.joinpath('C_test.npy'))
            C_val = np.load(dir_.joinpath('C_val.npy'))
            
            ord = OrdinalEncoder()
            C_train = ord.fit_transform(C_train)
            C_test = ord.transform(C_test)
            C_val = ord.transform(C_val)
            C = [C_train, C_test, C_val]
            
            if cat_policy == 'indices':
                C = C
            elif cat_policy == 'ohe':
                ohe = OneHotEncoder(
                    handle_unknown='ignore', sparse_output=False, dtype='float32'
                )
                ohe.fit(C[0])
                C[0] = ohe.transform(C[0])
                C[1] = ohe.transform(C[1])
                C[2] = ohe.transform(C[2])
            elif cat_policy == 'counter':
                assert seed is not None
                loo = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
                loo.fit(C[0], y[0])
                C[0] = loo.transform(C[0])
                C[1] = loo.transform(C[1])
                C[2] = loo.transform(C[2])
            result = C
                    
        if dir_.joinpath('N_train.npy').exists():
            N_train = np.load(dir_.joinpath('N_train.npy'))
            N_test = np.load(dir_.joinpath('N_test.npy'))
            N_val = np.load(dir_.joinpath('N_val.npy'))
            N = [N_train, N_test, N_val]
            result = N
            
        if ('N' in locals()) and ('C' in locals()):
            result[0] = np.hstack((C[0], N[0]))
            result[1] = np.hstack((C[1], N[1]))
            result[2] = np.hstack((C[2], N[2]))
            
        # dropna
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
            
        if config['sampling'] < 1:
            print('Warning : Sampling being applied ! ')
            idx = np.random.choice(range(result[0].shape[0]), int(result[0].shape[0]*config['sampling']), replace=False)
            result[0] = result[0][idx]
            y[0] = y[0][idx]
            
        return result[0], result[1], result[2], y[0], y[1], y[2]
    
    @classmethod
    def calculate_pearson(cls, data):
        """Calculate Pearson correlation for feature ordering"""
        return np.argsort(np.corrcoef(data.T)[0])
    
    def __init__(self, config, datadir, dataset_name, mode='train', client=1, transform=ToTensorNormalize(), NS=False):
        """Dataset class for tabular data format."""
        self.client = client
        self.config = config
        self.baseGlobal = config['baseGlobal']
        self.mode = mode
        self.paths = config["paths"]
        self.dataset_name = dataset_name
        self.data_path = os.path.join(self.paths["data"], dataset_name)
        self.ns = NS
        self.transform = transform
        
        # Load data from cache or load it if not available
        base_data = self.load_base_data(config, dataset_name)
        
        # Process the data based on mode, client, and NS flag
        self.data, self.labels = self._process_data(base_data)
    
    def _process_data(self, base_data):
        """Process the data based on mode, client, and NS flag"""
        # Extract data from the base_data dictionary
        N_train = base_data['N_train']
        N_test = base_data['N_test']
        N_val = base_data['N_val']
        y_train = base_data['y_train']
        y_test = base_data['y_test']
        y_val = base_data['y_val']
        
        # Continue with the processing logic from the original _load_data method
        trimIndex = -(N_train.shape[1] % self.config['fl_cluster']) if (N_train.shape[1] % self.config['fl_cluster']) != 0 else N_train.shape[1]
        
        x_train, y_train = N_train[:, :trimIndex], y_train
        x_test, y_test = N_test[:, :trimIndex], y_test
        x_val, y_val = N_val[:, :trimIndex], y_val
        
        x_train = np.vstack((x_train, x_val))
        y_train = np.hstack((y_train, y_val))
        
        # Define the ratio of training-validation split
        training_data_ratio = self.config["training_data_ratio"]
        
        # If validation is on, and training_data_ratio==1, stop and warn
        if self.config["validate"] and training_data_ratio >= 1.0:
            print(f"training_data_ratio must be < 1.0 if you want to run validation during training.")
            exit()
            
        # Shuffle and cut data
        np.random.seed(0)  # make sure similar permutation across client test and validate
        data = x_train
        featShuffle = np.random.permutation(data.shape[1])
        min_lim = int(data.shape[1]/self.config['fl_cluster'] * self.client)
        max_lim = int(data.shape[1]/self.config['fl_cluster'] * (self.client+1))
        
        x_train = x_train[:, featShuffle]
        # base global to get all data without split
        if (self.baseGlobal == False):
            x_train = x_train[:, min_lim:max_lim]
            
        x_test = x_test[:, featShuffle]
        if (self.baseGlobal == False):
            x_test = x_test[:, min_lim:max_lim]
            
        # Use fixed index array for reproducibility
        idx = np.arange(x_train.shape[0])
        
        # Divide training and validation data
        tr_idx = idx[:int(len(idx) * training_data_ratio)]
        val_idx = idx[int(len(idx) * training_data_ratio):]
        
        # Validation data
        x_val = x_train[val_idx, :]
        y_val = y_train[val_idx]
        
        # Training data FL decode all
        x_train_fl = x_train.copy()
        y_train_fl = y_train.copy()
        
        # Training data
        x_train = x_train[tr_idx, :]
        y_train = y_train[tr_idx]
        
        # Create upper and lower halves of test data
        test_samples = x_test.shape[0]
        mid_point = test_samples // 3
        
        x_test_upper = x_test[:mid_point, :]
        y_test_upper = y_test[:mid_point]
        
        x_test_lower = x_test[mid_point:, :]
        y_test_lower = y_test[mid_point:]

        # ---------------------------------------------------------------
        # Add 10% of random training data to test_upper
        # Use a fixed seed for reproducibility (different from feat seed)
        rng = np.random.default_rng(seed=42)
        n_augment = max(1, int(x_train.shape[0] * 0.1))
        aug_idx = rng.choice(x_train.shape[0], size=n_augment, replace=False)

        x_test_upper = np.vstack((x_test_upper, x_train[aug_idx, :]))
        y_test_upper = np.hstack((y_test_upper, y_train[aug_idx]))
        # ---------------------------------------------------------------
        
        # Update number of classes in the config file in case that it is not correct
        n_classes = len(list(set(y_train.reshape(-1, ).tolist())))
        if self.config["n_classes"] != n_classes:
            self.config["n_classes"] = n_classes
            print(f"{10 * '>'} Number of classes changed "
                  f"from {self.config['n_classes']} to {n_classes} {10 * '<'}")
            
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
        elif self.mode == "test_upper":
            data = x_test_upper
            labels = y_test_upper
        elif self.mode == "test_lower":
            data = x_test_lower
            labels = y_test_lower
        else:
            print(f"Something is wrong with the data mode. "
                  f"Use one of options: train, validation, test, test_upper, test_lower, or train_fl.")
            exit()
            
        # Apply Pearson reordering if NS is False
        if self.ns == False:
            sampling = x_train_fl[:]  # data already shuffled
            z = self.calculate_pearson(sampling)
            data = data[:, z]
            
        return data, labels


    def __len__(self):
        """Returns number of samples in the data"""
        return len(self.data)
        
    def __getitem__(self, idx):
        """Returns batch"""
        sample = self.data[idx]
        cluster = int(self.labels[idx])
        
        return sample, cluster



