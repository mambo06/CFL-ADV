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
from sklearn.preprocessing import normalize


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
        """Loads one of many available datasets, and returns features and labels"""

        if self.dataset_name.lower() in ["mnist"]:
            x_train, y_train, x_test, y_test = self._load_mnist()
            # print(type(x_train))
        elif self.dataset_name.lower() in ["blog"]:
            x_train, y_train, x_test, y_test = self._load_blog()
        elif self.dataset_name.lower() in ["income"]:
            x_train, y_train, x_test, y_test = self._load_income()
        elif self.dataset_name.lower() in ["cifar10"]:
            x_train, y_train, x_test, y_test = self._load_cifar()
            # print(type(x_train))
            x_train = tuple(map(self.rgbToGrey, x_train))
            x_test = tuple(map(self.rgbToGrey, x_test))
            
            x_train, y_train, x_test, y_test = list(x_train), list(y_train), list(x_test), list(y_test)

            for i, item in enumerate(x_train) :
                # print(item[0,:,:].shape)
                x_train[i] = np.asarray(item) #.ravel()
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            for i, item in enumerate(x_test) :
                x_test[i] = np.asarray(item ) #[0,:,:]).ravel()
            x_test = np.array(x_test)
            y_test = np.array(y_test)

            # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            # print(x_train[0][0]*0.2989 + x_train[0][1]*0.5870 + x_train[0][2]*0.1140)
            
            # print(x_train[:5])

        elif self.dataset_name.lower() in ["syn"]:
            x_train, y_train, x_test, y_test = self._load_syn()

        elif self.dataset_name.lower() in ["covtype"]:
            x_train, y_train, x_test, y_test = self._load_covtype()
        elif self.dataset_name.lower() in ["tuandromd"]:
            x_train, y_train, x_test, y_test = self._load_tuandromd()
        elif self.dataset_name.lower() in ["sensorless"]:
            x_train, y_train, x_test, y_test = self._load_sensorless()
        else:
            print(f"Given dataset name is not found. Check for typos, or missing condition "
                  f"in _load_data() of TabularDataset class in utils/load_data.py .")
            exit()

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


    def _load_mnist(self):
        """Loads MNIST dataset"""
        
        self.data_path = os.path.join("data/", "mnist")
        
        with open(self.data_path + '/train.npy', 'rb') as f:
            x_train = np.load(f)
            y_train = np.load(f)

        with open(self.data_path + '/test.npy', 'rb') as f:
            x_test = np.load(f)
            y_test = np.load(f)

        x_train = x_train.reshape(-1, 28 * 28) / 255.
        x_test = x_test.reshape(-1, 28 * 28) / 255.
       
        return x_train, y_train, x_test, y_test

    def _load_blog(self):
       
        x_train = np.load("data/blog/xtrain.npy")
        y_train = np.load("data/blog/ytrain.npy")
        x_test = np.load("data/blog/xtest.npy")
        y_test = np.load("data/blog/ytest.npy")
       
        return x_train, y_train, x_test, y_test

    def _load_income(self):
       
        x_train = np.load("data/income/train_feat_std.npy")
        y_train = np.load("data/income/train_label.npy")
        x_test = np.load("data/income/test_feat_std.npy")
        y_test = np.load("data/income/test_label.npy")
       
        return x_train, y_train, x_test, y_test

    def _load_cifar(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        train = dts.CIFAR10(root="data/",train=True,download=False, transform=transform)
        test = dts.CIFAR10(root="data/",train=False,download=False, transform=transform)

        x_train,y_train = list(zip(*train))
        x_test,y_test = list(zip(*test))
       
       
        return x_train, y_train, x_test, y_test

    def _load_syn(self):
        f = h5py.File('data/syn/syn.hdf5', 'r')
        inx = np.arange(f['labels'].shape[0])
        np.random.shuffle(inx)
       
        x_train = f['features'][:].T[inx][:-100]
        y_train = f['labels'][:][inx][:-100]
        x_test = f['features'][:].T[inx][-100:]
        y_test = f['labels'][:][inx][-100:]
        # print(x_train.shape, x_test.shape)   
       
        return x_train, y_train, x_test, y_test

    def _load_covtype(self):
        
        cov_type = fetch_covtype()
        X=normalize(cov_type.data, norm="l1")
        y=cov_type.target
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)
        return x_train, y_train, x_test, y_test

    def _load_tuandromd(self):
        data = pd.read_csv('data/TUANDROMD/TUANDROMD.csv')
        data = data.dropna(how='any',axis=0)
        data = data.values
        X = data[:,:-2]
        X=normalize(X, norm="l1")
        y = data[:,-1]

        inx = np.arange(X.shape[1])
        np.random.shuffle(inx)
        X = X[:,inx]
       
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)
       
        return x_train, y_train, x_test, y_test

    def _load_sensorless(self):
        data = np.loadtxt('data/SensorlessDriveDiagnosis/Sensorless_drive_diagnosis.txt')
        X = data[:,:-1]
        X=normalize(X, norm="l1")
        y = data[:,-1]

        np.random.seed(5)
        inx = np.arange(X.shape[1])
        np.random.shuffle(inx)

        X = X[:,inx]
       
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)
       
        return x_train, y_train, x_test, y_test

    def rgbToGrey(self,rgb):
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).flatten()


