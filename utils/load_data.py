"""
Data loading utilities for federated learning on tabular datasets.
Reference: https://github.com/AstraZeneca/SubTab
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from category_encoders import LeaveOneOutEncoder


# ─── Data Loader ────────────────────────────────────────────────────────────

class Loader:
    """
    Main data loader for federated learning experiments.
    Creates multiple DataLoader instances for different data splits.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dataset_name: str,
        drop_last: bool = True,
        kwargs: Dict[str, Any] = None
    ) -> None:
        """
        Initialize data loaders for federated learning.
        
        Args:
            config: Configuration dictionary
            dataset_name: Name of the dataset to load
            drop_last: Whether to drop last incomplete batch
            kwargs: Additional arguments for DataLoader
        """
        self.client = config["client"]
        self.config = config
        
        batch_size = config["batch_size"]
        paths = config["paths"]
        file_path = os.path.join(paths["data"], dataset_name)
        
        kwargs = kwargs or {}
        
        # Load all dataset splits
        datasets = self._get_datasets(dataset_name, file_path)
        
        # Create data loaders
        self.trainFL_loader = DataLoader(
            datasets["train_fl"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            **kwargs
        )
        
        self.train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            **kwargs
        )
        
        self.test_loader = DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **kwargs
        )
        
        self.trainNS_loader = DataLoader(
            datasets["train_ns"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **kwargs
        )
        
        self.testNS_loader = DataLoader(
            datasets["test_ns"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **kwargs
        )
        
        self.testUpper_loader = DataLoader(
            datasets["test_upper"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **kwargs
        )
        
        self.testLower_loader = DataLoader(
            datasets["test_lower"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **kwargs
        )


    def _get_datasets(
        self,
        dataset_name: str,
        file_path: str
    ) -> Dict[str, Dataset]:
        """
        Create all dataset instances.
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to dataset files
            
        Returns:
            Dictionary of dataset instances
        """
        # Preload and cache data
        TabularDataset.load_base_data(self.config, dataset_name)
        
        # Create dataset instances
        datasets = {
            "train_fl": TabularDataset(
                self.config, file_path, dataset_name,
                mode="train_fl", client=self.client
            ),
            "train": TabularDataset(
                self.config, file_path, dataset_name,
                mode="train", client=self.client
            ),
            "test": TabularDataset(
                self.config, file_path, dataset_name,
                mode="test", client=self.client
            ),
            "train_ns": TabularDataset(
                self.config, file_path, dataset_name,
                mode="train_fl", client=self.client, NS=True
            ),
            "test_ns": TabularDataset(
                self.config, file_path, dataset_name,
                mode="test", client=self.client, NS=True
            ),
            "test_upper": TabularDataset(
                self.config, file_path, dataset_name,
                mode="test_upper", client=self.client
            ),
            "test_lower": TabularDataset(
                self.config, file_path, dataset_name,
                mode="test_lower", client=self.client
            ),
        }
        
        return datasets


# ─── Transform ──────────────────────────────────────────────────────────────

class ToTensorNormalize:
    """Convert numpy arrays to PyTorch tensors."""
    
    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        """
        Convert sample to tensor.
        
        Args:
            sample: Numpy array
            
        Returns:
            Float tensor
        """
        return torch.from_numpy(sample).float()


# ─── Dataset ────────────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data with federated learning support.
    Implements data caching to avoid redundant loading.
    """
    
    # Class-level cache for loaded data
    _cached_data: Dict[str, Dict[str, np.ndarray]] = {}
    
    # Datasets that require normalization
    DATASETS_REQUIRING_NORM = {
        'adult', 'aloi', 'california_housing', 'covtype',
        'epsilon', 'helena', 'higgs_small', 'jannis',
        'microsoft', 'yahoo', 'year'
    }


    def __init__(
        self,
        config: Dict[str, Any],
        datadir: str,
        dataset_name: str,
        mode: str = "train",
        client: int = 0,
        transform: Optional[ToTensorNormalize] = None,
        NS: bool = False
    ) -> None:
        """
        Initialize tabular dataset.
        
        Args:
            config: Configuration dictionary
            datadir: Directory containing dataset files
            dataset_name: Name of the dataset
            mode: Data split mode (train/test/validation/train_fl/test_upper/test_lower)
            client: Client ID for federated learning
            transform: Transform to apply to samples
            NS: If True, skip Pearson correlation ordering
        """
        self.client = client
        self.config = config
        self.baseGlobal = config.get("baseGlobal", False)
        self.mode = mode
        self.dataset_name = dataset_name
        self.data_path = datadir
        self.ns = NS
        self.transform = transform or ToTensorNormalize()
        
        # Load and process data
        base_data = self.load_base_data(config, dataset_name)
        self.data, self.labels = self._process_data(base_data)


    @classmethod
    def load_base_data(
        cls,
        config: Dict[str, Any],
        dataset_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Load and cache base dataset.
        
        Args:
            config: Configuration dictionary
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing train/test/val splits
        """
        if dataset_name in cls._cached_data:
            return cls._cached_data[dataset_name]
        
        print(f"[INFO] Loading dataset: {dataset_name}")
        
        # Load data based on dataset type
        if dataset_name in cls.DATASETS_REQUIRING_NORM:
            N_train, N_test, N_val, y_train, y_test, y_val = cls._join_data(
                config, dataset_name,
                cat_policy=config["cat_policy"],
                normalization=True,
                norm=config["norm"]
            )
        else:
            data_dir = Path(f"data/{dataset_name}")
            N_train = np.load(data_dir / "N_train.npy")
            N_test = np.load(data_dir / "N_test.npy")
            N_val = np.load(data_dir / "N_val.npy")
            y_train = np.load(data_dir / "y_train.npy")
            y_test = np.load(data_dir / "y_test.npy")
            y_val = np.load(data_dir / "y_val.npy")
        
        # Cache the data
        cls._cached_data[dataset_name] = {
            "N_train": N_train,
            "N_test": N_test,
            "N_val": N_val,
            "y_train": y_train,
            "y_test": y_test,
            "y_val": y_val,
        }
        
        print(f"[INFO] Dataset loaded - Train: {N_train.shape}, Test: {N_test.shape}")
        
        return cls._cached_data[dataset_name]


    @classmethod
    def _join_data(
        cls,
        config: Dict[str, Any],
        dataset_name: str,
        cat_policy: str = "ohe",
        seed: int = 9,
        normalization: bool = False,
        norm: str = "l1"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess dataset with categorical encoding.
        
        Args:
            config: Configuration dictionary
            dataset_name: Name of the dataset
            cat_policy: Categorical encoding policy ('ohe', 'counter', 'indices')
            seed: Random seed
            normalization: Whether to apply normalization
            norm: Normalization type
            
        Returns:
            Tuple of (X_train, X_test, X_val, y_train, y_test, y_val)
        """
        data_dir = Path(f"data/{dataset_name}")
        
        # Load labels
        y_train = np.load(data_dir / "y_train.npy")
        y_test = np.load(data_dir / "y_test.npy")
        y_val = np.load(data_dir / "y_val.npy")
        y = [y_train, y_test, y_val]
        
        result = []
        
        # Load and encode categorical features
        if (data_dir / "C_train.npy").exists():
            C_train = np.load(data_dir / "C_train.npy")
            C_test = np.load(data_dir / "C_test.npy")
            C_val = np.load(data_dir / "C_val.npy")
            
            # Ordinal encoding first
            ord_encoder = OrdinalEncoder()
            C_train = ord_encoder.fit_transform(C_train)
            C_test = ord_encoder.transform(C_test)
            C_val = ord_encoder.transform(C_val)
            C = [C_train, C_test, C_val]
            
            # Apply categorical policy
            if cat_policy == "indices":
                pass  # Keep as is
            elif cat_policy == "ohe":
                ohe = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype="float32"
                )
                C[0] = ohe.fit_transform(C[0])
                C[1] = ohe.transform(C[1])
                C[2] = ohe.transform(C[2])
            elif cat_policy == "counter":
                loo = LeaveOneOutEncoder(
                    sigma=0.1,
                    random_state=seed,
                    return_df=False
                )
                C[0] = loo.fit_transform(C[0], y[0])
                C[1] = loo.transform(C[1])
                C[2] = loo.transform(C[2])
            
            result = C
        
        # Load numerical features
        if (data_dir / "N_train.npy").exists():
            N_train = np.load(data_dir / "N_train.npy")
            N_test = np.load(data_dir / "N_test.npy")
            N_val = np.load(data_dir / "N_val.npy")
            N = [N_train, N_test, N_val]
            result = N
        
        # Concatenate categorical and numerical features
        if result and len(result) > 0:
            if 'C' in locals() and 'N' in locals():
                result[0] = np.hstack((C[0], N[0]))
                result[1] = np.hstack((C[1], N[1]))
                result[2] = np.hstack((C[2], N[2]))
        
        # Remove NaN values
        for i in range(3):
            mask = ~np.isnan(result[i]).any(axis=1)
            result[i] = result[i][mask]
            y[i] = y[i][mask]
        
        # Apply normalization
        if normalization:
            scaler = MinMaxScaler()
            result[0] = scaler.fit_transform(result[0])
            result[1] = scaler.transform(result[1])
            result[2] = scaler.transform(result[2])
        
        # Apply sampling if configured
        if config.get("sampling", 1.0) < 1.0:
            print(f"[WARNING] Sampling {config['sampling']*100:.0f}% of training data")
            n_samples = int(result[0].shape[0] * config["sampling"])
            idx = np.random.choice(result[0].shape[0], n_samples, replace=False)
            result[0] = result[0][idx]
            y[0] = y[0][idx]
        
        return result[0], result[1], result[2], y[0], y[1], y[2]


    def _process_data(
        self,
        base_data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process data for specific client and mode.
        
        Args:
            base_data: Dictionary containing base dataset splits
            
        Returns:
            Tuple of (data, labels) for the specified mode
        """
        # Extract base data
        N_train = base_data["N_train"]
        N_test = base_data["N_test"]
        N_val = base_data["N_val"]
        y_train = base_data["y_train"]
        y_test = base_data["y_test"]
        y_val = base_data["y_val"]
        
        # Trim features to be divisible by number of clients
        n_features = N_train.shape[1]
        n_clients = self.config["fl_cluster"]
        trim_idx = n_features - (n_features % n_clients) if n_features % n_clients != 0 else n_features
        
        x_train = N_train[:, :trim_idx]
        x_test = N_test[:, :trim_idx]
        x_val = N_val[:, :trim_idx]
        
        # Merge train and validation
        x_train = np.vstack((x_train, x_val))
        y_train = np.hstack((y_train, y_val))
        
        # Validate configuration
        training_ratio = self.config["training_data_ratio"]
        if self.config["validate"] and training_ratio >= 1.0:
            raise ValueError(
                "training_data_ratio must be < 1.0 when validation is enabled"
            )
        
        # Feature shuffling for federated split
        np.random.seed(0)  # Fixed seed for reproducibility
        feat_shuffle = np.random.permutation(x_train.shape[1])
        
        # Calculate feature range for this client
        features_per_client = x_train.shape[1] // n_clients
        min_lim = features_per_client * self.client
        max_lim = features_per_client * (self.client + 1)
        
        # Apply feature shuffling
        x_train = x_train[:, feat_shuffle]
        x_test = x_test[:, feat_shuffle]
        
        # Split features by client (unless baseGlobal is True)
        if not self.baseGlobal:
            x_train = x_train[:, min_lim:max_lim]
            x_test = x_test[:, min_lim:max_lim]
        
        # Split train/validation
        n_train = int(len(x_train) * training_ratio)
        x_val = x_train[n_train:]
        y_val = y_train[n_train:]
        
        x_train_fl = x_train.copy()
        y_train_fl = y_train.copy()
        
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
        
        # Create upper/lower test splits
        n_test = x_test.shape[0]
        split_point = n_test // 3
        
        x_test_upper = x_test[:split_point]
        y_test_upper = y_test[:split_point]
        
        x_test_lower = x_test[split_point:]
        y_test_lower = y_test[split_point:]
        
        # Augment upper test set with 10% random training samples
        rng = np.random.default_rng(seed=42)
        n_augment = max(1, int(x_train.shape[0] * 0.1))
        aug_idx = rng.choice(x_train.shape[0], size=n_augment, replace=False)
        
        x_test_upper = np.vstack((x_test_upper, x_train[aug_idx]))
        y_test_upper = np.hstack((y_test_upper, y_train[aug_idx]))
        
        # Update number of classes
        n_classes = len(np.unique(y_train))
        if self.config.get("n_classes") != n_classes:
            print(f"[INFO] Number of classes updated: {n_classes}")
            self.config["n_classes"] = n_classes
        
        # Sanity check for feature values
        max_val = np.max(np.abs(x_train))
        if max_val > 20:
            raise ValueError(
                f"Feature values too large (max={max_val:.2f}). "
                "Check preprocessing/normalization."
            )
        
        # Select data based on mode
        mode_map = {
            "train": (x_train, y_train),
            "train_fl": (x_train_fl, y_train_fl),
            "validation": (x_val, y_val),
            "test": (x_test, y_test),
            "test_upper": (x_test_upper, y_test_upper),
            "test_lower": (x_test_lower, y_test_lower),
        }
        
        if self.mode not in mode_map:
            raise ValueError(
                f"Invalid mode '{self.mode}'. "
                f"Choose from: {list(mode_map.keys())}"
            )
        
        data, labels = mode_map[self.mode]
        
        # Apply Pearson correlation ordering (unless NS=True)
        if not self.ns:
            pearson_order = self._calculate_pearson(x_train_fl)
            data = data[:, pearson_order]
        
        return data, labels


    @staticmethod
    def _calculate_pearson(data: np.ndarray) -> np.ndarray:
        """
        Calculate Pearson correlation ordering.
        
        Args:
            data: Input data array
            
        Returns:
            Array of feature indices sorted by correlation
        """
        corr_matrix = np.corrcoef(data.T)
        return np.argsort(corr_matrix[0])


    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sample tensor, label)
        """
        sample = self.data[idx]
        label = int(self.labels[idx])
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label