"""
Evaluation script for federated learning models.
Evaluates both original data and learned embeddings using logistic regression.
Reference: https://github.com/AstraZeneca/SubTab
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch as th
from tqdm import tqdm

from src.model import CFL
from utils.eval_utils import (
    linear_model_eval,
    plot_clusters,
    append_tensors_to_lists,
    concatenate_lists,
    aggregate
)
from utils.load_data import Loader
from utils.utils import set_dirs, update_config_with_model_dims


# Set random seed for reproducibility
th.manual_seed(1)


# ─── Helper Functions ───────────────────────────────────────────────────────

def build_experiment_tag(config: Dict[str, Any], client: int) -> str:
    """
    Build consistent experiment identifier string.
    
    Args:
        config: Configuration dictionary
        client: Client ID
        
    Returns:
        Experiment tag string
    """
    return (
        f"Cl-{client}-"
        f"{config['epochs']}e-"
        f"{config['fl_cluster']}fl-"
        f"{config['malClient']}mc-"
        f"{config['attack_type']}_at-"
        f"{config['defense_type']}_dt-"
        f"{config['randomLevel']}rl-"
        f"{config['dataset']}"
    )


def split_train_test(
    z: np.ndarray,
    labels: np.ndarray,
    train_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Args:
        z: Feature array
        labels: Label array
        train_ratio: Ratio of training data
        
    Returns:
        Tuple of (z_train, z_test, y_train, y_test)
    """
    n_train = int(len(z) * train_ratio)
    
    z_train = z[:n_train]
    z_test = z[n_train:]
    y_train = labels[:n_train]
    y_test = labels[n_train:]
    
    return z_train, z_test, y_train, y_test


# ─── Evaluation Functions ───────────────────────────────────────────────────

def evaluate_embeddings(
    data_loader: Loader,
    model: CFL,
    config: Dict[str, Any],
    client: int,
    plot_suffix: str = "_Test",
    mode: str = "train",
    z_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    nData: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluate learned embeddings using logistic regression.
    
    Args:
        data_loader: Data loader instance
        model: CFL model instance
        config: Configuration dictionary
        client: Client ID
        plot_suffix: Suffix for plot filenames
        mode: Evaluation mode ('train' or 'test')
        z_train: Pre-computed training embeddings (optional)
        y_train: Pre-computed training labels (optional)
        nData: Dataset identifier (optional)
        
    Returns:
        Evaluation results dictionary if mode=='test', else (z, labels) tuple
    """
    # Get encoder and set to evaluation mode
    encoder = model.encoder
    encoder.to(config["device"])
    encoder.eval()
    
    # Select data loader
    data_loader_selected = data_loader.trainFL_loader
    
    # Progress bar
    train_tqdm = tqdm(
        enumerate(data_loader_selected),
        total=len(data_loader_selected),
        desc=f"Extracting embeddings ({mode})",
        leave=True
    )
    
    # Storage for embeddings and labels
    z_list, labels_list = [], []
    
    # Extract embeddings
    with th.no_grad():
        for i, (x, label) in train_tqdm:
            # Generate subsets
            x_tilde_list = model.subset_generator(x)
            
            # Extract latent representations for each subset
            latent_list = []
            for xi in x_tilde_list:
                Xbatch = model._tensor(xi)
                _, latent, _ = encoder(Xbatch)
                latent_list.append(latent)
            
            # Aggregate latent representations
            latent = aggregate(latent_list, config)
            
            # Collect results
            z_list, labels_list = append_tensors_to_lists(
                [z_list, labels_list],
                [latent, label.int()]
            )
    
    # Concatenate results
    z = concatenate_lists([z_list])
    clabels = concatenate_lists([labels_list])
    
    # Split into train/test
    z_train, z_test, y_train, y_test = split_train_test(
        z, clabels, config["training_data_ratio"]
    )
    
    # Evaluate on test set
    if mode == "test":
        suffix = f"-Dataset-{nData}" if nData else ""
        description = "Sweeping C parameter. Smaller C values specify stronger regularization:"
        
        return linear_model_eval(
            config,
            z_train,
            y_train,
            f"Client-{client}{suffix}-contrastive-",
            z_test=z_test,
            y_test=y_test,
            description=description,
            nData=nData
        )
    else:
        return z, clabels


def evaluate_original(
    data_loader: Loader,
    config: Dict[str, Any],
    client: int,
    plot_suffix: str = "_Test",
    mode: str = "train",
    z_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    nData: Optional[str] = None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Evaluate original data (without embeddings) using logistic regression.
    
    Args:
        data_loader: Data loader instance
        config: Configuration dictionary
        client: Client ID
        plot_suffix: Suffix for plot filenames
        mode: Evaluation mode ('train' or 'test')
        z_train: Pre-computed training data (optional)
        y_train: Pre-computed training labels (optional)
        nData: Dataset identifier (optional)
        
    Returns:
        Evaluation results if mode=='test', else (data, labels) tuple
    """
    # Select data loader
    data_loader_selected = data_loader.trainNS_loader
    
    # Progress bar
    train_tqdm = tqdm(
        enumerate(data_loader_selected),
        total=len(data_loader_selected),
        desc=f"Loading original data ({mode})",
        leave=True
    )
    
    # Storage for data and labels
    z_list, labels_list = [], []
    
    # Collect data
    for i, (x, label) in train_tqdm:
        z_list, labels_list = append_tensors_to_lists(
            [z_list, labels_list],
            [x, label.int()]
        )
    
    # Concatenate results
    z = concatenate_lists([z_list])
    clabels = concatenate_lists([labels_list])
    
    # Split into train/test
    z_train, z_test, y_train, y_test = split_train_test(
        z, clabels, config["training_data_ratio"]
    )
    
    # Evaluate on test set
    if mode == "test":
        suffix = f"-Dataset-{nData}" if nData else ""
        if config.get("baseGlobal", False):
            suffix += "-baseGlobal"
        
        description = "Sweeping C parameter. Smaller C values specify stronger regularization:"
        
        linear_model_eval(
            config,
            z_train,
            y_train,
            f"Client-{client}{suffix}-original-",
            z_test=z_test,
            y_test=y_test,
            description=description,
            nData=nData
        )
    else:
        return z, clabels


def eval(
    data_loader: Loader,
    config: Dict[str, Any],
    nData: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Main evaluation wrapper function.
    
    Args:
        data_loader: Data loader instance
        config: Configuration dictionary
        nData: List of dataset identifiers (optional)
        
    Returns:
        Evaluation results dictionary
    """
    client = config["client"]
    prefix = build_experiment_tag(config, client)
    config["prefix"] = prefix
    
    # Instantiate and load model
    model = CFL(config)
    model.load_models(client)
    
    results = None
    
    with th.no_grad():
        # Evaluate original data (if not flOnly mode)
        if not config.get("flOnly", False):
            print(f"\n[INFO] Evaluating original dataset for client {client}")
            
            if nData:
                for item in nData:
                    evaluate_original(
                        data_loader, config, client,
                        plot_suffix="test", mode="test",
                        nData=item
                    )
            else:
                evaluate_original(
                    data_loader, config, client,
                    plot_suffix="test", mode="test"
                )
            
            print(f"[INFO] Original data evaluation complete")
        
        # Evaluate learned embeddings
        print(f"\n[INFO] Evaluating learned embeddings for client {client}")
        
        if nData:
            for item in nData:
                results = evaluate_embeddings(
                    data_loader, model, config, client,
                    plot_suffix="test", mode="test",
                    nData=item
                )
        else:
            results = evaluate_embeddings(
                data_loader, model, config, client,
                plot_suffix="test", mode="test"
            )
        
        print(f"[INFO] Embeddings evaluation complete")
        print("=" * 100)
    
    return results


# ─── Main Entry Point ───────────────────────────────────────────────────────

def main(config: Dict[str, Any], nData: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Main evaluation function.
    
    Args:
        config: Configuration dictionary
        nData: List of dataset identifiers (optional)
        
    Returns:
        Evaluation results dictionary
    """
    config = copy.deepcopy(config)
    
    # Setup directories
    set_dirs(config)
    
    # Load data
    ds_loader = Loader(
        config,
        dataset_name=config["dataset"],
        drop_last=False
    )
    
    # Update config with model dimensions
    config = update_config_with_model_dims(ds_loader, config)
    
    # Run evaluation
    return eval(ds_loader, config, nData)


# ─── Standalone Execution ───────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.arguments import get_arguments, get_config, print_config_summary
    import mlflow
    from utils.utils import run_with_profiler
    
    # Parse arguments
    args = get_arguments()
    config = get_config(args)
    
    # Configure for evaluation
    config["framework"] = config["dataset"]
    config["validate"] = False
    config["add_noise"] = False
    
    # Optional: print configuration
    if config.get("verbose"):
        print_config_summary(config, args)
    
    # Run with or without MLFlow tracking
    if config.get("mlflow", False):
        experiment_name = f"Evaluation_{config['dataset']}_{args.experiment}"
        mlflow.set_experiment(experiment_name=experiment_name)
        
        with mlflow.start_run():
            if config.get("profile", False):
                run_with_profiler(main, config)
            else:
                main(config)
    else:
        if config.get("profile", False):
            run_with_profiler(main, config)
        else:
            main(config)