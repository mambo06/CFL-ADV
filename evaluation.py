"""
Federated Learning Evaluation Script
Evaluates trained models across all federated clients.
Reference: https://github.com/AstraZeneca/SubTab
"""

import copy
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

import _eval as eval
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.utils import update_config_with_model_dims


# ─── Helpers ────────────────────────────────────────────────────────────────

def build_experiment_tag(config: Dict[str, Any], client_id: int) -> str:
    """Build a consistent experiment identifier string."""
    return (
        f"Cl-{client_id}-"
        f"{config['epochs']}e-"
        f"{config['fl_cluster']}fl-"
        f"{config['malClient']}mc-"
        f"{config['attack_type']}_at-"
        f"{config['defense_type']}_dt-"
        f"{config['randomLevel']}rl-"
        f"{config['dataset']}"
    )


def load_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Load dataset metadata from info.json."""
    info_path = Path(f"data/{dataset_name}/info.json")
    return json.loads(info_path.read_text())


def configure_for_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply evaluation-specific configuration overrides."""
    eval_config = copy.deepcopy(config)
    
    # Evaluation mode settings
    eval_config["add_noise"] = False
    eval_config["validate"] = False
    
    # Optional: uncomment to use full training set for evaluation
    # eval_config["training_data_ratio"] = 1.0
    
    return eval_config


# ─── Main ───────────────────────────────────────────────────────────────────

def main(config: Dict[str, Any]) -> None:
    """
    Evaluate trained federated learning models across all clients.
    
    Args:
        config: Configuration dictionary containing experiment parameters
    """
    config = configure_for_evaluation(config)
    
    results = []
    
    for client_id in range(config["fl_cluster"]):
        config["client"] = client_id
        config["prefix"] = build_experiment_tag(config, client_id)
        
        print(f"\n[INFO] Evaluating client {client_id}/{config['fl_cluster']-1}")
        print(f"[INFO] Tag: {config['prefix']}")
        
        # Evaluate single client
        client_results = eval.main(copy.deepcopy(config))
        results.append(client_results)
        
        # Optional: extract and aggregate specific metrics
        # if client_results and 'test_acc' in client_results[0]:
        #     results.append(client_results[0]['test_acc'][2])
    
    # Optional: compute and display aggregate statistics
    # if results:
    #     print(f"\n[SUMMARY] Mean test accuracy: {np.mean(results):.4f}")
    #     print(f"[SUMMARY] Std test accuracy:  {np.std(results):.4f}")
    
    print(f"\n[INFO] Evaluation complete for {len(results)} clients")


def setup_config(args) -> Dict[str, Any]:
    """
    Initialize and prepare configuration for evaluation.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Configured dictionary ready for evaluation
    """
    config = get_config(args)
    
    # Framework and dataset setup
    config["framework"] = config["dataset"]
    
    # Load dataset metadata
    dataset_info = load_dataset_info(config["dataset"])
    config.update({
        "task_type": dataset_info["task_type"],
        "cat_policy": dataset_info["cat_policy"],
        "norm": dataset_info["norm"],
    })
    
    # Preserve learning rate
    config.setdefault("learning_rate_reducer", config["learning_rate"])
    
    # Preserve original autoencoder dimensions if they were modified during training
    if "dims_original" in config:
        config["dims"] = config["dims_original"]
    
    return config


if __name__ == "__main__":
    args = get_arguments()
    config = setup_config(args)
    
    # Optional: print configuration summary
    if config.get("verbose"):
        print_config_summary(config, args)
    
    main(config)