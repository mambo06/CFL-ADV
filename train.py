"""
Federated Contrastive Learning with Adversarial Defense (CFL-ADV)

Main training script implementing federated learning with:
- Server-client architecture
- Model aggregation (FedAvg)
- Adversarial attack simulation
- Defense mechanisms

Reference: https://papers.ssrn.com/abstract=5799977
"""

import copy
import json
import random
from itertools import islice
from pathlib import Path
from typing import List, Dict, Any

import torch
import yaml
from tqdm import tqdm

import evaluation as eval
from src.model import CFL
from utils.arguments import get_arguments, get_config
from utils.load_data import Loader
from utils.utils import update_config_with_model_dims


# ─── Server Class ───────────────────────────────────────────────────────────

class Server:
    """
    Federated Learning Server.
    
    Responsibilities:
    - Maintain global model
    - Aggregate client updates
    - Distribute global model to clients
    """
    
    def __init__(self, model: CFL, config: Dict[str, Any]) -> None:
        """
        Initialize FL server.
        
        Args:
            model: Global CFL model
            config: Configuration dictionary
        """
        self.global_model = model
        self.global_dict = self.global_model.encoder.state_dict()
        self.config = config
    
    
    def aggregate_models(self, client_models: List['Client']) -> Dict[str, torch.Tensor]:
        """
        Aggregate client models using FedAvg (mean aggregation).
        
        Args:
            client_models: List of client objects with trained models
            
        Returns:
            Aggregated model parameters
        """
        aggregated_params = {}
        
        # Average parameters across all clients
        for key in self.global_dict.keys():
            # Stack parameters from all clients
            param_stack = [
                client.get_model_params()[key].float() 
                for client in client_models
            ]
            
            # Compute mean
            aggregated_params[key] = torch.stack(param_stack).mean(dim=0)
        
        # Update global model
        self.global_dict = aggregated_params
        
        return aggregated_params
    
    
    def distribute_model(self) -> Dict[str, torch.Tensor]:
        """
        Distribute global model parameters to clients.
        
        Returns:
            Global model state dict
        """
        return self.global_dict


# ─── Client Class ───────────────────────────────────────────────────────────

class Client:
    """
    Federated Learning Client.
    
    Responsibilities:
    - Maintain local model
    - Train on local data
    - Simulate adversarial behavior (if malicious)
    """
    
    def __init__(
        self, 
        model: CFL, 
        dataloader: torch.utils.data.DataLoader, 
        client_number: int
    ) -> None:
        """
        Initialize FL client.
        
        Args:
            model: Global model to copy
            dataloader: Local training data
            client_number: Client ID
        """
        # Deep copy model to ensure independence
        self.model = copy.deepcopy(model)
        self.dataloader = dataloader
        self.client_number = client_number
        
        # Create data iterator
        self.data_iter = iter(dataloader)
        
        # Poison flag (set externally)
        self.poison = False
    
    
    def train(self) -> torch.Tensor:
        """
        Train local model on one batch.
        
        Returns:
            Total loss for the batch
        """
        # Get next batch (restart iterator if exhausted)
        try:
            x, _ = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            x, _ = next(self.data_iter)
        
        # Shuffle batch
        idx = torch.randperm(x.shape[0])
        x = x[idx].to(self.model.device)
        
        # Forward pass
        self.model.optimizer_ae.zero_grad()
        tloss, closs, rloss, zloss = self.model.fit(x)
        
        # Record losses
        self.model.loss["tloss_o"].append(tloss.item())
        self.model.loss["tloss_b"].append(tloss.item())
        self.model.loss["closs_b"].append(closs.item())
        self.model.loss["rloss_b"].append(rloss.item())
        self.model.loss["zloss_b"].append(zloss.item())
        
        # Backward pass
        tloss.backward()
        self.model.optimizer_ae.step()
        
        return tloss
    
    
    def poison_model(self, scale: float) -> None:
        """
        Apply adversarial attack by scaling model parameters.
        
        This simulates a gradient poisoning attack where malicious
        clients amplify their updates to corrupt the global model.
        
        Args:
            scale: Scaling factor for parameters
        """
        for param in self.model.encoder.parameters():
            param.data.mul_(scale)
    
    
    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """
        Get local model parameters.
        
        Returns:
            Model state dict (deep copy)
        """
        return copy.deepcopy(self.model.encoder.state_dict())
    
    
    def set_model(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Update local model with global parameters.
        
        Args:
            params: Global model state dict
        """
        self.model.encoder.load_state_dict(params)


# ─── Training Loop ──────────────────────────────────────────────────────────

def run(config: Dict[str, Any], save_weights: bool = True) -> None:
    """
    Execute federated learning training loop.
    
    Args:
        config: Configuration dictionary
        save_weights: Whether to save model weights
    """
    # ─── Setup ───
    config = copy.deepcopy(config)
    config['client'] = 0
    
    # Load dataset
    ds_loader = Loader(config, dataset_name=config["dataset"])
    config = update_config_with_model_dims(ds_loader, config)
    
    # Initialize global model and server
    global_model = CFL(config)
    server = Server(global_model, config)
    
    # ─── Initialize Clients ───
    clients = []
    
    # Select malicious clients
    n_malicious = int(config["fl_cluster"] * config['malClient'])
    poison_clients = random.sample(
        range(config["fl_cluster"]), 
        n_malicious
    ) if n_malicious > 0 else []
    
    if poison_clients:
        print(f"[WARNING] Malicious clients: {poison_clients}")
    
    # Create clients
    for client_id in range(config["fl_cluster"]):
        # Update config for this client
        prefix = (
            f"Cl-{client_id}-"
            f"{config['epochs']}e-"
            f"{config['fl_cluster']}fl-"
            f"{config['malClient']}mc-"
            f"{config['attack_type']}_at-"
            f"{config['defense_type']}_dt-"
            f"{config['randomLevel']}rl-"
            f"{config['dataset']}"
        )
        config.update({"prefix": prefix, "client": client_id})
        
        # Create client
        loader = Loader(config, dataset_name=config["dataset"]).trainFL_loader
        client = Client(global_model, loader, client_id)
        client.poison = client_id in poison_clients
        clients.append(client)
    
    # ─── Training Loop ───
    total_batches = len(loader)
    
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        
        # Progress bar for batches
        batch_iterator = tqdm(
            range(total_batches),
            desc=f"Epoch {epoch + 1}/{config['epochs']}",
            leave=True
        )
        
        for batch_idx in batch_iterator:
            # ─── Local Training ───
            for client in clients:
                # Train on local data
                loss = client.train()
                epoch_loss += loss.item()
                
                # Apply adversarial attack (if malicious)
                if client.poison:
                    client.poison_model(config['attack_scale'])
            
            # ─── Server Aggregation ───
            server.aggregate_models(clients)
            
            # ─── Distribute Global Model ───
            global_params = server.distribute_model()
            for client in clients:
                client.set_model(global_params)
            
            # Update epoch loss in progress bar
            avg_loss = epoch_loss / ((batch_idx + 1) * config["fl_cluster"])
            batch_iterator.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # ─── Epoch Summary ───
        avg_epoch_loss = epoch_loss / (config["fl_cluster"] * total_batches)
        print(f"[INFO] Epoch {epoch + 1} - Avg Loss: {avg_epoch_loss:.4f}")
        
        # Record epoch loss for each client
        for client in clients:
            batch_losses = client.model.loss["tloss_b"][-total_batches:]
            epoch_avg = sum(batch_losses) / total_batches
            client.model.loss["tloss_e"].append(epoch_avg)
    
    # ─── Save Results ───
    for client_id, client in enumerate(clients):
        model = client.model
        
        # Save training parameters
        model.saveTrainParams(client_id)
        
        # Save model weights
        if save_weights:
            model.save_weights(client_id)
        
        # Save configuration
        config_path = Path(model._results_path) / f"config_{config['prefix']}.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print("[INFO] Training completed successfully")


# ─── Main Entry Point ───────────────────────────────────────────────────────

def main(config: Dict[str, Any]) -> None:
    """
    Main execution function.
    
    Args:
        config: Configuration dictionary
    """
    # ─── Load Dataset Info ───
    config["framework"] = config["dataset"]
    info_path = Path(f'data/{config["dataset"]}/info.json')
    
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    
    info = json.loads(info_path.read_text())
    
    # Update config with dataset info
    config.update({
        'task_type': info['task_type'],
        'cat_policy': info['cat_policy'],
        'norm': info['norm'],
        'learning_rate_reducer': config['learning_rate']
    })
    
    # ─── Run Training ───
    print(f"\n{'=' * 80}")
    print(f"Starting CFL-ADV Training")
    print(f"{'=' * 80}")
    print(f"Dataset: {config['dataset']}")
    print(f"Clients: {config['fl_cluster']}")
    print(f"Malicious: {int(config['fl_cluster'] * config['malClient'])}")
    print(f"Attack: {config['attack_type']}")
    print(f"Defense: {config['defense_type']}")
    print(f"Epochs: {config['epochs']}")
    print(f"{'=' * 80}\n")
    
    run(config, save_weights=True)
    
    # ─── Evaluate ───
    print(f"\n{'=' * 80}")
    print(f"Starting Evaluation")
    print(f"{'=' * 80}\n")
    
    eval.main(copy.deepcopy(config))
    
    print(f"\n{'=' * 80}")
    print(f"Experiment Completed")
    print(f"{'=' * 80}\n")


# ─── Standalone Execution ───────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse arguments
    args = get_arguments()
    config = get_config(args)
    
    # Run experiment
    main(config)
