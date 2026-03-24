"""
Contrastive Federated Learning (CFL) Model
Trains an Autoencoder with projection network using SubTab framework.
Reference: https://github.com/AstraZeneca/SubTab
"""

import gc
import itertools
import os
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import torch as th
from torch import Tensor

from utils.loss_functionsV2 import JointLoss
from utils.model_plot import save_loss_plot
from utils.model_utils import AEWrapper
from utils.utils import set_seed, set_dirs

# Enable anomaly detection for debugging
th.autograd.set_detect_anomaly(True)


# ─── Main Model Class ───────────────────────────────────────────────────────

class CFL:
    """
    Contrastive Federated Learning model with autoencoder and projection network.
    
    Implements SubTab framework for self-supervised learning on tabular data
    with support for federated learning scenarios.
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        """
        Initialize CFL model with configuration.
        
        Args:
            options: Configuration dictionary containing model hyperparameters
        """
        self.options = options
        self.device = options["device"]
        
        # Initialize model storage
        self.model_dict: Dict[str, th.nn.Module] = {}
        self.summary: Dict[str, List] = {}
        
        # Set random seed for reproducibility
        set_seed(self.options)
        
        # Setup paths and directories
        self._set_paths()
        set_dirs(self.options)
        
        # Determine if we need subset combinations
        self.is_combination = (
            self.options["contrastive_loss"] or 
            self.options["distance_loss"]
        )
        
        # Build model components
        print("[INFO] Building CFL model components...")
        self.set_autoencoder()
        self._set_scheduler()
        
        # Initialize loss tracking
        self.loss = {
            "tloss_b": [],  # Total loss per batch
            "tloss_e": [],  # Total loss per epoch
            "vloss_e": [],  # Validation loss per epoch
            "closs_b": [],  # Contrastive loss per batch
            "rloss_b": [],  # Reconstruction loss per batch
            "zloss_b": [],  # Latent distance loss per batch
            "tloss_o": []   # Original total loss
        }
        
        self.train_tqdm = None
        
        # Fisher information for continual learning (optional)
        self.fisher_dict: Optional[Dict[str, Tensor]] = None
        self.optpar_dict: Optional[Dict[str, Tensor]] = None


    # ─── Model Setup ────────────────────────────────────────────────────────

    def set_autoencoder(self) -> None:
        """Initialize autoencoder, optimizer, and loss function."""
        # Create autoencoder wrapper
        self.encoder = AEWrapper(self.options)
        self.model_dict["encoder"] = self.encoder
        
        # Move models to device
        for model in self.model_dict.values():
            model.to(self.device)
        
        # Setup joint loss function
        self.joint_loss = JointLoss(self.options)
        
        # Setup optimizer
        parameters = [model.parameters() for model in self.model_dict.values()]
        self.optimizer_ae = self._adam(parameters, lr=self.options["learning_rate"])
        
        # Initialize summary
        self.summary["recon_loss"] = []


    def _set_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        self.scheduler = th.optim.lr_scheduler.StepLR(
            self.optimizer_ae,
            step_size=1,
            gamma=0.99
        )


    # ─── Training ───────────────────────────────────────────────────────────

    def fit(self, data_loader: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform one training step.
        
        Args:
            data_loader: Input batch tensor
            
        Returns:
            Tuple of (total_loss, contrastive_loss, recon_loss, latent_loss)
        """
        x = data_loader.to(self.device)
        self.set_mode(mode="training")
        
        # Store original data for reconstruction target
        Xorig = self.process_batch(x, x)
        
        # Generate augmented subsets
        x_tilde_list = self.subset_generator(x, mode="train")
        
        # Create combinations if needed for contrastive/distance loss
        if self.is_combination:
            x_tilde_list = self.get_combinations_of_subsets(x_tilde_list)
        
        # Compute losses
        tloss, closs, rloss, zloss = self.calculate_loss(x_tilde_list, Xorig)
        
        return tloss, closs, rloss, zloss


    def calculate_loss(
        self,
        x_tilde_list: List[Tensor],
        Xorig: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculate losses across all subsets.
        
        Args:
            x_tilde_list: List of augmented input subsets
            Xorig: Original input for reconstruction target
            
        Returns:
            Tuple of averaged (total, contrastive, reconstruction, latent) losses
        """
        total_loss, contrastive_loss, recon_loss, zrecon_loss = [], [], [], []
        
        for xi in x_tilde_list:
            # Prepare input
            Xinput = xi if self.is_combination else self.process_batch(xi, xi)
            Xinput = Xinput.to(self.device).float()
            
            # Forward pass
            z, latent, Xrecon = self.encoder(Xinput)
            
            # Determine reconstruction target
            if self.options["reconstruction"] and self.options["reconstruct_subset"]:
                target = Xinput
            else:
                target = Xorig
            
            # Compute losses
            tloss, closs, rloss, zloss = self.joint_loss(z, Xrecon, target)
            
            # Accumulate
            total_loss.append(tloss)
            contrastive_loss.append(closs)
            recon_loss.append(rloss)
            zrecon_loss.append(zloss)
        
        # Average losses
        n = len(total_loss)
        return (
            sum(total_loss) / n,
            sum(contrastive_loss) / n,
            sum(recon_loss) / n,
            sum(zrecon_loss) / n
        )


    def update_autoencoder(self, tloss: Tensor, retain_graph: bool = True) -> None:
        """
        Backpropagate and update autoencoder parameters.
        
        Args:
            tloss: Total loss tensor
            retain_graph: Whether to retain computation graph
        """
        self._update_model(tloss, self.optimizer_ae, retain_graph=retain_graph)


    # ─── Validation ─────────────────────────────────────────────────────────

    def validate_train(
        self,
        client: int,
        epoch: int,
        total_batches: int,
        validation_loader: Tensor
    ) -> float:
        """
        Validate model during training.
        
        Args:
            client: Client ID
            epoch: Current epoch number
            total_batches: Total number of batches
            validation_loader: Validation data
            
        Returns:
            Validation loss (0.0 if validation is disabled)
        """
        if epoch % self.options["nth_epoch"] == 0 and self.options["validate"]:
            return self.validate(validation_loader, total_batches)
        return 0.0


    def validate(self, validation_loader: Tensor, total_batches: int) -> float:
        """
        Compute validation loss.
        
        Args:
            validation_loader: Validation data tensor
            total_batches: Number of batches (unused but kept for compatibility)
            
        Returns:
            Average validation loss
        """
        with th.no_grad():
            x = validation_loader
            x_tilde_list = self.subset_generator(x)
            
            if self.is_combination:
                x_tilde_list = self.get_combinations_of_subsets(x_tilde_list)
            
            Xorig = self.process_batch(x, x)
            val_losses = []
            
            for xi in x_tilde_list:
                Xinput = xi if self.is_combination else self.process_batch(xi, xi)
                
                # Forward pass
                z, latent, Xrecon = self.encoder(Xinput)
                
                # Compute loss
                val_loss, _, _, _ = self.joint_loss(z, Xrecon, Xorig)
                val_losses.append(val_loss)
            
            return sum(val_losses) / len(val_losses)


    # ─── Data Augmentation ──────────────────────────────────────────────────

    def subset_generator(
        self,
        x: Tensor,
        mode: str = "test",
        skip: List[int] = None
    ) -> List[Tensor]:
        """
        Generate augmented subsets with optional noise.
        
        Args:
            x: Input tensor
            mode: "train" or "test" mode
            skip: Indices to skip (unused)
            
        Returns:
            List of augmented subset tensors
        """
        n_subsets = self.options["n_subsets"]
        x_tilde_list = []
        
        # Alternate masking ratio for diversity
        use_high_mask = True
        
        for _ in range(n_subsets):
            x_bar = x.clone()
            
            # Add noise if enabled
            if self.options["add_noise"]:
                x_bar_noisy = self.generate_noisy_xbar(x_bar).to(self.device)
                
                # Apply masking
                p_m = self.options["masking_ratio"]
                if not use_high_mask:
                    p_m = 1 - p_m
                
                mask = th.bernoulli(th.full(x_bar.shape, p_m)).to(self.device)
                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask
            
            x_tilde_list.append(x_bar)
            use_high_mask = not use_high_mask
        
        return x_tilde_list


    def generate_noisy_xbar(self, x: Tensor) -> Tensor:
        """
        Generate noisy version of input.
        
        Args:
            x: Input tensor
            
        Returns:
            Noisy tensor based on configured noise type
        """
        no, dim = x.shape
        noise_type = self.options["noise_type"]
        noise_level = self.options["noise_level"]
        
        if noise_type == "swap_noise":
            x_bar = th.zeros_like(x)
            for i in range(dim):
                idx = th.randperm(no)
                x_bar[:, i] = x[idx, i]
        
        elif noise_type == "gaussian_noise":
            x_bar = x + th.normal(
                float(th.mean(x)),
                noise_level,
                size=x.shape
            )
        
        else:
            x_bar = th.zeros_like(x)
        
        return x_bar


    def get_combinations_of_subsets(self, x_tilde_list: List[Tensor]) -> List[Tensor]:
        """
        Create pairwise combinations of subsets.
        
        Args:
            x_tilde_list: List of subset tensors
            
        Returns:
            List of concatenated subset pairs
        """
        subset_combinations = list(itertools.combinations(x_tilde_list, 2))
        return [
            self.process_batch(xi, xj)
            for xi, xj in subset_combinations
        ]


    # ─── Utilities ──────────────────────────────────────────────────────────

    def process_batch(self, xi: Tensor, xj: Tensor) -> Tensor:
        """
        Concatenate two tensors and move to device.
        
        Args:
            xi: First tensor
            xj: Second tensor
            
        Returns:
            Concatenated tensor on device
        """
        Xbatch = th.cat((xi, xj), axis=0)
        return self._tensor(Xbatch)


    def set_mode(self, mode: str = "training") -> None:
        """
        Set model mode (train/eval).
        
        Args:
            mode: "training" or "evaluation"
        """
        for model in self.model_dict.values():
            if mode == "training":
                model.train()
            else:
                model.eval()


    def update_log(self, client: int, epoch: int, batch: int) -> None:
        """
        Update training progress display.
        
        Args:
            client: Client ID
            epoch: Current epoch
            batch: Current batch
        """
        if epoch < 1:
            desc = f"Losses - Total:{self.loss['tloss_b'][-1]:.4f}"
            desc += f", Recon:{self.loss['rloss_b'][-1]:.4f}"
            if self.options["contrastive_loss"]:
                desc += f", Contrast:{self.loss['closs_b'][-1]:.4f}"
            if self.options["distance_loss"]:
                desc += f", Z-dist:{self.loss['zloss_b'][-1]:.6f}"
        else:
            desc = f"Epoch {epoch} - Train:{self.loss['tloss_e'][-1]:.4f}"
            if self.options["validate"]:
                desc += f", Val:{self.loss['vloss_e'][-1]:.4f}"
        
        if self.train_tqdm:
            self.train_tqdm.set_description(desc)


    # ─── Persistence ────────────────────────────────────────────────────────

    def save_weights(self, client: int) -> None:
        """
        Save model weights to disk.
        
        Args:
            client: Client ID for filename
        """
        prefix = self.options["prefix"]
        
        for model_name, model in self.model_dict.items():
            save_path = os.path.join(
                self._model_path,
                f"{model_name}_{prefix}.pth"
            )
            th.save(model.state_dict(), save_path)
        
        print(f"[INFO] Model weights saved for client {client}")


    def load_models(self, client: int) -> None:
        """
        Load model weights from disk.
        
        Args:
            client: Client ID for filename
        """
        prefix = self.options["prefix"]
        
        for model_name, model in self.model_dict.items():
            load_path = os.path.join(
                self._model_path,
                f"{model_name}_{prefix}.pth"
            )
            model.load_state_dict(
                th.load(load_path, map_location=self.device)
            )
            model.eval()
            print(f"[INFO] Loaded {model_name} for client {client}")


    def saveTrainParams(self, client: int) -> None:
        """
        Save training metrics and plots.
        
        Args:
            client: Client ID for filename
        """
        prefix = self.options["prefix"]
        
        # Save loss plot
        save_loss_plot(self.loss, self._plots_path, prefix)
        
        # Save loss dataframe
        loss_df = pd.DataFrame({
            k: pd.Series(v, dtype="float")
            for k, v in self.loss.items()
        })
        
        loss_path = os.path.join(self._loss_path, f"{prefix}-losses.csv")
        loss_df.to_csv(loss_path)
        
        print(f"[INFO] Training parameters saved for client {client}")


    # ─── Continual Learning Support ─────────────────────────────────────────

    def on_task_update(self, data_loader) -> None:
        """
        Compute Fisher information for continual learning.
        
        Args:
            data_loader: Data loader for computing gradients
        """
        self.fisher_dict = {}
        self.optpar_dict = {}
        
        for data, _ in data_loader:
            self.optimizer_ae.zero_grad()
            
            tloss, _, _, _ = self.fit(data)
            tloss.backward()
        
        # Store Fisher information and optimal parameters
        for name, param in self.encoder.named_parameters():
            self.optpar_dict[name] = param.data.clone()
            
            if param.grad is not None:
                self.fisher_dict[name] = param.grad.data.clone().pow(2)
            else:
                print(f"[WARNING] No gradient for parameter: {name}")
                self.fisher_dict[name] = th.zeros_like(param.data)


    # ─── Properties ─────────────────────────────────────────────────────────

    def get_loss(self) -> Dict[str, List[float]]:
        """Get loss history."""
        return self.loss


    def set_loss(self, loss: Dict[str, List[float]]) -> None:
        """Set loss history."""
        self.loss = loss


    # ─── Private Methods ────────────────────────────────────────────────────

    def _update_model(
        self,
        loss: Tensor,
        optimizer: th.optim.Optimizer,
        retain_graph: bool = True
    ) -> None:
        """
        Perform backpropagation and update model.
        
        Args:
            loss: Loss tensor with computation graph
            optimizer: Optimizer to use
            retain_graph: Whether to retain computation graph
        """
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        th.cuda.empty_cache()


    def _set_paths(self) -> None:
        """Setup paths for saving results."""
        base = os.path.join(
            self.options["paths"]["results"],
            self.options["framework"]
        )
        
        training_base = os.path.join(base, "training", self.options["model_mode"])
        
        self._results_path = base
        self._model_path = os.path.join(training_base, "model")
        self._plots_path = os.path.join(training_base, "plots")
        self._loss_path = os.path.join(training_base, "loss")


    def _adam(
        self,
        params: List,
        lr: float = 1e-4
    ) -> th.optim.AdamW:
        """
        Create AdamW optimizer.
        
        Args:
            params: List of parameter groups
            lr: Learning rate
            
        Returns:
            AdamW optimizer
        """
        return th.optim.AdamW(
            itertools.chain(*params),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-07
        )


    def _tensor(self, data: Tensor) -> Tensor:
        """
        Move tensor to device and convert to float.
        
        Args:
            data: Input tensor
            
        Returns:
            Tensor on device as float
        """
        return data.to(self.device).float()


    def print_model_summary(self) -> None:
        """Print model architecture summary."""
        print("=" * 100)
        print(f"CFL Model Architecture - {self.options['model_mode'].upper()}")
        print("=" * 100)
        print(self.encoder)
        print("=" * 100)


    def clean_up_memory(self, losses: List[Tensor]) -> None:
        """
        Clean up memory by deleting loss tensors.
        
        Args:
            losses: List of loss tensors to delete
        """
        for loss in losses:
            del loss
        gc.collect()