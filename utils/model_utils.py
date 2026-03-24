"""
Neural network architectures for autoencoder-based federated learning.
Implements encoder-decoder with projection head for contrastive learning.
Reference: https://github.com/AstraZeneca/SubTab
"""

import copy
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─── Main Autoencoder Wrapper ───────────────────────────────────────────────

class AEWrapper(nn.Module):
    """
    Autoencoder wrapper with projection head for contrastive learning.
    
    Architecture:
        Input → Encoder → Latent → Projection Head → z
                       ↓
                    Decoder → Reconstruction
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        """
        Initialize autoencoder wrapper.
        
        Args:
            options: Configuration dictionary containing:
                - dims: List of layer dimensions
                - p_norm: Norm type for normalization (default: 2)
                - normalize: Whether to normalize projection output
                - shallow_architecture: Use shallow encoder/decoder
        """
        super(AEWrapper, self).__init__()
        
        self.options = options
        
        # Encoder and decoder
        self.encoder = ShallowEncoder(options)
        self.decoder = ShallowDecoder(options)
        
        # Projection head dimensions
        output_dim = options["dims"][-1]
        
        # Two-layer projection network for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LeakyReLU(inplace=False),
            nn.Linear(output_dim, output_dim)
        )


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor (batch_size × input_dim)
            
        Returns:
            Tuple of (z, latent, x_recon) where:
                - z: Projected representation for contrastive learning
                - latent: Encoder output (bottleneck representation)
                - x_recon: Reconstructed input
        """
        # Encode input to latent representation
        latent = self.encoder(x)
        
        # Project latent to contrastive space
        z = self.projection_head(latent)
        
        # Optionally normalize projection
        if self.options.get("normalize", False):
            z = F.normalize(z, p=self.options.get("p_norm", 2), dim=1)
        
        # Decode latent to reconstruct input
        x_recon = self.decoder(latent)
        
        return z, latent, x_recon


# ─── Encoder ────────────────────────────────────────────────────────────────

class ShallowEncoder(nn.Module):
    """
    Shallow encoder for tabular data.
    
    Maps input features to latent representation through
    a series of fully connected layers.
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        """
        Initialize shallow encoder.
        
        Args:
            options: Configuration dictionary containing:
                - dims: List of layer dimensions [input_dim, hidden1, ..., latent_dim]
                - n_subsets: Number of feature subsets (for federated learning)
                - overlap: Overlap ratio between subsets
        """
        super(ShallowEncoder, self).__init__()
        
        # Deep copy to avoid modifying original config
        self.options = copy.deepcopy(options)
        
        # Calculate subset dimensions for federated learning
        n_column_subset = self.options["dims"][0]
        overlap = self.options.get("overlap", 0.0)
        n_overlap = int(overlap * n_column_subset)
        
        # Update input dimension
        self.options["dims"][0] = n_column_subset
        
        # Build hidden layers
        self.hidden_layers = HiddenLayers(self.options)


    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor (batch_size × input_dim)
            
        Returns:
            Latent representation (batch_size × latent_dim)
        """
        return self.hidden_layers(x)


# ─── Decoder ────────────────────────────────────────────────────────────────

class ShallowDecoder(nn.Module):
    """
    Shallow decoder for tabular data.
    
    Maps latent representation back to input space for reconstruction.
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        """
        Initialize shallow decoder.
        
        Args:
            options: Configuration dictionary containing:
                - dims: List of layer dimensions
        """
        super(ShallowDecoder, self).__init__()
        
        self.options = copy.deepcopy(options)
        
        # Decoder is a single linear layer (shallow architecture)
        input_dim = self.options["dims"][-1]   # Latent dimension
        output_dim = self.options["dims"][0]   # Input dimension
        
        self.decoder_layer = nn.Linear(input_dim, output_dim)


    def forward(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent representation (batch_size × latent_dim)
            
        Returns:
            Reconstructed input (batch_size × input_dim)
        """
        return self.decoder_layer(z)


# ─── Hidden Layers Builder ──────────────────────────────────────────────────

class HiddenLayers(nn.Module):
    """
    Dynamically builds a sequence of hidden layers with optional
    batch normalization and dropout.
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        """
        Initialize hidden layers.
        
        Args:
            options: Configuration dictionary containing:
                - dims: List of layer dimensions
                - isBatchNorm: Whether to use batch normalization
                - isDropout: Whether to use dropout
                - dropout_rate: Dropout probability
        """
        super(HiddenLayers, self).__init__()
        
        self.layers = nn.ModuleList()
        dims = options["dims"]
        
        # Build layers from dims[0] to dims[-2]
        # (last layer is handled separately in encoder/decoder)
        for i in range(1, len(dims) - 1):
            # Linear layer
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            
            # Optional batch normalization
            if options.get("isBatchNorm", False):
                self.layers.append(nn.BatchNorm1d(dims[i]))
            
            # Activation function
            self.layers.append(nn.LeakyReLU(inplace=False))
            
            # Optional dropout
            if options.get("isDropout", False):
                dropout_rate = options.get("dropout_rate", 0.5)
                self.layers.append(nn.Dropout(dropout_rate))


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through hidden layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after all hidden layers
        """
        for layer in self.layers:
            x = layer(x)
        return x


# ─── Deep Encoder/Decoder (Alternative) ─────────────────────────────────────

class DeepEncoder(nn.Module):
    """
    Deep encoder with explicit latent layer.
    Alternative to ShallowEncoder for more complex architectures.
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        """
        Initialize deep encoder.
        
        Args:
            options: Configuration dictionary
        """
        super(DeepEncoder, self).__init__()
        
        self.options = copy.deepcopy(options)
        
        # Calculate subset dimensions
        n_column_subset = self.options["dims"][0]
        overlap = self.options.get("overlap", 0.0)
        n_overlap = int(overlap * n_column_subset)
        
        # Update input dimension
        self.options["dims"][0] = n_column_subset + n_overlap
        
        # Hidden layers
        self.hidden_layers = HiddenLayers(self.options)
        
        # Explicit latent layer
        self.latent_layer = nn.Linear(
            self.options["dims"][-2],
            self.options["dims"][-1]
        )


    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        h = self.hidden_layers(x)
        latent = self.latent_layer(h)
        return latent


class DeepDecoder(nn.Module):
    """
    Deep decoder with multiple hidden layers.
    Alternative to ShallowDecoder for more complex architectures.
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        """
        Initialize deep decoder.
        
        Args:
            options: Configuration dictionary
        """
        super(DeepDecoder, self).__init__()
        
        self.options = copy.deepcopy(options)
        
        # Adjust output dimension based on reconstruction target
        if (self.options.get("reconstruction", False) and 
            self.options.get("reconstruct_subset", False)):
            n_column_subset = self.options["dims"][0]
            self.options["dims"][0] = n_column_subset
        
        # Reverse dimensions for decoder
        self.options["dims"] = self.options["dims"][::-1]
        
        # Hidden layers
        self.hidden_layers = HiddenLayers(self.options)
        
        # Output layer
        self.output_layer = nn.Linear(
            self.options["dims"][-2],
            self.options["dims"][-1]
        )


    def forward(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed input
        """
        h = self.hidden_layers(z)
        logits = self.output_layer(h)
        return logits


# ─── Utility Functions ──────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """
    Print summary of model architecture.
    
    Args:
        model: PyTorch model
        model_name: Name to display
    """
    print("=" * 80)
    print(f"{model_name} Architecture")
    print("=" * 80)
    print(model)
    print("-" * 80)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 80)


def initialize_weights(model: nn.Module, init_type: str = "xavier") -> None:
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        init_type: Initialization type ('xavier', 'kaiming', 'normal')
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
            elif init_type == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# ─── Model Factory ──────────────────────────────────────────────────────────

def create_autoencoder(
    options: Dict[str, Any],
    architecture: str = "shallow"
) -> AEWrapper:
    """
    Factory function to create autoencoder with specified architecture.
    
    Args:
        options: Configuration dictionary
        architecture: Architecture type ('shallow' or 'deep')
        
    Returns:
        Configured AEWrapper instance
    """
    if architecture == "shallow":
        return AEWrapper(options)
    elif architecture == "deep":
        # Create custom wrapper with deep encoder/decoder
        options_copy = copy.deepcopy(options)
        wrapper = AEWrapper(options_copy)
        wrapper.encoder = DeepEncoder(options_copy)
        wrapper.decoder = DeepDecoder(options_copy)
        return wrapper
    else:
        raise ValueError(f"Unknown architecture: {architecture}")