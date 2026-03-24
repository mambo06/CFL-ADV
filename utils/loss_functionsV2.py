"""
Loss functions for federated contrastive learning.
Combines reconstruction, contrastive, and distance losses.
Reference: https://github.com/AstraZeneca/SubTab
"""

from typing import Dict, Any, Tuple, Callable

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─── Basic Loss Functions ───────────────────────────────────────────────────

def get_mse_loss(recon: Tensor, target: Tensor) -> Tensor:
    """
    Compute Mean Squared Error loss.
    
    Args:
        recon: Reconstructed data tensor
        target: Original data tensor
        
    Returns:
        MSE loss value
    """
    return F.mse_loss(recon, target, reduction="mean")


def get_bce_loss(prediction: Tensor, label: Tensor) -> Tensor:
    """
    Compute Binary Cross-Entropy loss.
    
    Args:
        prediction: Predicted probabilities
        label: Ground truth labels
        
    Returns:
        BCE loss value
    """
    return F.binary_cross_entropy(prediction, label, reduction="mean")


# ─── Joint Loss Module ──────────────────────────────────────────────────────

class JointLoss(nn.Module):
    """
    Combined loss module for contrastive federated learning.
    
    Integrates three loss components:
    1. Reconstruction loss (MSE or BCE)
    2. Contrastive loss (similarity-based)
    3. Distance loss (latent space consistency)
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        """
        Initialize joint loss module.
        
        Args:
            options: Configuration dictionary containing:
                - batch_size: Batch size for training
                - tau: Temperature parameter for contrastive loss
                - device: Torch device (CPU/GPU)
                - cosine_similarity: Whether to use cosine similarity
                - reconstruction: Whether to use MSE (True) or BCE (False)
                - contrastive_loss: Whether to include contrastive loss
                - distance_loss: Whether to include distance loss
        """
        super(JointLoss, self).__init__()
        
        self.options = options
        self.batch_size = options["batch_size"]
        self.temperature = options["tau"]
        self.device = options["device"]
        
        # Select similarity function
        if options["cosine_similarity"]:
            self.similarity_fn = self._cosine_similarity
        else:
            self.similarity_fn = self._dot_similarity
        
        # Loss criterion for contrastive learning
        self.criterion = nn.MSELoss(reduction="mean")
        
        # Precompute mask for negative samples
        self.mask_for_neg_samples = self._get_mask_for_neg_samples()


    def _get_mask_for_neg_samples(self) -> Tensor:
        """
        Generate mask for selecting negative samples in similarity matrix.
        
        Creates a 2N×2N boolean mask where:
        - Diagonal blocks are False (positive pairs)
        - Off-diagonal blocks are True (negative pairs)
        
        Returns:
            Boolean mask tensor
        """
        # Create diagonal mask for positive pairs
        diagonal = th.eye(
            2 * self.batch_size,
            device=self.device,
            dtype=th.bool
        )
        
        # Create block diagonal structure
        q1 = th.eye(self.batch_size, device=self.device, dtype=th.bool)
        q3 = th.eye(self.batch_size, device=self.device, dtype=th.bool)
        mask = th.block_diag(q1, q3)
        
        # Invert to select negative samples
        return ~mask


    @staticmethod
    def _dot_similarity(x: Tensor, y: Tensor) -> Tensor:
        """
        Compute normalized dot product similarity between two sets of vectors.
        
        Args:
            x: First set of vectors (2N × D)
            y: Second set of vectors (2N × D)
            
        Returns:
            Similarity matrix (N × N)
        """
        # Split into two halves
        x = x[:x.shape[0] // 2]
        y = y[y.shape[0] // 2:]
        
        # Normalize vectors
        x_normalized = F.normalize(x, dim=-1)
        y_normalized = F.normalize(y, dim=-1)
        
        # Compute similarity
        return th.matmul(x_normalized, y_normalized.T)


    @staticmethod
    def _cosine_similarity(x: Tensor, y: Tensor) -> Tensor:
        """
        Compute cosine similarity between two sets of vectors.
        
        Args:
            x: First set of vectors (2N × D)
            y: Second set of vectors (2N × D)
            
        Returns:
            Similarity matrix (N × N)
        """
        # Split into two halves
        x = x[:x.shape[0] // 2]
        y = y[y.shape[0] // 2:]
        
        # Normalize vectors
        x_normalized = F.normalize(x, dim=-1)
        y_normalized = F.normalize(y, dim=-1)
        
        # Compute cosine similarity
        similarity_fn = th.nn.CosineSimilarity(dim=-1)
        return similarity_fn(x_normalized, y_normalized)


    def contrastive_loss(self, representation: Tensor) -> Tensor:
        """
        Compute contrastive loss using similarity matrix.
        
        Encourages representations of augmented views to be similar
        while pushing apart representations from different samples.
        
        Args:
            representation: Latent representations (2N × D)
            
        Returns:
            Contrastive loss value
        """
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)
        
        # Scale by temperature
        logits = similarity / self.temperature
        
        # Create zero labels (target similarity)
        labels = th.zeros_like(logits).to(self.device)
        
        # Compute MSE loss
        loss = self.criterion(logits, labels)
        
        return loss


    def reconstruction_loss(self, xrecon: Tensor, xorig: Tensor) -> Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            xrecon: Reconstructed input
            xorig: Original input
            
        Returns:
            Reconstruction loss value
        """
        if self.options["reconstruction"]:
            return get_mse_loss(xrecon, xorig)
        else:
            return get_bce_loss(xrecon, xorig)


    def distance_loss(self, representation: Tensor) -> Tensor:
        """
        Compute distance loss between paired representations.
        
        Encourages consistency between representations of the same
        sample from different augmented views.
        
        Args:
            representation: Latent representations (2N × D)
            
        Returns:
            Distance loss value
        """
        # Split into two views
        zi, zj = th.chunk(representation, 2, dim=0)
        
        # Compute MSE between paired representations
        return get_mse_loss(zi, zj)


    def forward(
        self,
        representation: Tensor,
        xrecon: Tensor,
        xorig: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute combined loss.
        
        Args:
            representation: Latent representations (2N × D)
            xrecon: Reconstructed input (N × D_in)
            xorig: Original input (N × D_in)
            
        Returns:
            Tuple of (total_loss, contrastive_loss, recon_loss, distance_loss)
        """
        # 1. Reconstruction loss
        recon_loss = self.reconstruction_loss(xrecon, xorig)
        
        # 2. Contrastive loss (optional)
        if self.options["contrastive_loss"]:
            closs = self.contrastive_loss(representation)
        else:
            closs = th.tensor(0.0, device=self.device)
        
        # 3. Distance loss (optional)
        if self.options["distance_loss"]:
            zrecon_loss = self.distance_loss(representation)
        else:
            zrecon_loss = th.tensor(0.0, device=self.device)
        
        # 4. Total loss
        total_loss = recon_loss + closs + zrecon_loss
        
        return total_loss, closs, recon_loss, zrecon_loss


# ─── Loss Weighting (Optional) ──────────────────────────────────────────────

class WeightedJointLoss(JointLoss):
    """
    Extended joint loss with configurable loss weights.
    """

    def __init__(
        self,
        options: Dict[str, Any],
        recon_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        distance_weight: float = 1.0
    ) -> None:
        """
        Initialize weighted joint loss.
        
        Args:
            options: Configuration dictionary
            recon_weight: Weight for reconstruction loss
            contrastive_weight: Weight for contrastive loss
            distance_weight: Weight for distance loss
        """
        super().__init__(options)
        
        self.recon_weight = recon_weight
        self.contrastive_weight = contrastive_weight
        self.distance_weight = distance_weight


    def forward(
        self,
        representation: Tensor,
        xrecon: Tensor,
        xorig: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute weighted combined loss.
        
        Args:
            representation: Latent representations
            xrecon: Reconstructed input
            xorig: Original input
            
        Returns:
            Tuple of (total_loss, contrastive_loss, recon_loss, distance_loss)
        """
        # Compute individual losses
        recon_loss = self.reconstruction_loss(xrecon, xorig)
        
        if self.options["contrastive_loss"]:
            closs = self.contrastive_loss(representation)
        else:
            closs = th.tensor(0.0, device=self.device)
        
        if self.options["distance_loss"]:
            zrecon_loss = self.distance_loss(representation)
        else:
            zrecon_loss = th.tensor(0.0, device=self.device)
        
        # Apply weights
        total_loss = (
            self.recon_weight * recon_loss +
            self.contrastive_weight * closs +
            self.distance_weight * zrecon_loss
        )
        
        return total_loss, closs, recon_loss, zrecon_loss


# ─── Utility Functions ──────────────────────────────────────────────────────

def compute_similarity_matrix(
    z1: Tensor,
    z2: Tensor,
    temperature: float = 0.5,
    similarity_type: str = "cosine"
) -> Tensor:
    """
    Compute similarity matrix between two sets of representations.
    
    Args:
        z1: First set of representations (N × D)
        z2: Second set of representations (N × D)
        temperature: Temperature scaling parameter
        similarity_type: Type of similarity ('cosine' or 'dot')
        
    Returns:
        Similarity matrix (N × N)
    """
    # Normalize representations
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    
    # Compute similarity
    if similarity_type == "cosine":
        similarity = th.matmul(z1_norm, z2_norm.T)
    elif similarity_type == "dot":
        similarity = th.matmul(z1, z2.T)
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")
    
    # Apply temperature scaling
    return similarity / temperature


def info_nce_loss(
    z1: Tensor,
    z2: Tensor,
    temperature: float = 0.5
) -> Tensor:
    """
    Compute InfoNCE loss for contrastive learning.
    
    Args:
        z1: First set of representations (N × D)
        z2: Second set of representations (N × D)
        temperature: Temperature parameter
        
    Returns:
        InfoNCE loss value
    """
    batch_size = z1.shape[0]
    
    # Compute similarity matrix
    similarity = compute_similarity_matrix(z1, z2, temperature)
    
    # Create labels (diagonal elements are positive pairs)
    labels = th.arange(batch_size, device=z1.device)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity, labels)
    
    return loss