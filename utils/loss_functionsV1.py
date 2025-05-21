import torch as th
import torch.nn.functional as F
import torch.nn as nn


def getMSEloss(recon, target):
    """
    Compute Mean Squared Error loss.
    Args:
        recon (torch.FloatTensor): Reconstructed data.
        target (torch.FloatTensor): Original data.
    """
    return F.mse_loss(recon, target, reduction='mean')


def getBCELoss(prediction, label):
    """
    Compute Binary Cross-Entropy loss.
    Args:
        prediction (torch.FloatTensor): Predicted probabilities.
        label (torch.FloatTensor): Ground truth labels.
    """
    return F.binary_cross_entropy(prediction, label, reduction='mean')


class JointLoss(nn.Module):
    """
    A custom loss module combining reconstruction, contrastive, and distance losses.
    """
    def __init__(self, options):
        super(JointLoss, self).__init__()
        self.options = options
        self.batch_size = options["batch_size"]
        self.temperature = options["tau"]
        self.device = options["device"]
        self.mask_for_neg_samples = self._get_mask_for_neg_samples()
        self.similarity_fn = self._cosine_similarity if options["cosine_similarity"] else self._dot_similarity
        self.criterion = nn.CrossEntropyLoss(reduction="mean")


    def _get_mask_for_neg_samples(self):
        """
        Generate a mask for negative samples in the similarity matrix.
        """
        # Create identity matrices for the diagonal blocks
        diagonal = th.eye(2 * self.batch_size, device=self.device, dtype=th.bool)
        q1 = th.eye(self.batch_size, device=self.device, dtype=th.bool)
        q3 = th.eye(self.batch_size, device=self.device, dtype=th.bool)

        # Combine quadrants to form the full mask
        mask = th.block_diag(q1, q3)
        mask = ~mask  # Invert to select negative samples
        return mask

    @staticmethod
    def _dot_similarity(x, y):
        """
        Compute dot product similarity.
        """
        x = x[:int(x.shape[0]/2)]
        y = y[int(y.shape[0]/2):]
        return th.matmul(x, y.T)

    @staticmethod
    def _cosine_similarity(x, y):
        """
        Compute cosine similarity.
        """
        x = x[:int(x.shape[0]/2)]
        y = y[int(y.shape[0]/2):]

        x_normalized = F.normalize(x, dim=-1)
        y_normalized = F.normalize(y, dim=-1)
        return th.matmul(x_normalized, y_normalized.T)

    def XNegloss(self, representation):
        """
        Compute contrastive loss using a 2Nx2N similarity matrix.
        """
        similarity = self.similarity_fn(representation, representation)
        logits = similarity / self.temperature
        # labels = th.cat([th.arange(self.batch_size/2), th.arange(self.batch_size/2)]).to(self.device).long()
        labels = th.zeros( self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss

    def forward(self, representation, xrecon, xorig):
        """
        Compute the combined loss.
        Args:
            representation (torch.FloatTensor): Latent representations.
            xrecon (torch.FloatTensor): Reconstructed input.
            xorig (torch.FloatTensor): Original input.
        """
        # Reconstruction loss
        recon_loss = getMSEloss(xrecon, xorig) if self.options["reconstruction"] else getBCELoss(xrecon, xorig)

        # Contrastive loss
        closs = self.XNegloss(representation) if self.options["contrastive_loss"] else 0.0

        # Distance loss
        zrecon_loss = 0.0
        if self.options["distance_loss"]:
            zi, zj = th.chunk(representation, 2, dim=0)
            zrecon_loss = getMSEloss(zi, zj)

        # Total loss
        total_loss = recon_loss + closs + zrecon_loss
        return total_loss, closs, recon_loss, zrecon_loss
