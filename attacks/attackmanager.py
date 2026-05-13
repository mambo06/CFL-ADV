import torch
import torch.nn.functional as F
from enum import Enum


class AttackType(Enum):
    SCALE = "scale"
    MODEL_REPLACEMENT = "model_replacement"
    DIRECTION = "direction"
    GRADIENT_ASCENT = "gradient_ascent"
    TARGETED = "targeted"


class AttackManager:
    def __init__(self, config):
        self.config = config

        self.attack_type = AttackType(config.get("attack_type", "scale"))

        # Lower default scale. 10.0 is usually too aggressive.
        self.attack_scale = config.get("attack_scale", 1.5)

        self.target_layer = config.get("target_layer", None)
        self.noise_std = config.get("noise_std", 0.01)

        self.target_direction = None

        # Smaller gradient ascent multiplier
        self.gradient_scale = config.get("gradient_scale", 0.001)

        # New safety controls
        self.max_update_norm = config.get("max_update_norm", 1.0)
        self.direction_strength = config.get("direction_strength", 0.1)
        self.target_strength = config.get("target_strength", 0.1)

        # Optional warmup
        self.use_warmup = config.get("use_warmup", True)
        self.warmup_rounds = config.get("warmup_rounds", 20)

    def get_warmup_factor(self, current_round=None):
        """
        Slowly increases attack strength from 0 to 1.
        Prevents early-round instability.
        """
        if not self.use_warmup or current_round is None:
            return 1.0

        return min(1.0, float(current_round + 1) / float(self.warmup_rounds))

    def global_normalize(self, tensor, eps=1e-12):
        """
        Normalize the entire tensor by its global L2 norm.
        This is safer than F.normalize(..., dim=-1).
        """
        norm = torch.norm(tensor)
        return tensor / (norm + eps)

    def clip_by_norm(self, update, max_norm=None, eps=1e-12):
        """
        Clips update so its global L2 norm does not exceed max_norm.
        """
        if max_norm is None:
            max_norm = self.max_update_norm

        norm = torch.norm(update)

        if norm > max_norm:
            update = update * (max_norm / (norm + eps))

        return update

    def generate_target_direction(self, param_shape, device=None, dtype=None):
        """
        Generate a globally normalized target direction.
        """
        need_new_direction = (
            self.target_direction is None
            or self.target_direction.shape != param_shape
        )

        if need_new_direction:
            self.target_direction = torch.randn(
                param_shape,
                device=device,
                dtype=dtype
            )
            self.target_direction = self.global_normalize(self.target_direction)

        else:
            if device is not None:
                self.target_direction = self.target_direction.to(device)
            if dtype is not None:
                self.target_direction = self.target_direction.to(dtype)

        return self.target_direction
