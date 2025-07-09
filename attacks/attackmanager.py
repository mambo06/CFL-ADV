import torch.nn.functional as F
from enum import Enum
import torch


class AttackType(Enum):
    SCALE = "scale" 
    MODEL_REPLACEMENT = "model_replacement"
    DIRECTION = "direction"
    GRADIENT_ASCENT = "gradient_ascent"
    TARGETED = "targeted"

class AttackManager:
    def __init__(self, config):
        self.config = config
        self.attack_type = AttackType(config.get('attack_type', 'scale'))
        self.attack_scale = config.get('attack_scale', 10.0)
        self.target_layer = config.get('target_layer', None)
        self.noise_std = config.get('noise_std', 0.1)
        self.target_direction = None
        self.gradient_scale = 0.01
        
    def generate_target_direction(self, param_shape):
        """Generate a malicious target direction for direction-based attacks"""
        if self.target_direction is None or self.target_direction.shape != param_shape:
            self.target_direction = torch.randn(param_shape)
            self.target_direction = F.normalize(self.target_direction, dim=-1)
        return self.target_direction