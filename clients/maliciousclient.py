from train import Client
import random
import torch

class MaliciousClient(Client):
    def __init__(self, model, dataloader, client_number, config, attack_manager):
        super().__init__(model, dataloader, client_number)
        self.attack_manager = attack_manager
        self.original_params = None
        self.attack_probability = config.get('attack_probability', 1.0)
        
    def should_attack(self):
        """Probabilistic attack decision"""
        return random.random() < self.attack_probability

    def scale_attack(self, params):
        """Scale-based attack implementation"""
        for key, param in params.items():
            params[key] = param * self.attack_manager.attack_scale
        return params

    def model_replacement_attack(self, params):
        """Model replacement attack implementation"""
        for key, param in params.items():
            if self.attack_manager.target_layer is None or self.attack_manager.target_layer in key:
                params[key] = -param  # Invert the parameters
        return params

    def direction_attack(self, params):
        """Direction-based attack implementation"""
        for key, param in params.items():
            target_direction = self.attack_manager.generate_target_direction(param.shape)
            params[key] = target_direction * torch.norm(param.float())
        return params

    def gradient_ascent_attack(self, params):
        """Gradient ascent attack implementation"""
        if self.original_params is None:
            self.original_params = {k: v.clone() for k, v in params.items()}
        
        for key, param in params.items():
            gradient = param - self.original_params[key]
            params[key] = param + gradient * self.attack_manager.gradient_scale
        return params

    def targeted_attack(self, params):
        """Targeted model poisoning attack"""
        target_value = torch.tensor(1.0)  # Example target value
        for key, param in params.items():
            if self.attack_manager.target_layer is None or self.attack_manager.target_layer in key:
                noise = torch.randn_like(param) * self.attack_manager.noise_std
                params[key] = target_value + noise
        return params

    def apply_attack(self, params):
        """Apply the selected attack strategy"""
        if not self.should_attack():
            return params

        if self.attack_manager.attack_type == AttackType.SCALE:
            return self.scale_attack(params)
        elif self.attack_manager.attack_type == AttackType.MODEL_REPLACEMENT:
            return self.model_replacement_attack(params)
        elif self.attack_manager.attack_type == AttackType.DIRECTION:
            return self.direction_attack(params)
        elif self.attack_manager.attack_type == AttackType.GRADIENT_ASCENT:
            return self.gradient_ascent_attack(params)
        elif self.attack_manager.attack_type == AttackType.TARGETED:
            return self.targeted_attack(params)
        return params

    def get_model_params(self):
        """Override to include attack"""
        params = super().get_model_params()
        if self.poison:
            params = self.apply_attack(params)
        return params

    def train(self):
        """Override to potentially modify training process for attacks"""
        tloss = super().train()
        if self.poison and self.attack_manager.attack_type == AttackType.GRADIENT_ASCENT:
            # Invert the gradient for gradient ascent attack
            for param in self.model.encoder.parameters():
                if param.grad is not None:
                    param.grad = -param.grad
        return tloss