import torch.nn.functional as F
import torch

class DefenseManager:
    def __init__(self, config):
        self.config = config
        self.historical_updates = []
        self.cosine_threshold = config.get('cosine_threshold', 0.75)
        # self.clip_threshold = config.get('clip_threshold', 100.0)
        self.trim_ratio = config.get('trim_ratio', 0.1)
        self.history_size = config.get('history_size', 10)
        
    def validate_update(self, param_update, historical_params, clip_threshold):
        """Validates parameter updates using multiple defense mechanisms"""
        if self.detect_scale_attack(param_update, clip_threshold):
            return False
        if self.detect_direction_attack(param_update, historical_params):
            return False
        return True
    
    def detect_scale_attack(self, param_update, clip_threshold):
        """Detects scaling-based attacks using norm thresholding"""
        param_norm = torch.norm(param_update.float())

        # if param_norm > clip_threshold: print(param_norm , clip_threshold)
        return param_norm > clip_threshold
    
    def detect_direction_attack(self, param_update, historical_params):
        """Detects suspicious direction changes using cosine similarity"""
        if not historical_params:
            return False
            
        current_direction = param_update.float().view(-1)
        historical_direction = historical_params[-1].float().view(-1)
        
        similarity = F.cosine_similarity(current_direction.unsqueeze(0),
                                      historical_direction.unsqueeze(0))
        # if similarity < self.cosine_threshold : print(similarity , self.cosine_threshold)
        return similarity < self.cosine_threshold