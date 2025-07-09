import numpy as np
from train import Server
import yaml
import random
import torch
from defenses.defensemanager import DefenseManager

class RobustServer(Server):
    def __init__(self, model, config):
        super().__init__(model,config)
        self.defense_manager = DefenseManager(config)
        self.historical_updates = {k: [] for k in self.global_dict.keys()}
        self.aggregated_method = self.random_aggregate if config['randomLevel'] < 1 else self.trimmed_aggregate
        
    def trimmed_aggregate(self, client_models):
        validated_models = []
        list_paramNorm = []
        for client in client_models:
            params = client.get_model_params()
            paramNorm = [torch.norm(params[k].float()) for k in params.keys()]
            list_paramNorm.append(paramNorm)
        all_norms = torch.stack(sum(list_paramNorm, [])).detach().cpu().numpy()

        median_norm = np.median(all_norms)
        q75 = np.percentile(all_norms, 75)
        q25 = np.percentile(all_norms, 25)
        iqr = q75 - q25
        
        # Update threshold using robust statistics
        clip_threshold = q75 + 1.5 * iqr 

        
        for client in client_models:
            params = client.get_model_params()
            is_valid = True
            is_valid = all(
                self.defense_manager.validate_update(
                    params[k],
                    self.historical_updates[k],
                    clip_threshold
                ) for k in params.keys()
            )
            if is_valid:
                validated_models.append(client)
            # else:
            #     print(f"Detected suspicious update from client {client.client_number}")
        if not validated_models:
            print("Warning: No valid updates received")
            return
            
        # Apply robust aggregation
        
        # self.global_dict = self.robust_aggregate(validated_models)

        self.global_dict = self.aggregate_models(validated_models)
        
        # Update historical records
        for k in self.global_dict.keys():
            self.historical_updates[k].append(self.global_dict[k].clone())
            if len(self.historical_updates[k]) > self.defense_manager.history_size:
                self.historical_updates[k].pop(0)
                
    def random_aggregate(self, client_models):
        for k in self.global_dict.keys():
            param_stack = [client.get_model_params()[k].float() for client in client_models]
            # Randomly sample parameters based on config['randomLevel']
            param_stack = random.sample(param_stack, int(len(client_models) * self.config['randomLevel']))
            self.global_dict[k] = torch.stack(param_stack).mean(0)


    def robust_aggregate(self, validated_models, weights=None):
        """Implements robust aggregation using trimmed mean"""
        param_updates = [client.get_model_params() for client in validated_models]
        if weights is None:
            weights = [1/len(param_updates)] * len(param_updates)
            
        aggregated_params = {}
        for key in self.global_dict.keys():
            # Stack parameters and convert to numpy for trimmed mean
            params = torch.stack([update[key].float() for update in param_updates])
            trimmed = self.trim_(params).mean(0)
            aggregated_params[key] = trimmed
        return aggregated_params
    

    def trim_(self, tensors):
        """
        Efficient implementation of trimmed mean using pure PyTorch operations
        Args:
            tensors: List of tensors or stacked tensor [num_models, *tensor_shape]
        """
        if isinstance(tensors, list):
            tensors = torch.stack(tensors)
        
        n = tensors.size(0)
        k = int(n * self.defense_manager.trim_ratio)
        
        if k == 0:
            return tensors
        
        # Sort along the first dimension
        sorted_tensors, _ = torch.sort(tensors, dim=0)
        
        # Remove k smallest and k largest elements
        trimmed = sorted_tensors[k:n-k]
        
        # Compute mean of remaining elements
        return trimmed
