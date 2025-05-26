import torch.nn.functional as F
from enum import Enum
import numpy as np
from train import Server, Client
import yaml

from utils.arguments import get_arguments, get_config, print_config_summary
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims, set_seed
import json
from pathlib import Path
from utils.load_data import Loader
from src.model import CFL
import random
from tqdm import tqdm
import torch
import evaluation as eval
import copy




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
        # if self.detect_scale_attack(param_update, clip_threshold):
        #     return False
        # if self.detect_direction_attack(param_update, historical_params):
        #     return False
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


class SecureClient(Client):
    def __init__(self, model, dataloader, client_number, config):
        super().__init__(model, dataloader, client_number)
        self.clip_norm = config.get('clip_norm', 1.0)
        
    def train(self):
        tloss = super().train()
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.encoder.parameters(), 
            self.clip_norm
        )
        return tloss

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

def run(config, save_weights):
    config = copy.deepcopy(config)
    ds_loader = Loader(config, dataset_name=config["dataset"], client=0)
    config = update_config_with_model_dims(ds_loader, config)
    global_model = CFL(config)
    server = RobustServer(global_model, config)
    clients = []
    
    # Initialize attack manager
    attack_manager = AttackManager(config)

    poison_clients = random.sample(
        range(config["fl_cluster"]), 
        int(config["fl_cluster"] * config['malClient'])
    ) if config['malClient'] > 0.0 else []

    print(f'Warning: Poisoning applied to clients: {poison_clients}')
    print(f'Attack type: {attack_manager.attack_type.value}')

    for clt in range(config["fl_cluster"]):
        prefix = (f"Cl-{clt}-{config['epochs']}e-{config['fl_cluster']}fl-"
                 f"{config['malClient']}mc-{config['attack_type']}_at-"
                 f"{config['randomLevel']}rl-{config['dataset']}")
        config.update({"prefix":prefix})
        loader = Loader(config, dataset_name=config["dataset"], client=clt).trainFL_loader
        # Use MaliciousClient for poisoned clients
        if clt in poison_clients:
            client = MaliciousClient(global_model, loader, clt, config, attack_manager)
        else:
            client = SecureClient(global_model, loader, clt, config)
        client.poison = clt in poison_clients
        clients.append(client)

    # Training loop
    total_batches = len(loader)
    for epoch in range(config["epochs"]):
        tloss = 0
        for i in tqdm(range(total_batches)):
            for client in clients:
                tloss += client.train().item()

            
          
            server.aggregated_method(clients)

            for client in clients:
                client.set_model(server.distribute_model())
                # client.step()
                client.model.loss["tloss_e"].append(
                    sum(client.model.loss["tloss_b"][-total_batches:]) / total_batches
                )

        avg_loss = tloss / (config["fl_cluster"] * total_batches)
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

    # Save results
    for n, client in enumerate(clients):
        model = client.model
        model.saveTrainParams(n)

        if save_weights:
            model.save_weights(n)

        prefix = (f"Client-{n}-{config['epochs']}e-{config['fl_cluster']}fl-"
                 f"{config['malClient']}pc-{config['attack_type']}-"
                 f"{config['attack_scale']}-{config['dataset']}")
        
        with open(model._results_path + f"/config_{prefix}.yml", 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

def main(config):
    config["framework"] = config["dataset"]
    info_path = Path(f'data/{config["dataset"]}/info.json')
    info = json.loads(info_path.read_text())
    
    config.update({
        'task_type': info['task_type'],
        'cat_policy': info['cat_policy'],
        'norm': info['norm'],
        'learning_rate_reducer': config['learning_rate'],
        "attack_probability": 1.0,
        "target_layer": "encoder.layer1",
        "noise_std": 0.1,
    })

    run(copy.deepcopy(config), save_weights=True)
    eval.main(copy.deepcopy(config))


if __name__ == "__main__":
    args = get_arguments()
    config = get_config(args)
    # print_config_summary(config)
    main(config)

