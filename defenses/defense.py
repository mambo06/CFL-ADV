from typing import List, Dict, Any, Optional
import torch
import numpy as np
import random
import torch.nn.functional as F

class BaseDefense:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def validate_update(self, update: Dict[str, torch.Tensor]) -> bool:
        raise NotImplementedError
        
    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


# Add new defense classes
class TrimmedMeanDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trim_ratio = config.get('trim_ratio', 0.1)
        self.history_size = config.get('history_size', 10)
        self.historical_updates = {}

    def validate_update(self, update: torch.Tensor, 
                       history: List[torch.Tensor], 
                       clip_threshold: float) -> bool:
        """Validate individual parameter update"""
        param_norm = torch.norm(update.float())
        return param_norm <= clip_threshold

    def trim_(self, tensors: torch.Tensor) -> torch.Tensor:
        """Efficient implementation of trimmed mean using pure PyTorch operations"""
        if isinstance(tensors, list):
            tensors = torch.stack(tensors)
        
        n = tensors.size(0)
        k = int(n * self.trim_ratio)
        
        if k == 0:
            return tensors
        
        sorted_tensors, _ = torch.sort(tensors, dim=0)
        trimmed = sorted_tensors[k:n-k]
        return trimmed

    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Calculate parameter norms
        list_paramNorm = []
        for update in updates:
            paramNorm = [torch.norm(param.float()) for param in update.values()]
            list_paramNorm.append(paramNorm)
        all_norms = torch.stack(sum(list_paramNorm, [])).detach().cpu().numpy()

        # Calculate robust statistics
        median_norm = np.median(all_norms)
        q75 = np.percentile(all_norms, 75)
        q25 = np.percentile(all_norms, 25)
        iqr = q75 - q25
        clip_threshold = q75 + 1.5 * iqr

        # Validate updates
        validated_updates = []
        for update in updates:
            is_valid = True
            for k, v in update.items():
                if k not in self.historical_updates:
                    self.historical_updates[k] = []
                if not self.validate_update(v, self.historical_updates[k], clip_threshold):
                    is_valid = False
                    break
            if is_valid:
                validated_updates.append(update)

        if not validated_updates:
            return updates[0]  # Return first update if no valid updates

        # Aggregate validated updates
        aggregated = {}
        for k in validated_updates[0].keys():
            params = torch.stack([update[k].float() for update in validated_updates])
            aggregated[k] = self.trim_(params).mean(0)

            # Update historical records
            if k not in self.historical_updates:
                self.historical_updates[k] = []
            self.historical_updates[k].append(aggregated[k].clone())
            if len(self.historical_updates[k]) > self.history_size:
                self.historical_updates[k].pop(0)

        return aggregated

class RandomDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.random_level = config.get('random_level', 0.8)

    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        import random
        
        # Randomly sample updates
        sample_size = int(len(updates) * self.random_level)
        sampled_updates = random.sample(updates, sample_size)
        
        # Average sampled updates
        aggregated = {}
        for k in updates[0].keys():
            param_stack = [update[k].float() for update in sampled_updates]
            aggregated[k] = torch.stack(param_stack).mean(0)
        
        return aggregated

class RobustDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trim_ratio = config.get('trim_ratio', 0.1)

    def aggregate(self, updates: List[Dict[str, torch.Tensor]], 
                 weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        if weights is None:
            weights = [1/len(updates)] * len(updates)

        aggregated = {}
        for k in updates[0].keys():
            params = torch.stack([update[k].float() for update in updates])
            trimmed = self._trim(params)
            aggregated[k] = trimmed.mean(0)
        
        return aggregated

    def _trim(self, tensors: torch.Tensor) -> torch.Tensor:
        """Implements trimmed mean"""
        if isinstance(tensors, list):
            tensors = torch.stack(tensors)
        
        n = tensors.size(0)
        k = int(n * self.trim_ratio)
        
        if k == 0:
            return tensors
        
        sorted_tensors, _ = torch.sort(tensors, dim=0)
        trimmed = sorted_tensors[k:n-k]
        return trimmed

class MultiKrumDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_select = config.get('num_select', 10)
        self.byzantine_ratio = config.get('byzantine_ratio', 0.3)
        
    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n = len(updates)
        f = int(n * self.byzantine_ratio)
        
        # Calculate pairwise distances
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Convert tensors to float before calculating norm
                    dist = sum(torch.norm(updates[i][k].float() - updates[j][k].float())**2 
                             for k in updates[i].keys())
                    distances[i,j] = dist
                    
        # Score each update
        scores = []
        for i in range(n):
            closest = torch.sort(distances[i])[0][:(n-f-2)]
            scores.append(torch.sum(closest))
            
        # Select updates with lowest scores
        selected_idx = torch.argsort(torch.tensor(scores))[:self.num_select]
        selected_updates = [updates[i] for i in selected_idx]
        
        # Average selected updates
        aggregated = {}
        for k in updates[0].keys():
            # Convert to float before stacking
            aggregated[k] = torch.stack([upd[k].float() for upd in selected_updates]).mean(0)
        return aggregated


class GeometricMedianDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_iters = config.get('num_iters', 10)
        self.eps = config.get('eps', 1e-5)

    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Initialize with mean
        median = {}
        for k in updates[0].keys():
            # Convert to float before stacking and taking mean
            median[k] = torch.stack([upd[k].float() for upd in updates]).mean(0)
            
        # Weiszfeld algorithm
        for _ in range(self.num_iters):
            weights = []
            for update in updates:
                dist = sum(torch.norm(update[k].float() - median[k])**2 
                          for k in update.keys())**0.5
                weights.append(1 / (dist + self.eps))
                
            weights = torch.tensor(weights) / sum(weights)
            
            # Update median
            new_median = {}
            for k in median.keys():
                # Convert to float before weighted sum
                new_median[k] = sum(w * upd[k].float() 
                                  for w, upd in zip(weights, updates))
            median = new_median
            
        return median

import torch
from typing import Dict, Any, List


class FoolsGoldDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.memory = None
        self.epsilon = 1e-5

    def _get_valid_keys(self, updates: List[Dict[str, torch.Tensor]]):
        """
        Return keys that exist in all clients and have identical shapes.
        """
        common_keys = set(updates[0].keys())

        for update in updates[1:]:
            common_keys = common_keys.intersection(set(update.keys()))

        valid_keys = []

        for k in sorted(common_keys):
            ref_shape = updates[0][k].shape
            valid = True

            for update in updates:
                if update[k].shape != ref_shape:
                    valid = False
                    break

            if valid:
                valid_keys.append(k)

        return valid_keys

    def _flatten_update(self, update: Dict[str, torch.Tensor], valid_keys: List[str]):
        """
        Flatten one client update into a single vector using only valid keys.
        """
        flat_tensors = []

        for k in valid_keys:
            tensor = update[k].detach().float().reshape(-1)
            flat_tensors.append(tensor)

        if len(flat_tensors) == 0:
            raise RuntimeError("No valid matching keys found across client updates.")

        return torch.cat(flat_tensors)

    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n_clients = len(updates)

        if n_clients == 0:
            raise ValueError("No updates provided to FoolsGoldDefense.")

        valid_keys = self._get_valid_keys(updates)

        if len(valid_keys) == 0:
            raise RuntimeError(
                "No common parameters with identical shapes across all clients. "
                "Check that all clients use the same model architecture and update format."
            )

        # Flatten all client updates consistently
        flat_updates = []

        for update in updates:
            flat_update = self._flatten_update(update, valid_keys)
            flat_updates.append(flat_update)

        lengths = [x.numel() for x in flat_updates]

        if len(set(lengths)) != 1:
            raise RuntimeError(
                f"Flattened update sizes do not match: {lengths}. "
                "This means client updates are structurally inconsistent."
            )

        flat_updates = torch.stack(flat_updates, dim=0)

        # Initialize or reset memory if shape/client count changed
        if (
            self.memory is None
            or self.memory.shape != flat_updates.shape
        ):
            self.memory = torch.zeros_like(flat_updates)

        # Update memory
        self.memory += flat_updates

        # Cosine similarity matrix
        normalized_memory = self.memory / (
            torch.norm(self.memory, dim=1, keepdim=True) + self.epsilon
        )

        cs = torch.mm(normalized_memory, normalized_memory.t())

        # Ignore self-similarity
        cs.fill_diagonal_(0.0)

        # FoolsGold weighting
        max_cs, _ = torch.max(cs, dim=1)

        weights = torch.ones(n_clients, device=flat_updates.device)

        for i in range(n_clients):
            if max_cs[i] > self.epsilon:
                weights[i] = 1.0 - max_cs[i]
            else:
                weights[i] = 1.0

        # Clamp for numerical stability
        weights = torch.clamp(weights, min=0.0, max=1.0)

        # Normalize weights
        weights = weights / (torch.sum(weights) + self.epsilon)

        # Aggregate original tensors
        aggregated = {}

        for k in updates[0].keys():
            # Only aggregate keys with matching shape across all clients
            if k in valid_keys:
                aggregated[k] = torch.zeros_like(updates[0][k].float())

                for i, update in enumerate(updates):
                    aggregated[k] += weights[i].to(update[k].device) * update[k].float()

            else:
                # Fallback: if key is not valid, use simple average only among matching shapes
                ref_shape = updates[0][k].shape
                matching = [
                    update[k].float()
                    for update in updates
                    if k in update and update[k].shape == ref_shape
                ]

                if len(matching) > 0:
                    aggregated[k] = torch.stack(matching, dim=0).mean(dim=0)
                else:
                    aggregated[k] = updates[0][k].float()

        return aggregated


class MomentumDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.momentum = config.get('momentum', 0.9)
        self.velocity = {}
        
    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # First aggregate using mean
        aggregated = {}
        for k in updates[0].keys():
            aggregated[k] = torch.stack([upd[k].float() for upd in updates]).mean(0)
            
        # Apply momentum
        if not self.velocity:
            self.velocity = {k: torch.zeros_like(v) for k,v in aggregated.items()}
            
        filtered = {}
        for k in aggregated.keys():
            self.velocity[k] = (self.momentum * self.velocity[k] + 
                              (1 - self.momentum) * aggregated[k])
            filtered[k] = self.velocity[k]
            
        return filtered