from typing import List, Dict, Any, Optional
import torch
import numpy as np
import random

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
                    dist = sum(torch.norm(updates[i][k] - updates[j][k])**2 
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
            aggregated[k] = torch.stack([upd[k] for upd in selected_updates]).mean(0)
        return aggregated

class GeometricMedianDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_iterations = config.get('num_iterations', 5)
        
    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        median = {}
        
        # Initialize with mean
        for k in updates[0].keys():
            median[k] = torch.stack([upd[k] for upd in updates]).mean(0)
        
        # Iterative refinement
        for _ in range(self.num_iterations):
            weights = []
            for update in updates:
                dist = sum(torch.norm(update[k] - median[k]) 
                          for k in median.keys())
                weights.append(1 / max(dist, 1e-8))
                
            weights = torch.tensor(weights)
            weights = weights / weights.sum()
            
            # Update median
            for k in median.keys():
                stacked = torch.stack([upd[k] for upd in updates])
                median[k] = (stacked * weights.view(-1,1,1)).sum(0)
                
        return median

class FoolsGoldDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.memory = {}
        
    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n = len(updates)
        
        # Calculate cosine similarities
        similarities = torch.zeros(n, n)
        for i in range(n):
            for j in range(i+1, n):
                sim = self._compute_update_similarity(updates[i], updates[j])
                similarities[i,j] = similarities[j,i] = sim
                
        # Calculate weights using FoolsGold algorithm
        weights = self._get_foolsgold_weights(similarities)
        
        # Weighted aggregation
        aggregated = {}
        for k in updates[0].keys():
            stacked = torch.stack([upd[k] for upd in updates])
            aggregated[k] = (stacked * weights.view(-1,1,1)).sum(0)
            
        return aggregated
    
    def _compute_update_similarity(self, update1: Dict[str, torch.Tensor], 
                                 update2: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cosine similarity between two updates"""
        v1 = torch.cat([update1[k].flatten() for k in update1.keys()])
        v2 = torch.cat([update2[k].flatten() for k in update2.keys()])
        return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))[0]
    
    def _get_foolsgold_weights(self, similarities: torch.Tensor) -> torch.Tensor:
        """Calculate FoolsGold weights based on cosine similarities"""
        n = similarities.size(0)
        weights = torch.ones(n)
        
        # Penalize similar updates
        for i in range(n):
            similar_indices = similarities[i] > 0.5
            if similar_indices.sum() > 1:
                weights[i] *= 1.0 / similar_indices.sum()
                
        return F.softmax(weights, dim=0)

class MomentumDefense(BaseDefense):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.momentum = config.get('momentum', 0.9)
        self.velocity = {}
        
    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # First aggregate using mean
        aggregated = {}
        for k in updates[0].keys():
            aggregated[k] = torch.stack([upd[k] for upd in updates]).mean(0)
            
        # Apply momentum
        if not self.velocity:
            self.velocity = {k: torch.zeros_like(v) for k,v in aggregated.items()}
            
        filtered = {}
        for k in aggregated.keys():
            self.velocity[k] = (self.momentum * self.velocity[k] + 
                              (1 - self.momentum) * aggregated[k])
            filtered[k] = self.velocity[k]
            
        return filtered