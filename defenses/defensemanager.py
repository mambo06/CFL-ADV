import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum

from .defense import (
    MultiKrumDefense,
    GeometricMedianDefense,
    FoolsGoldDefense,
    MomentumDefense,
    TrimmedMeanDefense,
    RandomDefense,
    RobustDefense
)


class DefenseType(Enum):
    MULTI_KRUM = "multi_krum"
    GEOMETRIC_MEDIAN = "geometric_median"
    FOOLSGOLD = "foolsgold"
    TRIMMED_MEAN = "trimmed_mean"
    MOMENTUM = "momentum"
    RANDOM = "random"
    ROBUST = "robust"


class DefenseManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.defense_type = DefenseType(config.get('defense_type', 'multi_krum'))
        
        # Initialize defense mechanisms
        self.defenses = {
            DefenseType.MULTI_KRUM: MultiKrumDefense(config),
            DefenseType.GEOMETRIC_MEDIAN: GeometricMedianDefense(config),
            DefenseType.FOOLSGOLD: FoolsGoldDefense(config),
            DefenseType.MOMENTUM: MomentumDefense(config),
            DefenseType.TRIMMED_MEAN: TrimmedMeanDefense(config),
            DefenseType.RANDOM: RandomDefense(config),
            DefenseType.ROBUST: RobustDefense(config)
        }
        
        self.current_defense = self.defenses[self.defense_type]
        
    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return self.current_defense.aggregate(updates)
    
    def adapt_defense(self, round_number: int, attack_detection_rate: float):
        """Dynamically adjust defense mechanism based on observed behavior"""
        if attack_detection_rate > 0.3:
            self.defense_type = DefenseType.GEOMETRIC_MEDIAN
        elif attack_detection_rate > 0.2:
            self.defense_type = DefenseType.TRIMMED_MEAN
        elif attack_detection_rate > 0.1:
            self.defense_type = DefenseType.MULTI_KRUM
        else:
            self.defense_type = DefenseType.MOMENTUM
            
        self.current_defense = self.defenses[self.defense_type]

    def validate_update(self, update: torch.Tensor, 
                       history: List[torch.Tensor], 
                       clip_threshold: float) -> bool:
        """Validate individual parameter update"""
        return self.current_defense.validate_update(update, history, clip_threshold)
