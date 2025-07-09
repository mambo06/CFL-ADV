import numpy as np
from train import Server
import yaml
import random
import torch
from defenses.defensemanager import DefenseManager

class RobustServer(Server):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.defensemanager = DefenseManager(config)
        self.attack_detection_rate = 0.0
        self.round_number = 0
        
    def aggregate(self, client_models):
        """Aggregate updates using selected defense mechanism"""
        updates = [client.get_model_params() for client in client_models]
        
        # Update defense strategy if needed
        # self.defensemanager.adapt_defense(self.round_number, self.attack_detection_rate)
        
        # Aggregate using current defense
        aggregated = self.defensemanager.aggregate(updates)
        
        # Update global model
        self.global_dict = aggregated
        self.round_number += 1




