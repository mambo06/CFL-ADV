from train import Client
import torch

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