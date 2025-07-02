# --- modality_utilization.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityUtilizationRegulator(nn.Module):
    """Regulates modality utilization to ensure balanced learning early in training."""
    def __init__(self, regulation_strength=0.1, prime_window_epochs=10):
        super().__init__()
        self.regulation_strength = regulation_strength
        self.prime_window_epochs = prime_window_epochs

    def _calculate_info_acquisition(self, modality_embeddings):
        """Calculates a proxy for information acquisition using prediction entropy."""
        info_rates = {}
        for mod_name, embeddings in modality_embeddings.items():
            valid_mask = embeddings.abs().sum(dim=1) > 0
            if valid_mask.any():
                probs = F.softmax(embeddings[valid_mask], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                info_rates[mod_name] = torch.mean(entropy).item()
        return info_rates

    def forward(self, modality_embeddings, epoch):
        """Returns regulation factors to scale gradients."""
        info_rates = self._calculate_info_acquisition(modality_embeddings)
        regulation_factors = {'uah': 1.0, 'physio': 1.0}

        # Apply regulation only during the prime learning window
        if epoch <= self.prime_window_epochs and len(info_rates) == 2:
            uah_rate, physio_rate = info_rates.get('uah', 0), info_rates.get('physio', 0)
            # Slow down the dominant modality and boost the weaker one
            if uah_rate > physio_rate:
                regulation_factors['uah'] = 1.0 - self.regulation_strength
                regulation_factors['physio'] = 1.0 + self.regulation_strength
            else:
                regulation_factors['uah'] = 1.0 + self.regulation_strength
                regulation_factors['physio'] = 1.0 - self.regulation_strength
        
        return regulation_factors, info_rates

def apply_utilization_regulation(model, regulation_factors):
    """Applies regulation by scaling encoder gradients."""
    for mod_name, factor in regulation_factors.items():
        encoder_name = f"{mod_name}_encoder"
        for name, param in model.named_parameters():
            if encoder_name in name and param.grad is not None:
                param.grad *= factor