# --- adaptive_loss.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveModalityLoss(nn.Module):
    """Automatically weighted multi-modality loss using uncertainty."""
    def __init__(self, num_modalities=3):
        super(AdaptiveModalityLoss, self).__init__()
        # Learnable parameters for [both, uah_only, physio_only] scenarios
        self.log_vars = nn.Parameter(torch.zeros(num_modalities, requires_grad=True))

    def forward(self, losses):
        """
        Args:
            losses: A list of loss tensors for [both, uah_only, physio_only].
        """
        precision = torch.exp(-self.log_vars)
        weighted_loss = 0.0
        num_valid_losses = 0
        for i, loss in enumerate(losses):
            # A loss is valid if it's not None and is a finite number
            if loss is not None and loss.numel() > 0 and not torch.isnan(loss) and not torch.isinf(loss):
                weighted_loss += precision[i] * loss + self.log_vars[i]
                num_valid_losses += 1
        
        return weighted_loss / num_valid_losses if num_valid_losses > 0 else 0.0

    def get_current_weights(self):
        """Returns the normalized weights for monitoring purposes."""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars)
            weights = weights / weights.sum()
            return weights.cpu().numpy()

class VarianceAwareLoss(nn.Module):
    """Calculates a variance-aware loss to help balance modality contributions."""
    def __init__(self, temperature=0.1):
        super(VarianceAwareLoss, self).__init__()
        self.temperature = temperature

    def forward(self, modality_embeddings, labels, base_loss_fn):
        """
        Args:
            modality_embeddings: A dict with 'uah' and 'physio' embeddings.
            labels: Ground truth labels.
            base_loss_fn: The base loss function (e.g., CrossEntropyLoss).
        """
        losses, variances = {}, {}

        for mod_name, embeddings in modality_embeddings.items():
            valid_mask = embeddings.abs().sum(dim=1) > 0
            if valid_mask.any():
                logits = embeddings[valid_mask]
                current_labels = labels[valid_mask]
                if logits.shape[0] == 0 or current_labels.shape[0] == 0: continue

                losses[mod_name] = base_loss_fn(logits, current_labels)
                probs = F.softmax(logits / self.temperature, dim=1)
                variances[mod_name] = torch.var(probs, dim=1).mean()

        if not losses: return 0.0, {}

        # Weight loss inversely to prediction variance (more weight to confused modality)
        if len(losses) == 2:
            uah_var = variances.get('uah', torch.tensor(1.0, device=labels.device))
            physio_var = variances.get('physio', torch.tensor(1.0, device=labels.device))
            total_var = uah_var + physio_var + 1e-8
            uah_weight = physio_var / total_var
            physio_weight = uah_var / total_var
            weighted_loss = (uah_weight * losses.get('uah', 0) + physio_weight * losses.get('physio', 0))
        else:
            weighted_loss = sum(losses.values())

        return weighted_loss, {k: v.item() for k, v in variances.items()}