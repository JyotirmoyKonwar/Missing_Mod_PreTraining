# --- loss.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import configs
from adaptive_loss import AdaptiveModalityLoss, VarianceAwareLoss # Import new modules

class MultiComponentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha, self.beta = 0.0, 0.0 # Set per stage
        self.gamma = configs.GAMMA
        self.delta = configs.DELTA
        self.epsilon = configs.EPSILON
        self.temperature = configs.TEMPERATURE
        
        # -- SOLUTION 2: INSTANTIATE ADAPTIVE LOSS MODULES --
        self.adaptive_loss = AdaptiveModalityLoss(num_modalities=3)
        self.variance_loss = VarianceAwareLoss(temperature=self.temperature)

    def _info_nce_loss(self, z1, z2):
        z1_norm, z2_norm = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        similarity = torch.mm(z1_norm, z2_norm.t()) / self.temperature
        return F.cross_entropy(similarity, torch.arange(z1.size(0), device=z1.device))

    def _auxiliary_loss(self, aux_logits, labels, modality_mask):
        aux_loss = 0.0
        if 'uah' in aux_logits: aux_loss += F.cross_entropy(aux_logits['uah'], labels[modality_mask[:, 0]])
        if 'physio' in aux_logits: aux_loss += F.cross_entropy(aux_logits['physio'], labels[modality_mask[:, 1]])
        return aux_loss

    def _reconstruction_loss(self, uah_emb, physio_emb, model):
        recon_physio = model.uah_to_physio_decoder(uah_emb)
        recon_uah = model.physio_to_uah_decoder(physio_emb)
        return F.mse_loss(recon_physio, physio_emb) + F.mse_loss(recon_uah, uah_emb)

    def forward(self, model, main_logits, aux_logits, labels, shared_embedding, modality_embeddings, modality_mask):
        # -- SOLUTION 2: ADAPTIVE CLASSIFICATION LOSS --
        both_mask = modality_mask.all(dim=1)
        uah_only_mask = modality_mask[:, 0] & ~modality_mask[:, 1]
        physio_only_mask = ~modality_mask[:, 0] & modality_mask[:, 1]
        
        scenario_losses = [
            F.cross_entropy(main_logits[both_mask], labels[both_mask]) if both_mask.any() else None,
            F.cross_entropy(main_logits[uah_only_mask], labels[uah_only_mask]) if uah_only_mask.any() else None,
            F.cross_entropy(main_logits[physio_only_mask], labels[physio_only_mask]) if physio_only_mask.any() else None
        ]
        classification_loss = self.adaptive_loss(scenario_losses)
        
        # -- SOLUTION 2: VARIANCE-AWARE LOSS --
        variance_loss, variances = self.variance_loss(modality_embeddings, labels, F.cross_entropy)

        # -- Original Loss Components --
        aux_loss = self._auxiliary_loss(aux_logits, labels, modality_mask)
        reg_loss = shared_embedding.norm(p=2, dim=1).mean()

        contrastive_loss, consistency_loss, recon_loss = 0.0, 0.0, 0.0
        if both_mask.any():
            uah_both, physio_both = modality_embeddings['uah'][both_mask], modality_embeddings['physio'][both_mask]
            contrastive_loss = self._info_nce_loss(uah_both, physio_both)
            consistency_loss = F.mse_loss(uah_both, physio_both)
            recon_loss = self._reconstruction_loss(uah_both, physio_both, model)
        
        total_loss = (classification_loss +
                     self.delta * aux_loss +
                     self.alpha * contrastive_loss +
                     self.beta * consistency_loss +
                     self.epsilon * recon_loss +
                     self.gamma * reg_loss +
                     0.1 * variance_loss) # Add variance loss with small weight
        
        return {
            'total': total_loss, 'classification': classification_loss, 'auxiliary': aux_loss,
            'contrastive': contrastive_loss, 'consistency': consistency_loss,
            'reconstruction': recon_loss, 'regularization': reg_loss,
            'variance': variance_loss, 'adaptive_weights': self.adaptive_loss.get_current_weights(),
            'modality_variances': variances
        }