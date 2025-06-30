# --- loss.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import configs

class MultiComponentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha, self.beta = 0.0, 0.0
        self.gamma = configs.GAMMA
        self.delta = configs.DELTA
        self.epsilon = configs.EPSILON
        self.temperature = configs.TEMPERATURE

    def _info_nce_loss(self, z1, z2):
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        similarity_matrix = torch.mm(z1_norm, z2_norm.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(similarity_matrix, labels)

    def _consistency_loss(self, uah_emb, physio_emb):
        return F.mse_loss(uah_emb, physio_emb)

    def _regularization_loss(self, shared_embedding, modality_embeddings):
        reg_loss = shared_embedding.norm(p=2, dim=1).mean()
        for emb in modality_embeddings.values():
            reg_loss += emb.norm(p=2, dim=1).mean()
        return reg_loss / (len(modality_embeddings) + 1)

    def _auxiliary_loss(self, aux_logits, labels, modality_mask):
        aux_loss = 0.0
        if 'uah' in aux_logits and modality_mask[:, 0].any():
            aux_loss += F.cross_entropy(aux_logits['uah'], labels[modality_mask[:, 0]])
        if 'physio' in aux_logits and modality_mask[:, 1].any():
            aux_loss += F.cross_entropy(aux_logits['physio'], labels[modality_mask[:, 1]])
        return aux_loss

    def _reconstruction_loss(self, modality_embeddings, model):
        uah_emb = modality_embeddings['uah']
        physio_emb = modality_embeddings['physio']
        reconstructed_physio = model.uah_to_physio_decoder(uah_emb)
        reconstructed_uah = model.physio_to_uah_decoder(physio_emb)
        return F.mse_loss(reconstructed_physio, physio_emb) + F.mse_loss(reconstructed_uah, uah_emb)

    def forward(self, model, main_logits, aux_logits, labels, shared_embedding, modality_embeddings, modality_mask):
        # CORRECTED: Classification loss is now uniform and unbiased.
        classification_loss = F.cross_entropy(main_logits, labels)

        aux_loss = self._auxiliary_loss(aux_logits, labels, modality_mask)

        both_present = modality_mask.all(dim=1)
        contrastive_loss, consistency_loss, recon_loss = 0.0, 0.0, 0.0
        if both_present.any():
            uah_both = modality_embeddings['uah'][both_present]
            physio_both = modality_embeddings['physio'][both_present]
            contrastive_loss = self._info_nce_loss(uah_both, physio_both)
            consistency_loss = self._consistency_loss(uah_both, physio_both)
            recon_loss = self._reconstruction_loss({'uah': uah_both, 'physio': physio_both}, model)

        reg_loss = self._regularization_loss(shared_embedding, modality_embeddings)

        total_loss = (classification_loss +
                     self.delta * aux_loss +
                     self.alpha * contrastive_loss +
                     self.beta * consistency_loss +
                     self.epsilon * recon_loss +
                     self.gamma * reg_loss)

        return {
            'total': total_loss, 'classification': classification_loss, 'auxiliary': aux_loss,
            'contrastive': contrastive_loss, 'consistency': consistency_loss,
            'reconstruction': recon_loss, 'regularization': reg_loss
        }