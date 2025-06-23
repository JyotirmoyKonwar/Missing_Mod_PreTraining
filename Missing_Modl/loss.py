# --- loss.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import configs

class MultiComponentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = configs.GAMMA
        self.temperature = configs.TEMPERATURE
        
    def _info_nce_loss(self, z1, z2):
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        similarity_matrix = torch.mm(z1_norm, z2_norm.t()) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(similarity_matrix, labels)
    
    def _consistency_loss(self, uah_emb, physio_emb):
        return F.mse_loss(uah_emb, physio_emb)
    
    def _regularization_loss(self, shared_embedding, modality_embeddings):
        reg_loss = shared_embedding.norm(2) / shared_embedding.size(0)
        for emb in modality_embeddings.values():
            reg_loss += emb.norm(2) / emb.size(0)
        return reg_loss
    
    def forward(self, logits, labels, shared_embedding, modality_embeddings, modality_mask):
        classification_loss = F.cross_entropy(logits, labels)
        
        both_present = modality_mask.all(dim=1)
        contrastive_loss = 0.0
        consistency_loss = 0.0

        if both_present.any():
            uah_emb = modality_embeddings['uah'][both_present]
            physio_emb = modality_embeddings['physio'][both_present]
            contrastive_loss = self._info_nce_loss(uah_emb, physio_emb)
            consistency_loss = self._consistency_loss(uah_emb, physio_emb)
            
        reg_loss = self._regularization_loss(shared_embedding, modality_embeddings)
        
        total_loss = (classification_loss + 
                     self.alpha * contrastive_loss + 
                     self.beta * consistency_loss + 
                     self.gamma * reg_loss)
        
        return {
            'total': total_loss, 'classification': classification_loss, 'contrastive': contrastive_loss,
            'consistency': consistency_loss, 'regularization': reg_loss
        }