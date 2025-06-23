# --- models/fusion.py ---

import torch
import torch.nn as nn
import configs
from .encoders import UAHTransformer, PhysionetTransformer

class SharedLatentSpaceFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pretrained encoders
        self.uah_encoder = self._load_pretrained_encoder(configs.UAH_MODEL_PATH, 'uah')
        self.physio_encoder = self._load_pretrained_encoder(configs.PHYSIO_MODEL_PATH, 'physio')
        
        # Freeze pretrained encoder parameters
        for param in self.uah_encoder.parameters():
            param.requires_grad = False
        for param in self.physio_encoder.parameters():
            param.requires_grad = False
        
        # Modality-specific projectors
        self.uah_to_shared = nn.Sequential(
            nn.Linear(configs.ENCODER_DIM, configs.SHARED_DIM), nn.LayerNorm(configs.SHARED_DIM), nn.ReLU(), nn.Dropout(0.1)
        )
        self.physio_to_shared = nn.Sequential(
            nn.Linear(configs.ENCODER_DIM, configs.SHARED_DIM), nn.LayerNorm(configs.SHARED_DIM), nn.ReLU(), nn.Dropout(0.1)
        )
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=configs.SHARED_DIM, num_heads=configs.ATTENTION_HEADS, dropout=0.1, batch_first=True
        )
        
        # Shared space transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.SHARED_DIM, nhead=configs.ATTENTION_HEADS, dim_feedforward=configs.SHARED_DIM * 2, dropout=0.1, batch_first=True
        )
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Adaptive fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(configs.SHARED_DIM * 2, configs.SHARED_DIM), nn.Tanh(), nn.Linear(configs.SHARED_DIM, 1), nn.Sigmoid()
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(configs.SHARED_DIM, configs.HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(configs.HIDDEN_DIM, configs.HIDDEN_DIM // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(configs.HIDDEN_DIM // 2, configs.NUM_CLASSES)
        )
        
    def _load_pretrained_encoder(self, model_path, modality_type):
        try:
            encoder = UAHTransformer() if modality_type == 'uah' else PhysionetTransformer()
            encoder.load_state_dict(torch.load(model_path, map_location=configs.DEVICE))
            encoder.classifier = nn.Identity() # Remove classification head
            print(f"Successfully loaded pretrained {modality_type} encoder.")
            return encoder
        except FileNotFoundError:
            print(f"Warning: Pretrained model not found at {model_path}. Using a randomly initialized encoder.")
            encoder = UAHTransformer() if modality_type == 'uah' else PhysionetTransformer()
            encoder.classifier = nn.Identity()
            return encoder

    def forward(self, uah_seq, physio_seq, modality_mask):
        batch_size = uah_seq.size(0)
        
        modality_embeddings = {}
        
        # Process UAH modality if available
        if modality_mask[:, 0].any():
            with torch.no_grad():
                _, uah_raw = self.uah_encoder(uah_seq[modality_mask[:, 0]])
            uah_shared = self.uah_to_shared(uah_raw)
            # Create a full-size tensor and place embeddings at correct indices
            full_uah_shared = torch.zeros(batch_size, configs.SHARED_DIM, device=configs.DEVICE)
            full_uah_shared[modality_mask[:, 0]] = uah_shared
            modality_embeddings['uah'] = full_uah_shared
        
        # Process Physio modality if available
        if modality_mask[:, 1].any():
            with torch.no_grad():
                _, physio_raw = self.physio_encoder(physio_seq[modality_mask[:, 1]])
            physio_shared = self.physio_to_shared(physio_raw)
            full_physio_shared = torch.zeros(batch_size, configs.SHARED_DIM, device=configs.DEVICE)
            full_physio_shared[modality_mask[:, 1]] = physio_shared
            modality_embeddings['physio'] = full_physio_shared
            
        # Fusion based on modality availability
        both_present = modality_mask.all(dim=1)
        uah_only = modality_mask[:, 0] & ~modality_mask[:, 1]
        physio_only = ~modality_mask[:, 0] & modality_mask[:, 1]

        fused_shared = torch.zeros(batch_size, configs.SHARED_DIM, device=configs.DEVICE)

        if both_present.any():
            uah_emb = modality_embeddings['uah'][both_present]
            physio_emb = modality_embeddings['physio'][both_present]
            
            uah_attended, _ = self.cross_modal_attention(uah_emb.unsqueeze(1), physio_emb.unsqueeze(1), physio_emb.unsqueeze(1))
            physio_attended, _ = self.cross_modal_attention(physio_emb.unsqueeze(1), uah_emb.unsqueeze(1), uah_emb.unsqueeze(1))
            
            concat_features = torch.cat([uah_attended.squeeze(1), physio_attended.squeeze(1)], dim=1)
            fusion_weight = self.fusion_gate(concat_features)
            fused_both = (fusion_weight * uah_attended.squeeze(1) + (1 - fusion_weight) * physio_attended.squeeze(1))
            fused_shared[both_present] = fused_both

        if uah_only.any():
            fused_shared[uah_only] = modality_embeddings['uah'][uah_only]

        if physio_only.any():
            fused_shared[physio_only] = modality_embeddings['physio'][physio_only]

        # Shared space refinement and classification
        refined_shared = self.shared_transformer(fused_shared.unsqueeze(1)).squeeze(1)
        logits = self.classifier(refined_shared)
        
        return logits, refined_shared, modality_embeddings