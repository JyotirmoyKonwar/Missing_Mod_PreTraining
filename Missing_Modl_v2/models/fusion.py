# --- models/fusion.py ---

import torch
import torch.nn as nn
import configs
from .encoders import UAHTransformer, PhysionetTransformer

class SharedLatentSpaceFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.uah_encoder = self._load_pretrained_encoder(configs.UAH_MODEL_PATH, 'uah')
        self.physio_encoder = self._load_pretrained_encoder(configs.PHYSIO_MODEL_PATH, 'physio')
        
        # FIX 2: Corrected unfreeze schedule with valid layer names and reachable epochs
        self.unfreeze_schedule = {
            10: ['transformer.layers.2'],  # last encoder block
            20: ['transformer.layers.1'],  # mid block
            30: ['transformer.layers.0'],  # first block
            40: ['embed']                  # input token projector
        }
        
        for param in self.uah_encoder.parameters(): param.requires_grad = False
        for param in self.physio_encoder.parameters(): param.requires_grad = False
        
        # --- Main Fusion Components ---
        self.uah_to_shared = nn.Sequential(
            nn.Linear(configs.ENCODER_DIM, configs.SHARED_DIM), nn.LayerNorm(configs.SHARED_DIM), nn.ReLU(), nn.Dropout(0.1)
        )
        self.physio_to_shared = nn.Sequential(
            nn.Linear(configs.ENCODER_DIM, configs.SHARED_DIM), nn.LayerNorm(configs.SHARED_DIM), nn.ReLU(), nn.Dropout(0.1)
        )
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=configs.SHARED_DIM, num_heads=configs.ATTENTION_HEADS, dropout=0.1, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.SHARED_DIM, nhead=configs.ATTENTION_HEADS, dim_feedforward=configs.SHARED_DIM * 2, dropout=0.1, batch_first=True
        )
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fusion_gate = nn.Sequential(
            nn.Linear(configs.SHARED_DIM * 2, configs.SHARED_DIM), nn.Tanh(), nn.Linear(configs.SHARED_DIM, 1), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(configs.SHARED_DIM, configs.HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(configs.HIDDEN_DIM, configs.HIDDEN_DIM // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(configs.HIDDEN_DIM // 2, configs.NUM_CLASSES)
        )

        # --- Auxiliary and Reconstruction Components ---
        self.uah_aux_classifier = nn.Sequential(
            nn.Linear(configs.SHARED_DIM, configs.HIDDEN_DIM // 2), nn.ReLU(),
            nn.Linear(configs.HIDDEN_DIM // 2, configs.NUM_CLASSES)
        )
        self.physio_aux_classifier = nn.Sequential(
            nn.Linear(configs.SHARED_DIM, configs.HIDDEN_DIM // 2), nn.ReLU(),
            nn.Linear(configs.HIDDEN_DIM // 2, configs.NUM_CLASSES)
        )
        self.uah_to_physio_decoder = nn.Sequential(
            nn.Linear(configs.SHARED_DIM, configs.SHARED_DIM), nn.ReLU(),
            nn.Linear(configs.SHARED_DIM, configs.SHARED_DIM)
        )
        self.physio_to_uah_decoder = nn.Sequential(
            nn.Linear(configs.SHARED_DIM, configs.SHARED_DIM), nn.ReLU(),
            nn.Linear(configs.SHARED_DIM, configs.SHARED_DIM)
        )
        
    def _load_pretrained_encoder(self, model_path, modality_type):
        try:
            encoder = UAHTransformer() if modality_type == 'uah' else PhysionetTransformer()
            encoder.load_state_dict(torch.load(model_path, map_location=configs.DEVICE))
            # The 'classifier' layer is replaced, so unfreezing it by name won't work.
            encoder.classifier = nn.Identity()
            print(f"Successfully loaded pretrained {modality_type} encoder.")
            return encoder
        except FileNotFoundError:
            print(f"FATAL: Pretrained model not found at {model_path}. Please update configs.py.")
            exit()

    def progressive_unfreeze(self, epoch):
        """Unfreezes parts of the encoders based on the current epoch."""
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            print(f"\nEpoch {epoch}: Unfreezing layers containing: {layers_to_unfreeze}")
            # Ensure layers_to_unfreeze is a list
            if not isinstance(layers_to_unfreeze, list):
                layers_to_unfreeze = [layers_to_unfreeze]

            for pattern in layers_to_unfreeze:
                self._unfreeze_specific_layers(self.uah_encoder, pattern)
                self._unfreeze_specific_layers(self.physio_encoder, pattern)
            return True 
        return False

    def _unfreeze_specific_layers(self, encoder, layer_pattern):
        for name, param in encoder.named_parameters():
            if layer_pattern in name:
                param.requires_grad = True

    def forward(self, uah_seq, physio_seq, modality_mask):
        batch_size = uah_seq.size(0)
        modality_embeddings = {}
        aux_logits = {}

        uah_present_mask = modality_mask[:, 0]
        physio_present_mask = modality_mask[:, 1]
        
        # Initialize full batch tensors
        full_uah = torch.zeros(batch_size, configs.SHARED_DIM, device=configs.DEVICE)
        full_physio = torch.zeros(batch_size, configs.SHARED_DIM, device=configs.DEVICE)

        if uah_present_mask.any():
            # Use set_grad_enabled for safety, though requires_grad should handle it
            with torch.set_grad_enabled(any(p.requires_grad for p in self.uah_encoder.parameters())):
                _, uah_raw = self.uah_encoder(uah_seq[uah_present_mask])
            uah_shared = self.uah_to_shared(uah_raw)
            aux_logits['uah'] = self.uah_aux_classifier(uah_shared)
            full_uah[uah_present_mask] = uah_shared
        
        if physio_present_mask.any():
            with torch.set_grad_enabled(any(p.requires_grad for p in self.physio_encoder.parameters())):
                _, physio_raw = self.physio_encoder(physio_seq[physio_present_mask])
            physio_shared = self.physio_to_shared(physio_raw)
            aux_logits['physio'] = self.physio_aux_classifier(physio_shared)
            full_physio[physio_present_mask] = physio_shared

        # Always populate modality_embeddings for consistent loss calculation
        modality_embeddings['uah'] = full_uah
        modality_embeddings['physio'] = full_physio
        
        both_mask = uah_present_mask & physio_present_mask
        uah_only_mask = uah_present_mask & ~physio_present_mask
        physio_only_mask = ~uah_present_mask & physio_present_mask

        fused_shared = torch.zeros_like(full_uah)

        if both_mask.any():
            uah_emb, physio_emb = full_uah[both_mask].unsqueeze(1), full_physio[both_mask].unsqueeze(1)
            uah_attended, _ = self.cross_modal_attention(uah_emb, physio_emb, physio_emb)
            physio_attended, _ = self.cross_modal_attention(physio_emb, uah_emb, uah_emb)
            concat_features = torch.cat([uah_attended.squeeze(1), physio_attended.squeeze(1)], dim=1)
            fusion_weight = self.fusion_gate(concat_features)
            fused_shared[both_mask] = (fusion_weight * uah_attended.squeeze(1) + (1 - fusion_weight) * physio_attended.squeeze(1))
        
        if uah_only_mask.any():
            fused_shared[uah_only_mask] = full_uah[uah_only_mask]
        
        if physio_only_mask.any():
            fused_shared[physio_only_mask] = full_physio[physio_only_mask]

        refined_shared = self.shared_transformer(fused_shared.unsqueeze(1)).squeeze(1)
        main_logits = self.classifier(refined_shared)
        
        return main_logits, aux_logits, refined_shared, modality_embeddings