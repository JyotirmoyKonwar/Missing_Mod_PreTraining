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

        # -- SOLUTION 4: USE MODALITY-SPECIFIC UNFREEZING SCHEDULES --
        if configs.ENABLE_MODALITY_SPECIFIC_UNFREEZING:
            self.uah_unfreeze_schedule = configs.UAH_UNFREEZE_SCHEDULE
            self.physio_unfreeze_schedule = configs.PHYSIO_UNFREEZE_SCHEDULE
        else:
            self.unfreeze_schedule = { 10: ['classifier'], 20: ['transformer.layers.2'], 30: ['transformer.layers.1', 'transformer.layers.0'], 50: ['embed'] }

        for param in self.uah_encoder.parameters(): param.requires_grad = False
        for param in self.physio_encoder.parameters(): param.requires_grad = False
        
        self.uah_to_shared = nn.Sequential(nn.Linear(configs.ENCODER_DIM, configs.SHARED_DIM), nn.LayerNorm(configs.SHARED_DIM), nn.ReLU(), nn.Dropout(0.1))
        self.physio_to_shared = nn.Sequential(nn.Linear(configs.ENCODER_DIM, configs.SHARED_DIM), nn.LayerNorm(configs.SHARED_DIM), nn.ReLU(), nn.Dropout(0.1))
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim=configs.SHARED_DIM, num_heads=configs.ATTENTION_HEADS, dropout=0.1, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=configs.SHARED_DIM, nhead=configs.ATTENTION_HEADS, dim_feedforward=configs.SHARED_DIM*2, dropout=0.1, batch_first=True)
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fusion_gate = nn.Sequential(nn.Linear(configs.SHARED_DIM*2, configs.SHARED_DIM), nn.Tanh(), nn.Linear(configs.SHARED_DIM, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(configs.SHARED_DIM, configs.HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(configs.HIDDEN_DIM, configs.HIDDEN_DIM//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(configs.HIDDEN_DIM//2, configs.NUM_CLASSES)
        )
        self.uah_aux_classifier = nn.Sequential(nn.Linear(configs.SHARED_DIM, configs.HIDDEN_DIM // 2), nn.ReLU(), nn.Linear(configs.HIDDEN_DIM // 2, configs.NUM_CLASSES))
        self.physio_aux_classifier = nn.Sequential(nn.Linear(configs.SHARED_DIM, configs.HIDDEN_DIM // 2), nn.ReLU(), nn.Linear(configs.HIDDEN_DIM // 2, configs.NUM_CLASSES))
        self.uah_to_physio_decoder = nn.Sequential(nn.Linear(configs.SHARED_DIM, configs.SHARED_DIM), nn.ReLU(), nn.Linear(configs.SHARED_DIM, configs.SHARED_DIM))
        self.physio_to_uah_decoder = nn.Sequential(nn.Linear(configs.SHARED_DIM, configs.SHARED_DIM), nn.ReLU(), nn.Linear(configs.SHARED_DIM, configs.SHARED_DIM))

    def _load_pretrained_encoder(self, model_path, modality_type):
        try:
            encoder = UAHTransformer() if modality_type == 'uah' else PhysionetTransformer()
            encoder.load_state_dict(torch.load(model_path, map_location=configs.DEVICE))
            encoder.classifier = nn.Identity()
            print(f"Successfully loaded pretrained {modality_type} encoder.")
            return encoder
        except FileNotFoundError:
            print(f"FATAL: Pretrained model not found at {model_path}. Exiting."); exit()

    def progressive_unfreeze(self, epoch):
        """Modality-specific progressive unfreezing."""
        unfroze_params = False
        if not configs.ENABLE_MODALITY_SPECIFIC_UNFREEZING:
            if epoch in self.unfreeze_schedule:
                patterns = self.unfreeze_schedule[epoch]
                print(f"\nEpoch {epoch}: Unfreezing layers: {patterns}")
                for p in patterns: self._unfreeze_specific_layers(self.uah_encoder, p); self._unfreeze_specific_layers(self.physio_encoder, p)
                unfroze_params = True
            return unfroze_params

        if epoch in self.uah_unfreeze_schedule:
            patterns = self.uah_unfreeze_schedule[epoch]
            print(f"\nEpoch {epoch}: Unfreezing UAH layers: {patterns}")
            for p in patterns: self._unfreeze_specific_layers(self.uah_encoder, p)
            unfroze_params = True
        if epoch in self.physio_unfreeze_schedule:
            patterns = self.physio_unfreeze_schedule[epoch]
            print(f"\nEpoch {epoch}: Unfreezing Physio layers: {patterns}")
            for p in patterns: self._unfreeze_specific_layers(self.physio_encoder, p)
            unfroze_params = True
        return unfroze_params

    def _unfreeze_specific_layers(self, encoder, pattern):
        for name, param in encoder.named_parameters():
            if pattern in name: param.requires_grad = True
    
    def get_modality_frozen_status(self):
        """Returns the ratio of frozen parameters for each modality encoder."""
        uah_frozen = sum(1 for p in self.uah_encoder.parameters() if not p.requires_grad)
        physio_frozen = sum(1 for p in self.physio_encoder.parameters() if not p.requires_grad)
        return {
            'uah': f"{(uah_frozen / len(list(self.uah_encoder.parameters()))):.2%}",
            'physio': f"{(physio_frozen / len(list(self.physio_encoder.parameters()))):.2%}"
        }

    def forward(self, uah_seq, physio_seq, modality_mask):
        batch_size = uah_seq.size(0)
        uah_mask, physio_mask = modality_mask[:, 0], modality_mask[:, 1]
        full_uah = torch.zeros(batch_size, configs.SHARED_DIM, device=configs.DEVICE)
        full_physio = torch.zeros(batch_size, configs.SHARED_DIM, device=configs.DEVICE)
        aux_logits = {}

        if uah_mask.any():
            with torch.set_grad_enabled(any(p.requires_grad for p in self.uah_encoder.parameters())):
                _, uah_raw = self.uah_encoder(uah_seq[uah_mask])
            uah_shared = self.uah_to_shared(uah_raw)
            aux_logits['uah'] = self.uah_aux_classifier(uah_shared)
            full_uah[uah_mask] = uah_shared
        
        if physio_mask.any():
            with torch.set_grad_enabled(any(p.requires_grad for p in self.physio_encoder.parameters())):
                _, physio_raw = self.physio_encoder(physio_seq[physio_mask])
            physio_shared = self.physio_to_shared(physio_raw)
            aux_logits['physio'] = self.physio_aux_classifier(physio_shared)
            full_physio[physio_mask] = physio_shared
            
        modality_embeddings = {'uah': full_uah, 'physio': full_physio}
        
        # --- CORRECTED FUSION LOGIC ---
        both_mask = uah_mask & physio_mask
        uah_only_mask = uah_mask & ~physio_mask
        physio_only_mask = ~uah_mask & physio_mask

        fused = torch.zeros_like(full_uah)
        if both_mask.any():
            uah, physio = full_uah[both_mask].unsqueeze(1), full_physio[both_mask].unsqueeze(1)
            uah_att, _ = self.cross_modal_attention(uah, physio, physio)
            physio_att, _ = self.cross_modal_attention(physio, uah, uah)
            gate = self.fusion_gate(torch.cat([uah_att.squeeze(1), physio_att.squeeze(1)], dim=1))
            fused[both_mask] = (gate * uah_att.squeeze(1) + (1 - gate) * physio_att.squeeze(1))

        if uah_only_mask.any(): fused[uah_only_mask] = full_uah[uah_only_mask]
        if physio_only_mask.any(): fused[physio_only_mask] = full_physio[physio_only_mask]

        refined_shared = self.shared_transformer(fused.unsqueeze(1)).squeeze(1)
        main_logits = self.classifier(refined_shared)

        return main_logits, aux_logits, refined_shared, modality_embeddings