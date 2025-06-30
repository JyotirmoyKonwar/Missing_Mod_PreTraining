# --- models/encoders.py ---

import torch.nn as nn

class UAHTransformer(nn.Module):
    """Definition for the UAH-Driveset Transformer Encoder."""
    def __init__(self, input_dim=9, d_model=128, nhead=4, num_layers=3, num_classes=3):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.embed(x).permute(1, 0, 2)
        out = self.transformer(h)
        cls_token = out[0]
        logits = self.classifier(cls_token)
        return logits, cls_token

class PhysionetTransformer(nn.Module):
    """Definition for the Physionet Transformer Encoder."""
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=3, num_classes=3):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.embed(x).permute(1, 0, 2)
        out = self.transformer(h)
        cls_token = out[0]
        logits = self.classifier(cls_token)
        return logits, cls_token