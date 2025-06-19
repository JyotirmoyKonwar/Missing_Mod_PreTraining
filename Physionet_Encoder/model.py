import torch
import torch.nn as nn

class PhysionetTransformer(nn.Module):
    """
    Transformer-based encoder for the Physionet dataset.
    This model is adapted to handle longer sequences and a different feature dimension.
    """
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=3, num_classes=3):
        """
        Initializes the PhysionetTransformer model.

        Args:
            input_dim (int): The number of features in the input data (default: 6).
            d_model (int): The dimensionality of the model's embeddings (default: 128).
            nhead (int): The number of heads in the multi-head attention mechanism (default: 4).
            num_layers (int): The number of Transformer encoder layers (default: 3).
            num_classes (int): The number of output classes for classification (default: 3).
        """
        super().__init__()
        # Project raw features to model dimension
        self.embed = nn.Linear(input_dim, d_model)

        # Stacked Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=False # Expects (Seq_Len, Batch, Features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head on pooled CLS-like token
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Features), e.g., [B, 1000, 6].

        Returns:
            torch.Tensor: The output logits of shape (Batch, num_classes).
            torch.Tensor: The extracted class token embedding of shape (Batch, d_model).
        """
        # x: [B, T=1000, 6]
        h = self.embed(x)               # -> [B, 1000, 128]
        h = h.permute(1, 0, 2)          # -> [T=1000, B, 128] for Transformer
        out = self.transformer(h)       # -> [T=1000, B, 128]
        cls_token = out[0]              # Use the output of the first time step as the summary token
        logits = self.classifier(cls_token) # -> [B, 3]
        return logits, cls_token