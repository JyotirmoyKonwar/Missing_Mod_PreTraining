# --- gradient_modulation.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassifierGuidedGradientModulation(nn.Module):
    """Implements CGGM for balanced multimodal learning based on NeurIPS 2024 concepts."""
    def __init__(self, shared_dim, num_classes, num_modalities=2, num_heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.num_modalities = num_modalities
        
        # Individual transformer-based classifiers for each modality
        self.modality_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=shared_dim, nhead=num_heads, 
                                        dim_feedforward=shared_dim*2, dropout=dropout, batch_first=True), num_layers=layers),
                nn.Linear(shared_dim, num_classes)
            ) for _ in range(num_modalities)
        ])
        self.modality_acc_history = [0.0] * num_modalities

    def forward(self, modality_embeddings, labels, main_model, criterion):
        mod_map = {'uah': 0, 'physio': 1}
        mod_predictions, classifier_inputs, available_indices = [], [], []

        # 1. Get predictions from individual modality classifiers
        for mod_name, embedding in modality_embeddings.items():
            valid_mask = embedding.abs().sum(dim=1) > 0
            if valid_mask.any():
                mod_idx = mod_map[mod_name]
                pred = self.modality_classifiers[mod_idx](embedding[valid_mask].unsqueeze(1))[:, 0]
                mod_predictions.append(pred)
                classifier_inputs.append((pred, labels[valid_mask]))
                available_indices.append(mod_idx)

        if not mod_predictions: return None, None, [1.0] * self.num_modalities

        # 2. Calculate classifier loss and backpropagate to get gradients
        classifier_loss = sum(criterion(pred, true_labels) for pred, true_labels in classifier_inputs)
        if isinstance(classifier_loss, torch.Tensor):
            classifier_loss.backward(retain_graph=True)

        # 3. Extract gradients and calculate performance metrics
        mod_grads = [next((p.grad.clone() for n, p in self.modality_classifiers[i][0].layers[-1].linear2.named_parameters() if 'weight' in n and p.grad is not None), None) for i in available_indices]
        fusion_grad = next((p.grad.clone() for n, p in main_model.classifier.named_parameters() if 'weight' in n and p.grad is not None), None)

        if fusion_grad is None or not any(g is not None for g in mod_grads):
            return classifier_loss, None, [1.0] * self.num_modalities

        # 4. Calculate modulation coefficients based on performance improvement
        current_accs = [(torch.argmax(pred, dim=1) == lbls).float().mean().item() for pred, lbls in classifier_inputs]
        acc_diffs = [current_accs[i] - self.modality_acc_history[idx] for i, idx in enumerate(available_indices)]
        for i, acc in enumerate(current_accs): self.modality_acc_history[available_indices[i]] = acc

        diff_sum = sum(acc_diffs) + 1e-8
        coeffs_map = {idx: max(0.1, min(2.0, (diff_sum - acc_diffs[i]) / diff_sum if diff_sum != 0 else 1.0)) for i, idx in enumerate(available_indices)}
        final_coeffs = [coeffs_map.get(i, 1.0) for i in range(self.num_modalities)]

        # 5. Calculate modulation loss based on gradient similarity
        similarities = [F.cosine_similarity(g.view(-1), fusion_grad.view(-1), dim=0) for g in mod_grads if g is not None]
        weighted_sim = sum(coeffs_map[idx] * sim for idx, sim in zip(available_indices, similarities))
        grad_mod_loss = np.sum(np.abs([c for c in final_coeffs if c != 1.0])) - weighted_sim

        return classifier_loss, grad_mod_loss / len(available_indices), final_coeffs

def apply_gradient_modulation(model, coefficients, modulation_factor=1.3):
    """Scales encoder gradients based on the calculated coefficients."""
    for i, mod_name in enumerate(['uah_encoder', 'physio_encoder']):
        for name, param in model.named_parameters():
            if mod_name in name and param.grad is not None:
                param.grad *= (coefficients[i] * modulation_factor)