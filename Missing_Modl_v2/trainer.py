# --- trainer.py ---

import torch
import torch.optim as optim
from tqdm import tqdm
import configs
from loss import MultiComponentLoss
import numpy as np # For metrics logging

class MissingModalityTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(configs.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        # FIX 1: Add a global epoch counter for the unfreezing schedule
        self.global_epoch = 0 
        self.loss_fn = MultiComponentLoss()
        
        fusion_params = [p for n, p in self.model.named_parameters() if 'encoder' not in n and p.requires_grad]
        self.fusion_optimizer = optim.Adam(fusion_params, lr=configs.INITIAL_LR)
        self.encoder_optimizer = None

    def _update_encoder_optimizer(self):
        """Creates or updates the encoder optimizer with modality-specific learning rates."""
        # Get all currently trainable parameters for each encoder separately
        uah_params = [p for n, p in self.model.uah_encoder.named_parameters() if p.requires_grad]
        physio_params = [p for n, p in self.model.physio_encoder.named_parameters() if p.requires_grad]

        # If no encoder parameters are trainable yet, do nothing.
        if not uah_params and not physio_params:
            return

        # Prepare the parameter groups with their modality-specific LRs from configs.py
        param_groups_to_set = []
        if uah_params:
            param_groups_to_set.append({
                'params': uah_params,
                'lr': configs.INITIAL_LR * configs.UAH_LR_FACTOR, #
                'name': 'uah_encoder'
            })
        if physio_params:
            param_groups_to_set.append({
                'params': physio_params,
                'lr': configs.INITIAL_LR * configs.PHYSIO_LR_FACTOR, #
                'name': 'physio_encoder'
            })

        if not param_groups_to_set:
            return

        if self.encoder_optimizer is None:
            print("Creating encoder optimizer for the first time with modality-specific LRs.")
            self.encoder_optimizer = optim.Adam(param_groups_to_set)
        else:
            print("Updating encoder optimizer with newly unfrozen parameters and modality-specific LRs.")
            # Directly assigning to param_groups is the most reliable way to update an optimizer
            self.encoder_optimizer.param_groups = param_groups_to_set

    def run_training_stage(self, stage_num, epochs, alpha, beta, model_save_path):
        print(f"\n--- Starting Stage {stage_num}: {epochs} epochs ---")
        
        self.loss_fn.alpha, self.loss_fn.beta = alpha, beta
        # FIX 5b: Removed unused is_stratified flag
        
        best_val_acc, min_loss_for_best_acc = 0.0, float('inf')

        for epoch in range(1, epochs + 1):
            # FIX 1: Increment and use the global epoch counter
            self.global_epoch += 1
            if self.model.progressive_unfreeze(self.global_epoch):
                 self._update_encoder_optimizer()
            
            train_metrics = self._train_epoch()
            val_metrics = self._validate_epoch()

            print(f"[Stage {stage_num}] Ep {epoch}/{epochs} (Global: {self.global_epoch}) | Train Loss: {train_metrics['total']:.4f} | Val Acc: {val_metrics['accuracy']:.3f}")

            current_accuracy = val_metrics['accuracy']
            current_train_loss = train_metrics['total']
            
            if current_accuracy > best_val_acc or (abs(current_accuracy - best_val_acc) < 1e-5 and current_train_loss < min_loss_for_best_acc):
                best_val_acc = current_accuracy
                min_loss_for_best_acc = current_train_loss
                print(f"-> New best performance in stage. Saving model to {model_save_path}")
                torch.save(self.model.state_dict(), model_save_path)

    def _train_epoch(self):
        self.model.train()
        total_metrics = {k: 0.0 for k in ['total', 'classification', 'auxiliary', 'contrastive', 'consistency', 'reconstruction', 'regularization']}
        
        for uah, physio, mask, labels in tqdm(self.train_loader, desc=f"Training"):
            uah, physio, mask, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), mask.to(configs.DEVICE), labels.to(configs.DEVICE)
            
            self.fusion_optimizer.zero_grad()
            if self.encoder_optimizer:
                self.encoder_optimizer.zero_grad()
            
            main_logits, aux_logits, shared_emb, modality_embs = self.model(uah, physio, mask)
            loss_dict = self.loss_fn(self.model, main_logits, aux_logits, labels, shared_emb, modality_embs, mask)
            
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_((p for p in self.model.parameters() if p.requires_grad), max_norm=1.0)
            
            self.fusion_optimizer.step()
            if self.encoder_optimizer:
                self.encoder_optimizer.step()
            
            for key, value in loss_dict.items():
                # FIX 5a: Log all numeric metrics, not just tensors
                if isinstance(value, (torch.Tensor, float, int, np.number)):
                    total_metrics[key] += value.item() if isinstance(value, torch.Tensor) else value

        return {k: v / len(self.train_loader) for k, v in total_metrics.items()}
    
    def _validate_epoch(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                # The val_loader from utils.py yields 3 items
                uah, physio, labels = batch

                uah, physio, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), labels.to(configs.DEVICE)
                val_mask = torch.tensor([[True, True]] * uah.size(0), device=configs.DEVICE)
                
                main_logits, _, _, _ = self.model(uah, physio, val_mask)
                _, predicted = main_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return {'accuracy': correct / total if total > 0 else 0}