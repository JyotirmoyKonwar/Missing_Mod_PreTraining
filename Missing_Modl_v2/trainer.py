# --- trainer.py ---

import torch
import torch.optim as optim
from tqdm import tqdm
import configs
from loss import MultiComponentLoss

class MissingModalityTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(configs.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.global_epoch = 0 
        self.loss_fn = MultiComponentLoss()
        
        # Optimizer for the main fusion model (all non-encoder parts)
        fusion_params = [p for n, p in self.model.named_parameters() if 'encoder' not in n and p.requires_grad]
        self.fusion_optimizer = optim.Adam(fusion_params, lr=configs.INITIAL_LR)
        
        # CORRECTED: Initialize encoder_optimizer to None. It will be created when needed.
        self.encoder_optimizer = None

    def _update_encoder_optimizer(self):
        """Creates or updates the encoder optimizer with newly unfrozen parameters."""
        encoder_params = [p for n, p in self.model.named_parameters() if 'encoder' in n and p.requires_grad]
        
        if not encoder_params:
            # No unfrozen encoder parameters yet
            return

        # CORRECTED: Create the optimizer if it doesn't exist, or update it if it does.
        if self.encoder_optimizer is None:
            print("Creating encoder optimizer for the first time with unfrozen parameters.")
            self.encoder_optimizer = optim.Adam(encoder_params, lr=configs.INITIAL_LR * configs.ENCODER_LR_FACTOR)
        else:
            # If optimizer already exists, clear its old parameter groups and add the new ones.
            print("Updating encoder optimizer with newly unfrozen parameters.")
            self.encoder_optimizer.param_groups.clear()
            self.encoder_optimizer.add_param_group({'params': encoder_params})


    def run_training_stage(self, stage_num, epochs, alpha, beta, model_save_path):
        print(f"\n--- Starting Stage {stage_num}: {epochs} epochs ---")
        
        self.loss_fn.alpha, self.loss_fn.beta = alpha, beta
        # Enable stratified sampling for Stage 2 and 3
        is_stratified = stage_num > 1
        # The collate_fn handles this, so we don't need to set a flag on the dataset anymore.

        best_val_acc, min_loss_for_best_acc = 0.0, float('inf')

        for epoch in range(1, epochs + 1):
            
            for epoch in range(1, epochs + 1):
                # count epochs globally
                self.global_epoch += 1

                # use the cumulative epoch for progressive unfreezing
                if self.model.progressive_unfreeze(self.global_epoch):
                    self._update_encoder_optimizer()
            
            train_metrics = self._train_epoch()
            val_metrics = self._validate_epoch()

            print(f"[Stage {stage_num}] Ep {epoch}/{epochs} | Train Loss: {train_metrics['total']:.4f} | Val Acc: {val_metrics['accuracy']:.3f}")

            current_accuracy = val_metrics['accuracy']
            current_train_loss = train_metrics['total']
            save_this_epoch = False

            if current_accuracy > best_val_acc:
                best_val_acc = current_accuracy
                min_loss_for_best_acc = current_train_loss
                save_this_epoch = True
                print(f"-> New best accuracy in stage! Saving model to {model_save_path}")
            elif abs(current_accuracy - best_val_acc) < 1e-5 and current_train_loss < min_loss_for_best_acc:
                min_loss_for_best_acc = current_train_loss
                save_this_epoch = True
                print(f"-> Same accuracy, but lower training loss. Saving model to {model_save_path}")

            if save_this_epoch:
                torch.save(self.model.state_dict(), model_save_path)

    def _train_epoch(self):
        self.model.train()
        total_metrics = {k: 0.0 for k in ['total', 'classification', 'auxiliary', 'contrastive', 'consistency', 'reconstruction', 'regularization']}
        
        for uah, physio, mask, labels in tqdm(self.train_loader, desc=f"Training"):
            uah, physio, mask, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), mask.to(configs.DEVICE), labels.to(configs.DEVICE)
            
            self.fusion_optimizer.zero_grad()
            # CORRECTED: Only step the encoder optimizer if it has been created
            if self.encoder_optimizer:
                self.encoder_optimizer.zero_grad()
            
            main_logits, aux_logits, shared_emb, modality_embs = self.model(uah, physio, mask)
            loss_dict = self.loss_fn(self.model, main_logits, aux_logits, labels, shared_emb, modality_embs, mask)
            
            loss_dict['total'].backward()
            # It's good practice to clip gradients of the entire model
            torch.nn.utils.clip_grad_norm_((p for p in self.model.parameters() if p.requires_grad), max_norm=1.0)
            
            self.fusion_optimizer.step()
            if self.encoder_optimizer:
                self.encoder_optimizer.step()
            
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    total_metrics[key] += value.item()
        
        return {k: v / len(self.train_loader) for k, v in total_metrics.items()}
    
    def _validate_epoch(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                # val_loader might not use collate_fn, so handle both cases
                if len(batch) == 4:
                    uah, physio, _, labels = batch
                else:
                    uah, physio, labels = batch

                uah, physio, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), labels.to(configs.DEVICE)
                val_mask = torch.tensor([[True, True]] * uah.size(0), device=configs.DEVICE)
                
                main_logits, _, _, _ = self.model(uah, physio, val_mask)
                _, predicted = main_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return {'accuracy': correct / total if total > 0 else 0}