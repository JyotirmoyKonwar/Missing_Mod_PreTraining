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
        self.loss_fn = MultiComponentLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=configs.INITIAL_LR)
        # Note: A single scheduler across all stages might not be ideal.
        # We will manually adjust LR for Stage 3 as a more robust approach.

    def run_training_stage(self, stage_num, epochs, alpha, beta, missing_rate, model_save_path):
        """Runs a single stage of the training pipeline."""
        print(f"\n--- Starting Stage {stage_num}: {epochs} epochs ---")
        
        # Set stage-specific loss weights and missing rate
        self.loss_fn.alpha, self.loss_fn.beta = alpha, beta
        self.train_loader.dataset.missing_rate = missing_rate

        best_val_accuracy_stage = 0.0
        min_train_loss_for_best_acc = float('inf')

        for epoch in range(1, epochs + 1):
            # Train one epoch
            train_metrics = self._train_epoch()
            
            # Validate one epoch
            val_metrics = self._validate_epoch()

            print(f"[Stage {stage_num}] Epoch {epoch}/{epochs} | Train Loss: {train_metrics['total']:.4f} | Val Acc: {val_metrics['accuracy']:.3f}")

            # --- Advanced Saving Logic ---
            current_accuracy = val_metrics['accuracy']
            current_train_loss = train_metrics['total']
            save_this_epoch = False

            if current_accuracy > best_val_accuracy_stage:
                best_val_accuracy_stage = current_accuracy
                min_train_loss_for_best_acc = current_train_loss
                save_this_epoch = True
                print(f"-> New best accuracy in stage! Saving model.")
            elif abs(current_accuracy - best_val_accuracy_stage) < 1e-5 and current_train_loss < min_train_loss_for_best_acc:
                min_train_loss_for_best_acc = current_train_loss
                save_this_epoch = True
                print(f"-> Same accuracy, but lower training loss. Saving model.")

            if save_this_epoch:
                torch.save(self.model.state_dict(), model_save_path)

    def _train_epoch(self):
        self.model.train()
        total_metrics = {k: 0.0 for k in ['total', 'classification', 'contrastive', 'consistency', 'regularization']}
        
        for uah, physio, mask, labels in tqdm(self.train_loader, desc=f"Training"):
            uah, physio, mask, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), mask.to(configs.DEVICE), labels.to(configs.DEVICE)
            
            self.optimizer.zero_grad()
            logits, shared_emb, modality_embs = self.model(uah, physio, mask)
            loss_dict = self.loss_fn(logits, labels, shared_emb, modality_embs, mask)
            
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    total_metrics[key] += value.item()
        
        return {k: v / len(self.train_loader) for k, v in total_metrics.items()}
    
    def _validate_epoch(self):
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for uah, physio, mask, labels in tqdm(self.val_loader, desc="Validating"):
                uah, physio, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), labels.to(configs.DEVICE)
                val_mask = torch.tensor([[True, True]] * uah.size(0)).to(configs.DEVICE)
                logits, _, _ = self.model(uah, physio, val_mask)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {'accuracy': correct / total}

    def adjust_lr_for_finetuning(self):
        """Manually adjusts the learning rate for the fine-tuning stage."""
        print("Adjusting learning rate for fine-tuning stage...")
        for g in self.optimizer.param_groups:
            g['lr'] = configs.INITIAL_LR * configs.STAGE3_LR_FACTOR