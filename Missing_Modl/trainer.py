# --- trainer.py ---

import torch
from tqdm import tqdm
import configs
from loss import MultiComponentLoss

class MissingModalityTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(configs.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = MultiComponentLoss()
        
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=configs.INITIAL_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.NUM_EPOCHS)
        best_val_acc = 0.0

        for epoch in range(1, configs.NUM_EPOCHS + 1):
            # --- Set stage-specific hyperparameters ---
            if 1 <= epoch <= configs.STAGE1_EPOCHS:
                stage = 1
                self.loss_fn.alpha, self.loss_fn.beta = configs.STAGE1_ALPHA, configs.STAGE1_BETA
                self.train_loader.dataset.missing_rate = configs.STAGE1_MISSING_RATE
            elif configs.STAGE1_EPOCHS < epoch <= configs.STAGE2_EPOCHS:
                stage = 2
                self.loss_fn.alpha, self.loss_fn.beta = configs.STAGE2_ALPHA, configs.STAGE2_BETA
                self.train_loader.dataset.missing_rate = configs.STAGE2_MISSING_RATE
            else: # Stage 3
                stage = 3
                if epoch == configs.STAGE2_EPOCHS + 1: # Adjust LR only once at the start of stage 3
                    print("\n--- Entering Stage 3: Adjusting learning rate for fine-tuning ---")
                    for g in optimizer.param_groups: g['lr'] = configs.INITIAL_LR * configs.STAGE3_LR_FACTOR
                self.loss_fn.alpha, self.loss_fn.beta = configs.STAGE3_ALPHA, configs.STAGE3_BETA
                self.train_loader.dataset.missing_rate = configs.STAGE3_MISSING_RATE

            # --- Train and Validate ---
            train_metrics = self._train_epoch()
            val_metrics = self._validate_epoch()
            scheduler.step()

            print(f"[Stage {stage}] Epoch {epoch}/{configs.NUM_EPOCHS} | Train Loss: {train_metrics['total']:.4f} | Val Acc: {val_metrics['accuracy']:.3f}")

            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(self.model.state_dict(), configs.BEST_FUSION_MODEL_PATH)
                print(f"-> New best model saved with accuracy: {best_val_acc:.3f}")
    
    def _train_epoch(self):
        self.model.train()
        total_metrics = {k: 0.0 for k in ['total', 'classification', 'contrastive', 'consistency', 'regularization']}
        
        for uah, physio, mask, labels in tqdm(self.train_loader, desc="Training"):
            uah, physio, mask, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), mask.to(configs.DEVICE), labels.to(configs.DEVICE)
            
            self.model.optimizer.zero_grad() # Corrected: assuming optimizer is part of the model or trainer
            
            logits, shared_emb, modality_embs = self.model(uah, physio, mask)
            loss_dict = self.loss_fn(logits, labels, shared_emb, modality_embs, mask)
            
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.model.optimizer.step()
            
            for key, value in loss_dict.items():
                total_metrics[key] += value.item()
        
        return {k: v / len(self.train_loader) for k, v in total_metrics.items()}
    
    def _validate_epoch(self):
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for uah, physio, mask, labels in tqdm(self.val_loader, desc="Validating"):
                uah, physio, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), labels.to(configs.DEVICE)
                
                # In validation, we always provide both modalities
                val_mask = torch.tensor([[True, True]] * uah.size(0))
                logits, _, _ = self.model(uah, physio, val_mask)
                _, predicted = logits.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {'accuracy': correct / total}