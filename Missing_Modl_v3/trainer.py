# --- trainer.py ---

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import configs
from loss import MultiComponentLoss
from gradient_modulation import ClassifierGuidedGradientModulation, apply_gradient_modulation
from modality_utilization import ModalityUtilizationRegulator, apply_utilization_regulation

class MissingModalityTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(configs.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = MultiComponentLoss()
        self._batch_count = 0

        # -- SOLUTION 2: Optimizer includes adaptive loss parameters --
        fusion_params = [p for n, p in self.model.named_parameters() if 'encoder' not in n and p.requires_grad]
        adaptive_params = list(self.loss_fn.adaptive_loss.parameters())
        self.fusion_optimizer = optim.Adam([
            {'params': fusion_params}, {'params': adaptive_params, 'lr': configs.INITIAL_LR * 2, 'weight_decay': 0}
        ], lr=configs.INITIAL_LR)

        # -- SOLUTION 1: Placeholders for modality-specific optimizers --
        self.uah_optimizer, self.physio_optimizer, self.encoder_optimizer = None, None, None
        
        # -- SOLUTION 3: CGGM module and its optimizer --
        self.cggm = ClassifierGuidedGradientModulation(configs.SHARED_DIM, configs.NUM_CLASSES).to(configs.DEVICE)
        self.cggm_optimizer = optim.Adam(self.cggm.parameters(), lr=configs.INITIAL_LR * 2)

        # -- SOLUTION 4: Utilization regulator --
        self.utilization_regulator = ModalityUtilizationRegulator(prime_window_epochs=15, regulation_strength=0.15)

    def _update_encoder_optimizer(self):
        """SOLUTION 1: Creates or updates modality-specific/unified encoder optimizers."""
        if not configs.ENABLE_MODALITY_SPECIFIC_LR:
            params = [p for n,p in self.model.named_parameters() if 'encoder' in n and p.requires_grad]
            if params: self.encoder_optimizer = optim.Adam(params, lr=configs.INITIAL_LR * 0.1)
            return

        uah_params = [p for p in self.model.uah_encoder.parameters() if p.requires_grad]
        physio_params = [p for p in self.model.physio_encoder.parameters() if p.requires_grad]
        if uah_params: self.uah_optimizer = optim.Adam(uah_params, lr=configs.INITIAL_LR * configs.UAH_LR_FACTOR)
        if physio_params: self.physio_optimizer = optim.Adam(physio_params, lr=configs.INITIAL_LR * configs.PHYSIO_LR_FACTOR)

    def run_training_stage(self, stage_num, epochs, alpha, beta, model_save_path):
        print(f"\n--- Starting Stage {stage_num}: {epochs} epochs ---")
        self.loss_fn.alpha, self.loss_fn.beta = alpha, beta
        best_val_acc, min_loss_for_best_acc = 0.0, float('inf')

        for epoch in range(1, epochs + 1):
            if self.model.progressive_unfreeze(epoch): self._update_encoder_optimizer()
            
            # -- SOLUTION 4: Log frozen status --
            frozen_status = self.model.get_modality_frozen_status()
            print(f"Frozen Ratios -> UAH: {frozen_status['uah']}, Physio: {frozen_status['physio']}")
            
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch()
            print(f"[Stage {stage_num}] Ep {epoch}/{epochs} | Train Loss: {train_metrics['total']:.4f} | Val Acc: {val_metrics['accuracy']:.3f}")

            if val_metrics['accuracy'] > best_val_acc or (abs(val_metrics['accuracy'] - best_val_acc)<1e-5 and train_metrics['total'] < min_loss_for_best_acc):
                best_val_acc, min_loss_for_best_acc = val_metrics['accuracy'], train_metrics['total']
                print(f"-> New best performance. Saving model to {model_save_path}")
                torch.save(self.model.state_dict(), model_save_path)

    def _train_epoch(self, epoch):
        self.model.train(); self.cggm.train()
        epoch_metrics = {}

        for uah, physio, mask, labels in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            uah, physio, mask, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), mask.to(configs.DEVICE), labels.to(configs.DEVICE)
            
            # 1. Zero all gradients
            self.fusion_optimizer.zero_grad(); self.cggm_optimizer.zero_grad()
            for opt in [self.encoder_optimizer, self.uah_optimizer, self.physio_optimizer]:
                if opt: opt.zero_grad()

            # 2. Main forward pass
            main_logits, aux_logits, shared_emb, mod_embs = self.model(uah, physio, mask)

            # 3. Calculate all loss components (Solutions 2, 3, 4)
            loss_dict = self.loss_fn(self.model, main_logits, aux_logits, labels, shared_emb, mod_embs, mask)
            class_loss, grad_mod_loss, coeffs = self.cggm(mod_embs, labels, self.model, F.cross_entropy)
            reg_factors, info_rates = self.utilization_regulator(mod_embs, epoch)
            
            total_loss = loss_dict['total']
            if grad_mod_loss: total_loss += 0.2 * grad_mod_loss
            if class_loss: total_loss += 0.1 * class_loss

            # 4. Backward pass and gradient modulation
            total_loss.backward()
            apply_utilization_regulation(self.model, reg_factors)
            if coeffs: apply_gradient_modulation(self.model, coeffs)
            torch.nn.utils.clip_grad_norm_((p for p in self.model.parameters() if p.requires_grad), max_norm=1.0)
            
            # 5. Step all optimizers
            self.fusion_optimizer.step(); self.cggm_optimizer.step()
            for opt in [self.encoder_optimizer, self.uah_optimizer, self.physio_optimizer]:
                if opt: opt.step()

            for key, value in loss_dict.items():
                val = value.item() if isinstance(value, torch.Tensor) else np.mean(value)
                epoch_metrics[key] = epoch_metrics.get(key, 0) + val
        return {k: v / len(self.train_loader) for k, v in epoch_metrics.items()}

    def _validate_epoch(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for uah, physio, labels in self.val_loader:
                uah, physio, labels = uah.to(configs.DEVICE), physio.to(configs.DEVICE), labels.to(configs.DEVICE)
                mask = torch.tensor([[True, True]] * uah.size(0), device=configs.DEVICE)
                logits, _, _, _ = self.model(uah, physio, mask)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return {'accuracy': correct / total if total > 0 else 0}