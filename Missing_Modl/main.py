# --- main.py ---

import torch
import configs
from models.fusion import SharedLatentSpaceFusion
from utils import get_dataloaders
from trainer import MissingModalityTrainer
from evaluate import evaluate_missing_modality_performance

def main():
    """
    Main execution script that orchestrates the 3-stage training pipeline
    and evaluates all saved models at the end.
    """
    
    # 1. Get DataLoaders
    train_loader, val_loader = get_dataloaders()
    # Using validation set for final testing for simplicity. 
    # For publication, a separate, unseen test set is recommended.
    test_loader = val_loader  

    # 2. Initialize Model and Trainer
    model = SharedLatentSpaceFusion().to(configs.DEVICE)
    trainer = MissingModalityTrainer(model, train_loader, val_loader)

    # --- Stage 1: Warm-up Training ---
    trainer.run_training_stage(
        stage_num=1, epochs=configs.NUM_EPOCHS_STAGE1,
        alpha=configs.STAGE1_ALPHA, beta=configs.STAGE1_BETA,
        missing_rate=configs.STAGE1_MISSING_RATE,
        model_save_path=configs.BEST_STAGE1_MODEL_PATH
    )

    # --- Stage 2: Joint Training with Modality Dropout ---
    print("\nLoading best model from Stage 1 to begin Stage 2...")
    model.load_state_dict(torch.load(configs.BEST_STAGE1_MODEL_PATH, map_location=configs.DEVICE))
    trainer.run_training_stage(
        stage_num=2, epochs=configs.NUM_EPOCHS_STAGE2,
        alpha=configs.STAGE2_ALPHA, beta=configs.STAGE2_BETA,
        missing_rate=configs.STAGE2_MISSING_RATE,
        model_save_path=configs.BEST_STAGE2_MODEL_PATH
    )

    # --- Stage 3: Contrastive Fine-tuning ---
    print("\nLoading best model from Stage 2 to begin Stage 3...")
    model.load_state_dict(torch.load(configs.BEST_STAGE2_MODEL_PATH, map_location=configs.DEVICE))
    trainer.adjust_lr_for_finetuning() # Lower the learning rate for fine-tuning
    trainer.run_training_stage(
        stage_num=3, epochs=configs.NUM_EPOCHS_STAGE3,
        alpha=configs.STAGE3_ALPHA, beta=configs.STAGE3_BETA,
        missing_rate=configs.STAGE3_MISSING_RATE,
        model_save_path=configs.BEST_STAGE3_MODEL_PATH
    )

    # 4. Save the final model state from the very last epoch
    print(f"\nSaving final model state to {configs.LAST_EPOCH_MODEL_PATH}")
    torch.save(model.state_dict(), configs.LAST_EPOCH_MODEL_PATH)

    # --- ADDED: Comprehensive Final Evaluation of All Saved Models ---
    print("\n" + "="*50)
    print("      COMPREHENSIVE FINAL MODEL EVALUATION")
    print("="*50)

    models_to_evaluate = {
        "Best Stage 1 Model": configs.BEST_STAGE1_MODEL_PATH,
        "Best Stage 2 Model": configs.BEST_STAGE2_MODEL_PATH,
        "Best Stage 3 Model": configs.BEST_STAGE3_MODEL_PATH,
        "Last Epoch Model": configs.LAST_EPOCH_MODEL_PATH
    }

    for model_name, model_path in models_to_evaluate.items():
        print(f"\n--- Evaluating: {model_name} ---")
        try:
            # Re-initialize a clean model instance for evaluation
            evaluation_model = SharedLatentSpaceFusion().to(configs.DEVICE)
            evaluation_model.load_state_dict(torch.load(model_path, map_location=configs.DEVICE))
            
            # Run evaluation across different missing modality scenarios
            evaluate_missing_modality_performance(evaluation_model, test_loader)
        except FileNotFoundError:
            print(f"Could not find model file at {model_path}. Skipping evaluation for this model.")
    
    print("\nProcess finished!")

if __name__ == "__main__":
    main()