# --- main.py ---

import torch
import configs
from models.fusion import SharedLatentSpaceFusion
from utils import get_dataloaders
from trainer import MissingModalityTrainer
from evaluate import evaluate_missing_modality_performance

def main():
    """
    Main script to orchestrate the 3-stage training pipeline and evaluate all
    saved model checkpoints on the final held-out test set.
    """
    
    # --- UPDATED: Unpack all three dataloaders ---
    train_loader, val_loader, test_loader = get_dataloaders()

    model = SharedLatentSpaceFusion().to(configs.DEVICE)
    trainer = MissingModalityTrainer(model, train_loader, val_loader)

    # --- Stage 1 ---
    trainer.run_training_stage(
        stage_num=1, epochs=configs.NUM_EPOCHS_STAGE1,
        alpha=configs.STAGE1_ALPHA, beta=configs.STAGE1_BETA,
        model_save_path=configs.BEST_STAGE1_MODEL_PATH
    )

    # --- Stage 2 ---
    print("\nLoading best model from Stage 1 to begin Stage 2...")
    model.load_state_dict(torch.load(configs.BEST_STAGE1_MODEL_PATH, map_location=configs.DEVICE))
    trainer.run_training_stage(
        stage_num=2, epochs=configs.NUM_EPOCHS_STAGE2,
        alpha=configs.STAGE2_ALPHA, beta=configs.STAGE2_BETA,
        model_save_path=configs.BEST_STAGE2_MODEL_PATH
    )

    # --- Stage 3 ---
    print("\nLoading best model from Stage 2 to begin Stage 3...")
    model.load_state_dict(torch.load(configs.BEST_STAGE2_MODEL_PATH, map_location=configs.DEVICE))
    trainer.run_training_stage(
        stage_num=3, epochs=configs.NUM_EPOCHS_STAGE3,
        alpha=configs.STAGE3_ALPHA, beta=configs.STAGE3_BETA,
        model_save_path=configs.BEST_STAGE3_MODEL_PATH
    )
    
    print(f"\nSaving final model state to {configs.LAST_EPOCH_MODEL_PATH}")
    torch.save(model.state_dict(), configs.LAST_EPOCH_MODEL_PATH)

    # --- Final Evaluation on the TEST SET ---
    print("\n" + "="*50)
    print("      COMPREHENSIVE FINAL MODEL EVALUATION ON TEST SET")
    print("="*50)

    models_to_evaluate = {
        "Best Stage 1 Model": configs.BEST_STAGE1_MODEL_PATH,
        "Best Stage 2 Model": configs.BEST_STAGE2_MODEL_PATH,
        "Best Stage 3 Model": configs.BEST_STAGE3_MODEL_PATH,
        "Last Epoch Model": configs.LAST_EPOCH_MODEL_PATH
    }

    for model_name, model_path in models_to_evaluate.items():
        print(f"\n--- Evaluating: {model_name} ({model_path}) ---")
        try:
            evaluation_model = SharedLatentSpaceFusion().to(configs.DEVICE)
            evaluation_model.load_state_dict(torch.load(model_path, map_location=configs.DEVICE))
            # --- UPDATED: Pass the test_loader to the evaluation function ---
            evaluate_missing_modality_performance(evaluation_model, test_loader)
        except FileNotFoundError:
            print(f"Could not find model file '{model_path}'. Skipping evaluation.")
    
    print("\nProcess finished!")

if __name__ == "__main__":
    main()