# --- main.py ---

import torch
import configs
from models.fusion import SharedLatentSpaceFusion
from utils import get_dataloaders
from trainer import MissingModalityTrainer
from evaluate import evaluate_missing_modality_performance

def main():
    """Main execution script for missing modality training and evaluation."""
    
    # 1. Get DataLoaders
    # The get_dataloaders function in utils currently uses dummy data.
    # TODO: Modify utils.py to load your actual, aligned UAH and Physionet data.
    train_loader, val_loader = get_dataloaders()
    # Assuming the validation loader can be reused for final testing
    test_loader = val_loader 
    
    # 2. Initialize Model
    model = SharedLatentSpaceFusion()
    
    # 3. Initialize Trainer and run the 3-stage pipeline
    trainer = MissingModalityTrainer(model, train_loader, val_loader)
    trainer.train()
    
    # 4. Evaluate the best model
    print("\n--- Loading best model for final evaluation ---")
    # Re-initialize model to ensure clean state
    final_model = SharedLatentSpaceFusion().to(configs.DEVICE)
    final_model.load_state_dict(torch.load(configs.BEST_FUSION_MODEL_PATH))
    
    evaluate_missing_modality_performance(final_model, test_loader)
    
    print("\nProcess finished!")

if __name__ == "__main__":
    main()