# --- evaluate.py ---

import torch
from tqdm import tqdm
import configs

def evaluate_missing_modality_performance(model, test_loader):
    """Comprehensive evaluation under different missing modality scenarios."""
    model.eval()
    scenarios = {
        'both_modalities': torch.tensor([True, True]),
        'uah_only': torch.tensor([True, False]), 
        'physio_only': torch.tensor([False, True])
    }
    
    results = {}
    print("\n--- Evaluating Final Model Performance ---")
    
    for scenario_name, mask in scenarios.items():
        correct, total = 0, 0
        
        with torch.no_grad():
            for uah_seq, physio_seq, _, labels in tqdm(test_loader, desc=f"Evaluating {scenario_name}"):
                uah_seq, physio_seq, labels = uah_seq.to(configs.DEVICE), physio_seq.to(configs.DEVICE), labels.to(configs.DEVICE)
                
                # Expand mask to batch size
                batch_mask = mask.unsqueeze(0).expand(uah_seq.size(0), -1)
                
                logits, _, _ = model(uah_seq, physio_seq, batch_mask)
                _, predicted = logits.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = correct / total
        results[scenario_name] = accuracy
        print(f"Accuracy for {scenario_name}: {accuracy:.4f}")
    
    return results