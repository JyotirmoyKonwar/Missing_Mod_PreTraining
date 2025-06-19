import torch
import torch.optim as optim
import torch.nn as nn
from model import PhysionetTransformer
from utils import get_data_loaders
import config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def main():
    """
    Main function to train and validate the PhysionetTransformer model.
    Now includes logging for F1 score, precision, and recall.
    """
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoaders
    train_loader, val_loader = get_data_loaders()

    # Model, Optimizer, and Loss Function
    model = PhysionetTransformer(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training on {device}...")
    # Update the log file header for the new metrics
    with open(config.LOG_FILE, "w") as f:
        f.write("Epoch,Train Loss,Val Accuracy,F1 Score,Precision,Recall\n")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for physio_seq, labels in train_loader:
            physio_seq, labels = physio_seq.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(physio_seq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for physio_seq, labels in val_loader:
                physio_seq, labels = physio_seq.to(device), labels.to(device)
                logits, _ = model(physio_seq)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics using sklearn
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_precision = precision_score(all_labels, all_preds, average='weighted')
        val_recall = recall_score(all_labels, all_preds, average='weighted')

        # Update console and file logging
        log_message = (
            f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | F1: {val_f1:.4f} | "
            f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f}"
        )
        print(log_message)
        with open(config.LOG_FILE, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{val_accuracy:.4f},{val_f1:.4f},{val_precision:.4f},{val_recall:.4f}\n")

    print("Training complete.")

if __name__ == "__main__":
    main()