import torch
import torch.optim as optim
import torch.nn as nn
from model import UAHTransformer
from utils import get_data_loaders
import config

def main():
    """
    Main function to train and validate the UAHTransformer model using settings from config.
    """
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoaders
    train_loader, val_loader = get_data_loaders()

    # Model, Optimizer, and Loss Function
    model = UAHTransformer(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training on {device}...")
    with open(config.LOG_FILE, "w") as f:
        f.write("Epoch,Train Loss,Val Accuracy\n")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for uah_seq, labels in train_loader:
            uah_seq, labels = uah_seq.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(uah_seq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for uah_seq, labels in val_loader:
                uah_seq, labels = uah_seq.to(device), labels.to(device)
                logits, _ = model(uah_seq)
                preds = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        
        log_message = f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%"
        print(log_message)
        with open(config.LOG_FILE, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{val_accuracy:.2f}\n")
            
    print("Training complete.")

if __name__ == "__main__":
    main()