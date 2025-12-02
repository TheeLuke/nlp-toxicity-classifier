import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

try:
    from dataset_loader import CONDADataset
except ImportError:
    pass

# --- 1. Define the CharCNN Architecture ---
class CharCNN(nn.Module):
    """
    Character-level CNN based on Zhang et al. (2015).
    Designed to capture morphological patterns like misspellings and slang.
    """
    def __init__(self, num_classes=2, embedding_dim=128, max_len=1014, num_chars=71):
        super(CharCNN, self).__init__()
        
        # 1. Embedding Layer
        # num_chars + 1 for padding (0 index)
        self.embedding = nn.Embedding(num_chars, embedding_dim, padding_idx=0)
        
        # 2. Convolutional Layers
        # "Small" configuration from Zhang et al. adapted for the CONDA dataset size
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            
            # Layer 2
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            
            # Layer 3
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            
            # Layer 4
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            
            # Layer 5
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
        
        # 3. Fully Connected Layers
        # Calculate input dimension dynamically or hardcode for max_len=1014
        # Input 1014 -> Pool(3) -> ~338 -> Pool(3) -> ~112 -> Pool(3) -> ~34
        # 34 * 256 channels = 8704 features
        self.fc_input_dim = 256 * 34 
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, max_len)
        
        # Embedding: (batch, max_len, emb_dim)
        x = self.embedding(x)
        
        # Permute for Conv1d: PyTorch expects (batch, channels, length)
        x = x.permute(0, 2, 1)
        
        # Convolutions
        x = self.conv_layers(x)
        
        # Flatten: (batch, features)
        x = x.view(x.size(0), -1)
        
        # Fully Connected
        logits = self.fc_layers(x)
        
        return logits


# --- 2. Training Function ---
def train_charcnn_baseline(train_path, val_path, save_dir='saved_models', epochs=10, batch_size=64):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"--- Using Device: {device} ---")
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Load Data ---
    print("Loading CharCNN Data...")
    
    # IMPORTANT: 
    # model_type='charcnn' -> Returns 'char_input' key in batch
    # use_context=False -> We isolate character features without history, per your report plan
    train_dataset = CONDADataset(train_path, model_type='charcnn', max_len=1014, use_context=False)
    val_dataset = CONDADataset(val_path, model_type='charcnn', max_len=1014, use_context=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # --- Initialize Model ---
    print("Initializing Standalone CharCNN...")
    model = CharCNN(num_classes=2, embedding_dim=128, max_len=1014)
    model.to(device)
    
    # Optimizer: Adam is generally faster to converge than SGD for this architecture
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"--- Starting Training for {epochs} Epochs ---")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # Unpack batch from new data loader
            b_input = batch['char_input'].to(device)
            b_labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(b_input)
            loss = criterion(logits, b_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 100 == 0 and step != 0:
                print(f"  Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                b_input = batch['char_input'].to(device)
                b_labels = batch['labels'].to(device)
                
                logits = model(b_input)
                loss = criterion(logits, b_labels)
                total_val_loss += loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                correct += (preds == b_labels).sum().item()
                total += b_labels.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, 'charcnn_baseline_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Improved! Model Saved to {save_path}")
            
    print("CharCNN Training Complete.")

if __name__ == "__main__":
    # Update these paths to match your actual file locations
    TRAIN_FILE = 'conda_train.csv' 
    VAL_FILE = 'conda_valid.csv'
    
    train_charcnn_baseline(TRAIN_FILE, VAL_FILE)