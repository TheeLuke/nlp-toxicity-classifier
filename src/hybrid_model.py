import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import os
import time
import numpy as np

# --- IMPORT YOUR DATASET ---
# Ensure 'data_loader.py' is in the same directory and contains the updated class
from data_loader import CONDADataset

# --- 1. Define the Hybrid Architecture ---
class HybridBERTCharCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(HybridBERTCharCNN, self).__init__()
        
        # --- Branch A: BERT (Context) ---
        # We load the pre-trained BERT base model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # --- Branch B: CharCNN (Form/Slang) ---
        # Standard CharCNN config (Zhang et al. 2015 simplified for smaller datasets)
        self.char_vocab_size = 71 # 70 chars + 1 padding
        self.char_emb_dim = 128
        self.char_embedding = nn.Embedding(self.char_vocab_size, self.char_emb_dim, padding_idx=0)
        
        self.char_cnn = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(256, 256, kernel_size=7), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(256, 256, kernel_size=3), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3), nn.ReLU(), nn.MaxPool1d(3)
        )
        
        # Calculate CharCNN output size for flattening:
        # Input 1014 -> Pool(3) -> ~338 -> Pool(3) -> ~112 -> Pool(3) -> ~34
        # Output shape: (Batch, 256, 34)
        self.cnn_flat_dim = 256 * 34 
        
        # Projection layer to reduce CNN dimensionality to match BERT's size (768)
        # This creates a balanced concatenation
        self.cnn_projection = nn.Linear(self.cnn_flat_dim, 768)
        
        # --- Fusion & Classification ---
        # BERT (768) + CNN (768) = 1536 input features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, char_input):
        # 1. BERT Forward Pass
        # Pooler output is the [CLS] token embedding (Batch, 768)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_vec = bert_output.pooler_output
        
        # 2. CharCNN Forward Pass
        x_char = self.char_embedding(char_input) # (Batch, Seq, Emb)
        x_char = x_char.permute(0, 2, 1)         # (Batch, Emb, Seq) for Conv1d
        x_char = self.char_cnn(x_char)
        x_char = x_char.view(x_char.size(0), -1) # Flatten
        char_vec = self.cnn_projection(x_char)   # Reduce to 768
        
        # 3. Concatenation (Fusion)
        # This joins the semantic context (BERT) with the stylistic form (CNN)
        combined_vec = torch.cat((bert_vec, char_vec), dim=1) # (Batch, 1536)
        
        # 4. Classification
        logits = self.classifier(combined_vec)
        
        return logits

# --- 2. Training Function ---
def train_hybrid_model(train_path, val_path, save_dir='saved_models', epochs=4, batch_size=16):
    
    # Setup Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"--- Using Device: {device} ---")
    os.makedirs(save_dir, exist_ok=True)
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # --- Load Data ---
    print("Loading Hybrid Data...")
    # use_context=True gives BERT the conversation history
    # model_type='hybrid' ensures the Dataset returns both 'input_ids' AND 'char_input'
    train_ds = CONDADataset(train_path, model_type='hybrid', tokenizer=tokenizer, use_context=True)
    val_ds = CONDADataset(val_path, model_type='hybrid', tokenizer=tokenizer, use_context=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize Model
    print("Initializing HybridBERTCharCNN...")
    model = HybridBERTCharCNN()
    model.to(device)
    
    # Optimizer: Differential Learning Rates
    # BERT layers usually require a lower learning rate (e.g., 2e-5) to preserve pre-trained knowledge.
    # The CNN and Classifier layers are initialized from scratch, so they need a higher rate (e.g., 1e-4) to learn effectively.
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.char_cnn.parameters(), 'lr': 1e-4},
        {'params': model.cnn_projection.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    
    criterion = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs)
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # Move all inputs to device
            b_ids = batch['input_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_char = batch['char_input'].to(device) # The Dataset provides this key for 'hybrid' mode
            b_labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with ALL inputs
            logits = model(input_ids=b_ids, attention_mask=b_mask, char_input=b_char)
            
            loss = criterion(logits, b_labels)
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients to prevent explosion
            optimizer.step()
            scheduler.step()
            
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
                b_ids = batch['input_ids'].to(device)
                b_mask = batch['attention_mask'].to(device)
                b_char = batch['char_input'].to(device)
                b_labels = batch['labels'].to(device)
                
                logits = model(input_ids=b_ids, attention_mask=b_mask, char_input=b_char)
                loss = criterion(logits, b_labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == b_labels).sum().item()
                total += b_labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # --- Save Best Model (Checkpointing) ---
        # Only save if validation loss improves to prevent overfitting
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, 'hybrid_model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Improved! Model Saved to {save_path}")

    print("Hybrid Training Complete.")

if __name__ == "__main__":
    # Update these paths to match your specific file locations
    TRAIN_FILE = 'conda_train.csv' 
    VAL_FILE = 'conda_valid.csv'
    
    train_hybrid_model(TRAIN_FILE, VAL_FILE)