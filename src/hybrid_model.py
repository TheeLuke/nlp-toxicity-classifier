import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import os
import time

try:
    from dataset_loader import CONDADataset
except ImportError:
    pass 

class HybridBERTCharCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(HybridBERTCharCNN, self).__init__()
        
        # --- Branch A: BERT (The Primary Brain) ---
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # --- Branch B: CharCNN (The Slang "Helper") ---
        self.char_vocab_size = 71 
        self.char_emb_dim = 128
        self.char_embedding = nn.Embedding(self.char_vocab_size, self.char_emb_dim, padding_idx=0)
        
        self.char_cnn = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(256, 256, kernel_size=7), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(256, 256, 3), nn.ReLU(),
            nn.Conv1d(256, 256, 3), nn.ReLU(),
            nn.Conv1d(256, 256, 3), nn.ReLU(), nn.MaxPool1d(3)
        )
        
        self.cnn_flat_dim = 256 * 34 
        
        # --- CHANGE 1: The Bottleneck ---
        # Instead of projecting to 768 (equal to BERT), we project to 128.
        # This makes the CharCNN features a "minority report" (approx 15% of the signal).
        self.cnn_projection = nn.Linear(self.cnn_flat_dim, 128)
        
        # --- CHANGE 2: Specific Dropout for CharCNN ---
        # We apply high dropout (0.5) to the CharCNN branch to filter noise.
        self.char_dropout = nn.Dropout(0.5)
        
        # --- Fusion & Classification ---
        # Input features: BERT (768) + CharCNN (128) = 896
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768 + 128, 512), # Adjusted input size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, char_input):
        # 1. BERT Forward Pass
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_vec = bert_output.pooler_output # (Batch, 768)
        
        # 2. CharCNN Forward Pass
        x_char = self.char_embedding(char_input) 
        x_char = x_char.permute(0, 2, 1)         
        x_char = self.char_cnn(x_char)
        x_char = x_char.view(x_char.size(0), -1) 
        
        char_vec = self.cnn_projection(x_char)   # (Batch, 128)
        char_vec = self.char_dropout(char_vec)   # Apply strict dropout to noise
        
        # 3. Concatenation (Fusion)
        # 768 + 128 = 896
        combined_vec = torch.cat((bert_vec, char_vec), dim=1) 
        
        # 4. Classification
        logits = self.classifier(combined_vec)
        
        return logits

# --- 2. Training Function ---
def train_hybrid_model(train_path, val_path, save_dir='saved_models', epochs=4, batch_size=16):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"--- Using Device: {device} ---")
    os.makedirs(save_dir, exist_ok=True)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("Loading Hybrid Data...")
    train_ds = CONDADataset(train_path, model_type='hybrid', tokenizer=tokenizer, use_context=True)
    val_ds = CONDADataset(val_path, model_type='hybrid', tokenizer=tokenizer, use_context=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print("Initializing HybridBERTCharCNN (Option B: Bottlenecked)...")
    model = HybridBERTCharCNN()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # PHASE 1: WARMUP (CharCNN Only)
    # ==========================================
    print("\n" + "="*40)
    print(">>> PHASE 1: WARMUP (Freezing BERT)")
    print("="*40)
    
    for param in model.bert.parameters():
        param.requires_grad = False
        
    optimizer_warmup = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    model.train()
    total_loss = 0
    print("  Starting Warmup Epoch...")
    
    for step, batch in enumerate(train_loader):
        b_ids = batch['input_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        b_char = batch['char_input'].to(device)
        b_labels = batch['labels'].to(device)
        
        optimizer_warmup.zero_grad()
        logits = model(input_ids=b_ids, attention_mask=b_mask, char_input=b_char)
        loss = criterion(logits, b_labels)
        loss.backward()
        optimizer_warmup.step()
        
        total_loss += loss.item()
        if step % 100 == 0 and step != 0:
            print(f"    Warmup Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # ==========================================
    # PHASE 2: JOINT TRAINING (Fine-Tuning)
    # ==========================================
    print("\n" + "="*40)
    print(f">>> PHASE 2: JOINT TRAINING (Unfreezing BERT for {epochs} Epochs)")
    print("="*40)
    
    for param in model.bert.parameters():
        param.requires_grad = True
        
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},           
        {'params': model.char_cnn.parameters(), 'lr': 1e-4},       
        {'params': model.cnn_projection.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            b_ids = batch['input_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_char = batch['char_input'].to(device) 
            b_labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids=b_ids, attention_mask=b_mask, char_input=b_char)
            loss = criterion(logits, b_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
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
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, 'hybrid_model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Improved! Model Saved to {save_path}")

    print("Hybrid Training Complete.")

if __name__ == "__main__":
    TRAIN_FILE = 'conda_train.csv' 
    VAL_FILE = 'conda_valid.csv'
    train_hybrid_model(TRAIN_FILE, VAL_FILE)