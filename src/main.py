import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import Config
from dataset import HybridDataset, CharTokenizer
from model import HybridBERT

def train():
    Config.set_seed()
    print(f"Using device: {Config.DEVICE}")
    
    # 1. Load Standardized Data
    print("Loading standardized data...")
    try:
        train_df = pd.read_csv(Config.PROCESSED_TRAIN_PATH)
        valid_df = pd.read_csv(Config.PROCESSED_TEST_PATH)
    except FileNotFoundError:
        print("Data not found! Please run 'python preprocess.py' first.")
        return

    # 2. Setup Tokenizers
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Add special tokens for Context
    bert_tok.add_special_tokens({'additional_special_tokens': ['[BEFORE]', '[CURRENT]', '[AFTER]']})
    char_tok = CharTokenizer()

    # 3. Create Datasets
    # Augment=True for Training to catch "f ck" and "ez"
    train_ds = HybridDataset(train_df, bert_tok, char_tok, augment=True)
    valid_ds = HybridDataset(valid_df, bert_tok, char_tok, augment=False)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    model = HybridBERT(char_tok.get_vocab_size()).to(Config.DEVICE)
    model.bert.resize_token_embeddings(len(bert_tok)) # Crucial for special tokens

    # 5. Optimizer (Differential Learning Rates)
    bert_params = list(map(id, model.bert.parameters()))
    cnn_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': Config.LR_BERT},
        {'params': cnn_params, 'lr': Config.LR_CNN}
    ])

    # Weighted Loss (Approx 1:4 ratio)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(Config.DEVICE))

    # 6. Training Loop
    best_f1 = 0.0
    print("\n--- Starting Training ---")
    
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        model.train()
        
        for batch in tqdm(train_loader, desc="Train"):
            b_ids = batch['b_ids'].to(Config.DEVICE)
            b_mask = batch['b_mask'].to(Config.DEVICE)
            c_ids = batch['c_ids'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask, c_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Valid"):
                b_ids = batch['b_ids'].to(Config.DEVICE)
                b_mask = batch['b_mask'].to(Config.DEVICE)
                c_ids = batch['c_ids'].to(Config.DEVICE)
                
                logits = model(b_ids, b_mask, c_ids)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                val_preds.extend(preds)
                val_labels.extend(batch['labels'].numpy())
        
        f1 = f1_score(val_labels, val_preds, pos_label=1)
        print(f"Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Saved Best Model ({f1:.4f})")

if __name__ == "__main__":
    train()