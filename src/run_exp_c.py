# run_exp_c.py
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

def run_experiment_c():
    print("\n=== EXPERIMENT C: Hybrid + Augmentation + CLEAN DATA ===")
    Config.set_seed()
    
    # 1. Load CLEAN Data
    try:
        train_df = pd.read_csv("train_context_cleaned.csv") # <--- Cleaned file
        valid_df = pd.read_csv(Config.PROCESSED_TEST_PATH)
    except FileNotFoundError:
        print("Cleaned data not found. Run 'clean_data.py' first.")
        return

    # 2. Setup (Same as Exp B)
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_tok.add_special_tokens({'additional_special_tokens': ['[BEFORE]', '[CURRENT]', '[AFTER]']})
    char_tok = CharTokenizer()

    train_ds = HybridDataset(train_df, bert_tok, char_tok, augment=True) # Keep Augmentation!
    valid_ds = HybridDataset(valid_df, bert_tok, char_tok, augment=False)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = HybridBERT(char_tok.get_vocab_size()).to(Config.DEVICE)
    model.bert.resize_token_embeddings(len(bert_tok))
    
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': Config.LR_BERT},
        {'params': filter(lambda p: id(p) not in list(map(id, model.bert.parameters())), model.parameters()), 
         'lr': Config.LR_CNN}
    ])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(Config.DEVICE))

    # 3. Train
    best_f1 = 0.0
    for epoch in range(Config.EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            b_ids, b_mask, c_ids, labels = (
                batch['b_ids'].to(Config.DEVICE), batch['b_mask'].to(Config.DEVICE),
                batch['c_ids'].to(Config.DEVICE), batch['labels'].to(Config.DEVICE)
            )
            optimizer.zero_grad()
            loss = criterion(model(b_ids, b_mask, c_ids), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in valid_loader:
                b_ids, b_mask, c_ids = (
                    batch['b_ids'].to(Config.DEVICE), batch['b_mask'].to(Config.DEVICE),
                    batch['c_ids'].to(Config.DEVICE)
                )
                logits = model(b_ids, b_mask, c_ids)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                targets.extend(batch['labels'].numpy())
        
        f1 = f1_score(targets, preds, pos_label=1)
        print(f"Epoch {epoch+1} F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "exp_c_hybrid_clean.pth")

    print(f"Exp C Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    run_experiment_c()