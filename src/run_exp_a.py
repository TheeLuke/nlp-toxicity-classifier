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

def run_experiment_a():
    print("\n=== EXPERIMENT A: Hybrid Model (Standard Data, No Augmentation) ===")
    Config.set_seed()
    
    # 1. Load Data
    train_df = pd.read_csv(Config.PROCESSED_TRAIN_PATH)
    valid_df = pd.read_csv(Config.PROCESSED_TEST_PATH)

    # 2. Setup
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_tok.add_special_tokens({'additional_special_tokens': ['[BEFORE]', '[CURRENT]', '[AFTER]']})
    char_tok = CharTokenizer()

    # 3. Datasets (AUGMENTATION = FALSE)
    # This is the critical difference for Exp A
    train_ds = HybridDataset(train_df, bert_tok, char_tok, augment=False)
    valid_ds = HybridDataset(valid_df, bert_tok, char_tok, augment=False)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 4. Model & Optimizer
    model = HybridBERT(char_tok.get_vocab_size()).to(Config.DEVICE)
    model.bert.resize_token_embeddings(len(bert_tok))
    
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': Config.LR_BERT},
        {'params': filter(lambda p: id(p) not in list(map(id, model.bert.parameters())), model.parameters()), 
         'lr': Config.LR_CNN}
    ])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(Config.DEVICE))

    # 5. Train
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

        # Validation
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
            torch.save(model.state_dict(), "exp_a_hybrid_no_aug.pth")

    print(f"Exp A Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    run_experiment_a()