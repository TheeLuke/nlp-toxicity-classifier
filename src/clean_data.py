# clean_data.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from cleanlab.filter import find_label_issues
from tqdm import tqdm
import gc

from config import Config
from dataset import HybridDataset, CharTokenizer
from model import HybridBERT

def generate_clean_data():
    print("\n=== EXPERIMENT C PART 1: Detecting Label Errors ===")
    df = pd.read_csv(Config.PROCESSED_TRAIN_PATH)
    
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_tok.add_special_tokens({'additional_special_tokens': ['[BEFORE]', '[CURRENT]', '[AFTER]']})
    char_tok = CharTokenizer()

    # Cross-Validation for Out-of-Sample Probs
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=Config.SEED)
    probs = np.zeros((len(df), 2))
    
    for fold, (train_idx, hold_idx) in enumerate(skf.split(df, df['label'])):
        print(f"  - Processing Fold {fold+1}/4...")
        
        # Subsets
        train_sub = torch.utils.data.Subset(HybridDataset(df, bert_tok, char_tok), train_idx)
        hold_sub = torch.utils.data.Subset(HybridDataset(df, bert_tok, char_tok), hold_idx)
        
        train_l = DataLoader(train_sub, batch_size=16, shuffle=True)
        hold_l = DataLoader(hold_sub, batch_size=16, shuffle=False)
        
        # Model
        model = HybridBERT(char_tok.get_vocab_size()).to(Config.DEVICE)
        model.bert.resize_token_embeddings(len(bert_tok))
        optim = AdamW([{'params': model.parameters(), 'lr': 2e-5}]) # Simple LR for cleaning
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(Config.DEVICE))
        
        # Train (2 Epochs is enough for error detection)
        for _ in range(2):
            model.train()
            for b in tqdm(train_l, leave=False):
                b_ids, b_mask, c_ids, lbs = b['b_ids'].to(Config.DEVICE), b['b_mask'].to(Config.DEVICE), b['c_ids'].to(Config.DEVICE), b['labels'].to(Config.DEVICE)
                optim.zero_grad()
                loss = crit(model(b_ids, b_mask, c_ids), lbs)
                loss.backward()
                optim.step()
        
        # Predict
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for b in tqdm(hold_l, leave=False):
                b_ids, b_mask, c_ids = b['b_ids'].to(Config.DEVICE), b['b_mask'].to(Config.DEVICE), b['c_ids'].to(Config.DEVICE)
                fold_preds.extend(torch.softmax(model(b_ids, b_mask, c_ids), dim=1).cpu().numpy())
        
        probs[hold_idx] = np.array(fold_preds)
        del model, optim; torch.cuda.empty_cache(); gc.collect()

    # Cleanlab
    print("  - Running Cleanlab filter...")
    issue_idx = find_label_issues(
        labels=df['label'].values,
        pred_probs=probs,
        return_indices_ranked_by='self_confidence'
    )
    
    print(f"Found {len(issue_idx)} label errors.")
    clean_df = df.drop(issue_idx).reset_index(drop=True)
    
    out_path = "train_context_cleaned.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path}")

if __name__ == "__main__":
    generate_clean_data()