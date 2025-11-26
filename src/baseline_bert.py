# baseline_bert.py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from config import Config

class SimpleBertDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use the CONTEXT column (History + Current)
        text = str(row['context_text'])
        label = int(row['label'])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def run_bert_context():
    print("\n--- Baseline 2: Context-Aware BERT (No CharCNN) ---")
    Config.set_seed()
    
    # 1. Load Data
    try:
        train_df = pd.read_csv(Config.PROCESSED_TRAIN_PATH)
        test_df = pd.read_csv(Config.PROCESSED_TEST_PATH)
    except FileNotFoundError:
        print("Error: Processed CSVs not found.")
        return

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # IMPORTANT: Add special tokens so BERT understands [BEFORE]/[AFTER]
    special_tokens = {'additional_special_tokens': ['[BEFORE]', '[CURRENT]', '[AFTER]']}
    tokenizer.add_special_tokens(special_tokens)

    # 3. Datasets
    # No augmentation here - we want a clean baseline comparison
    train_ds = SimpleBertDataset(train_df, tokenizer, Config.MAX_LEN_BERT)
    test_ds = SimpleBertDataset(test_df, tokenizer, Config.MAX_LEN_BERT)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 4. Model
    print("Initializing BERT...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.resize_token_embeddings(len(tokenizer)) # Resize for special tokens
    model.to(Config.DEVICE)

    # 5. Training Setup
    # Use standard BERT LR
    optimizer = AdamW(model.parameters(), lr=Config.LR_BERT)
    # Weighted Loss (Same as Hybrid for fair comparison)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(Config.DEVICE))

    best_f1 = 0.0
    
    # 6. Training Loop
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        model.train()
        
        for batch in tqdm(train_loader, desc="Train"):
            ids = batch['input_ids'].to(Config.DEVICE)
            mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            # HF models calculate loss internally if labels are provided, 
            # but we want to use our Custom Weighted Loss
            outputs = model(ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Eval"):
                ids = batch['input_ids'].to(Config.DEVICE)
                mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                outputs = model(ids, attention_mask=mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, pos_label=1)
        prec = precision_score(all_labels, all_preds, pos_label=1)
        rec = recall_score(all_labels, all_preds, pos_label=1)
        
        print(f"Val F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "bert_context_baseline.pth")
            
    print(f"\nFinal Baseline BERT F1: {best_f1:.4f}")

if __name__ == "__main__":
    run_bert_context()