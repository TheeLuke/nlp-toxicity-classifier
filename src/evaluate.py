# evaluate.py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import HybridDataset, CharTokenizer
from model import HybridBERT

# --- 1. Setup ---
Config.set_seed()
print(f"\n=== PHASE 4: FINAL EVALUATION ===")

# Load Data
valid_df = pd.read_csv(Config.PROCESSED_TEST_PATH)
# Recover original intents if they exist, else use label
if 'intentClass' not in valid_df.columns:
    # Fallback if pre-processing didn't save it (it should have)
    raw_df = pd.read_csv(Config.RAW_VALID_PATH)
    valid_df['intentClass'] = raw_df['intentClass']

# Setup Tokenizers
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_tok.add_special_tokens({'additional_special_tokens': ['[BEFORE]', '[CURRENT]', '[AFTER]']})
char_tok = CharTokenizer()

# Define the "Stress Test" List (The most important part!)
adversarial_examples = [
    "gg ez mid", 
    "f ck u", 
    "sh!t team", 
    "u r tr@sh",
    "rep0rted",
    "kill urself",
    "kys",
    "dont be an idiot",
    "nice job feeding", # Sarcasm/Implicit
    "we won", # False Negative check
    "?" # False Negative check
]

# --- 2. Helper Functions ---

def get_model_preds(model, model_type, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {model_type}", leave=False):
            if model_type == 'bert':
                ids = batch['input_ids'].to(Config.DEVICE)
                mask = batch['attention_mask'].to(Config.DEVICE)
                outputs = model(ids, attention_mask=mask)
                p = torch.argmax(outputs.logits, dim=1)
            else: # Hybrid
                b_ids = batch['b_ids'].to(Config.DEVICE)
                b_mask = batch['b_mask'].to(Config.DEVICE)
                c_ids = batch['c_ids'].to(Config.DEVICE)
                logits = model(b_ids, b_mask, c_ids)
                p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy())
    return preds

def test_adversarial(model, model_type):
    results = {}
    model.eval()
    for text in adversarial_examples:
        # Preprocess "on the fly"
        c_text = f"[CURRENT] {text}" # Simulate context
        
        if model_type == 'bert':
            enc = bert_tok(c_text, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
            with torch.no_grad():
                out = model(enc['input_ids'].to(Config.DEVICE), attention_mask=enc['attention_mask'].to(Config.DEVICE))
                pred = torch.argmax(out.logits, dim=1).item()
        else:
            b_enc = bert_tok(c_text, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
            c_ids = char_tok.encode(text).unsqueeze(0)
            with torch.no_grad():
                logits = model(b_enc['input_ids'].to(Config.DEVICE), b_enc['attention_mask'].to(Config.DEVICE), c_ids.to(Config.DEVICE))
                pred = torch.argmax(logits, dim=1).item()
        
        results[text] = "TOXIC" if pred == 1 else "Safe"
    return results

# --- 3. Load Models ---
models = {}

# Load Baseline BERT
try:
    print("Loading Baseline BERT...")
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    bert_model.resize_token_embeddings(len(bert_tok))
    bert_model.load_state_dict(torch.load("bert_context_baseline.pth"))
    bert_model.to(Config.DEVICE)
    models['Baseline BERT'] = ('bert', bert_model)
except FileNotFoundError:
    print("Warning: Baseline BERT model not found.")

# Load Hybrid A
try:
    print("Loading Hybrid A (No Aug)...")
    hyb_a = HybridBERT(char_tok.get_vocab_size())
    hyb_a.bert.resize_token_embeddings(len(bert_tok))
    hyb_a.load_state_dict(torch.load("exp_a_hybrid_no_aug.pth"))
    hyb_a.to(Config.DEVICE)
    models['Hybrid A'] = ('hybrid', hyb_a)
except FileNotFoundError:
    print("Warning: Hybrid A model not found.")

# Load Hybrid B
try:
    print("Loading Hybrid B (Augmented)...")
    hyb_b = HybridBERT(char_tok.get_vocab_size())
    hyb_b.bert.resize_token_embeddings(len(bert_tok))
    hyb_b.load_state_dict(torch.load("exp_b_hybrid_aug.pth"))
    hyb_b.to(Config.DEVICE)
    models['Hybrid B'] = ('hybrid', hyb_b)
except FileNotFoundError:
    print("Warning: Hybrid B model not found.")

# Load Hybrid C (Cleaned) - Optional if not done yet
try:
    print("Loading Hybrid C (Cleaned)...")
    hyb_c = HybridBERT(char_tok.get_vocab_size())
    hyb_c.bert.resize_token_embeddings(len(bert_tok))
    hyb_c.load_state_dict(torch.load("exp_c_hybrid_clean.pth"))
    hyb_c.to(Config.DEVICE)
    models['Hybrid C'] = ('hybrid', hyb_c)
except FileNotFoundError:
    print("Hybrid C not found (yet). Skipping.")

# --- 4. Run Evaluation ---
metrics_data = []
adv_data = {'Text': adversarial_examples}

# Create DataLoaders
class SimpleBertDS(torch.utils.data.Dataset): # Mini version for eval
    def __init__(self, df, tok): self.df=df; self.tok=tok
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        enc = self.tok(str(self.df.iloc[i]['context_text']), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten()}

bert_loader = DataLoader(SimpleBertDS(valid_df, bert_tok), batch_size=32, shuffle=False)
hybrid_loader = DataLoader(HybridDataset(valid_df, bert_tok, char_tok, augment=False), batch_size=32, shuffle=False)

for name, (m_type, model) in models.items():
    print(f"\nEvaluating {name}...")
    
    # 1. Standard Metrics
    loader = bert_loader if m_type == 'bert' else hybrid_loader
    preds = get_model_preds(model, m_type, loader)
    
    f1 = f1_score(valid_df['label'], preds, pos_label=1)
    prec = precision_score(valid_df['label'], preds, pos_label=1)
    rec = recall_score(valid_df['label'], preds, pos_label=1)
    
    metrics_data.append({'Model': name, 'Precision': prec, 'Recall': rec, 'F1': f1})
    
    # 2. Stress Test
    adv_results = test_adversarial(model, m_type)
    adv_data[name] = [adv_results[t] for t in adversarial_examples]

    # 3. Intent Breakdown (Only for Hybrid B/C)
    if 'Hybrid' in name:
        print(f"--- {name} Intent Breakdown ---")
        df_res = valid_df.copy()
        df_res['pred'] = preds
        
        # False Positives (Safe -> Toxic)
        fp = df_res[(df_res['pred']==1) & (df_res['label']==0)]['intentClass'].value_counts()
        print("False Positives (Model says Toxic, GT Safe):")
        print(fp)
        
        # False Negatives (Toxic -> Safe)
        fn = df_res[(df_res['pred']==0) & (df_res['label']==1)]['intentClass'].value_counts()
        print("False Negatives (Model says Safe, GT Toxic):")
        print(fn)

# --- 5. Display Results ---
print("\n=== FINAL METRICS TABLE ===")
print(pd.DataFrame(metrics_data).round(4))

print("\n=== ADVERSARIAL STRESS TEST ===")
print(pd.DataFrame(adv_data))