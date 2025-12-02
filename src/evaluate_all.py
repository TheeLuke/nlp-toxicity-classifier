import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# --- CONFIGURATION ---
TEST_FILE = 'conda_valid.csv'
TFIDF_PATH = 'saved_models/tfidf_baseline.pkl'
BERT_PATH = 'saved_models/bert_baseline_best'
CHAR_PATH = 'saved_models/charcnn_baseline_best.pth'
HYBRID_PATH = 'saved_models/hybrid_model_best.pth'
OUTPUT_FILE = 'final_predictions.csv'
BATCH_SIZE = 32

class CharCNN(nn.Module):
    def __init__(self, num_classes=2, embedding_dim=128, max_len=1014, num_chars=71):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim, padding_idx=0)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(128, 256, 7), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(256, 256, 7), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(256, 256, 3), nn.ReLU(),
            nn.Conv1d(256, 256, 3), nn.ReLU(),
            nn.Conv1d(256, 256, 3), nn.ReLU(), nn.MaxPool1d(3)
        )
        self.fc_input_dim = 256 * 34 
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.conv_layers(x).view(x.size(0), -1)
        return self.fc_layers(x)

class HybridBERTCharCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(HybridBERTCharCNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
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
        self.cnn_projection = nn.Linear(256 * 34, 128)  # Matches new 128 dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 + 128, 512), # Matches 896 input
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, input_ids, attention_mask, char_input):
        bert_vec = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x_char = self.char_embedding(char_input).permute(0, 2, 1)
        x_char = self.char_cnn(x_char).view(x_char.size(0), -1)
        char_vec = self.cnn_projection(x_char)
        combined_vec = torch.cat((bert_vec, char_vec), dim=1)
        return self.classifier(combined_vec)

# --- MAIN EVALUATION LOOP ---
def evaluate():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"--- Running Evaluation on {device} ---")
    
    # 1. Load Data
    try:
        from dataset_loader import CONDADataset
    except ImportError:
        print("Error: data_loader.py not found in directory.")
        return

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # We use 'hybrid' mode to get BOTH tokens (for BERT) and chars (for CharCNN/Hybrid)
    # We set use_context=True so BERT/Hybrid get the full conversation history
    print("Loading Test Data...")
    test_ds = CONDADataset(TEST_FILE, model_type='hybrid', tokenizer=tokenizer, use_context=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Models
    print("Loading Models...")
    
    # A. TF-IDF
    if os.path.exists(TFIDF_PATH):
        tfidf_model = joblib.load(TFIDF_PATH)
        print("  TF-IDF Loaded.")
    else:
        print("  Warning: TF-IDF model not found. Skipping.")
        tfidf_model = None
    
    # B. BERT
    if os.path.exists(BERT_PATH):
        bert_model = BertForSequenceClassification.from_pretrained(BERT_PATH)
        bert_model.to(device)
        bert_model.eval()
        print("  BERT Loaded.")
    else:
        print("  Warning: BERT model not found. Skipping.")
        bert_model = None
    
    # C. CharCNN
    if os.path.exists(CHAR_PATH):
        char_model = CharCNN()
        char_model.load_state_dict(torch.load(CHAR_PATH, map_location=device))
        char_model.to(device)
        char_model.eval()
        print("  CharCNN Loaded.")
    else:
        print("  Warning: CharCNN model not found. Skipping.")
        char_model = None
    
    # D. Hybrid
    if os.path.exists(HYBRID_PATH):
        hybrid_model = HybridBERTCharCNN()
        hybrid_model.load_state_dict(torch.load(HYBRID_PATH, map_location=device))
        hybrid_model.to(device)
        hybrid_model.eval()
        print("  Hybrid Model Loaded.")
    else:
        print("  Warning: Hybrid model not found. Skipping.")
        hybrid_model = None
    
    # 3. Inference Loop
    print("Starting Inference...")
    results = []
    
    # Pre-calculate TF-IDF if model exists (faster than batching)
    tfidf_preds = []
    if tfidf_model:
        print("Generating TF-IDF Predictions...")
        tfidf_preds = tfidf_model.predict(test_ds.texts)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0: print(f"Processing batch {i}/{len(test_loader)}")
            
            # Move inputs to device
            b_ids = batch['input_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_char = batch['char_input'].to(device)
            
            # --- BERT Prediction ---
            if bert_model:
                bert_out = bert_model(b_ids, attention_mask=b_mask).logits
                bert_preds = torch.argmax(bert_out, dim=1).cpu().numpy()
            else:
                bert_preds = [-1] * len(b_ids)
            
            # --- CharCNN Prediction ---
            if char_model:
                char_out = char_model(b_char)
                char_preds = torch.argmax(char_out, dim=1).cpu().numpy()
            else:
                char_preds = [-1] * len(b_ids)
            
            # --- Hybrid Prediction ---
            if hybrid_model:
                hybrid_out = hybrid_model(input_ids=b_ids, attention_mask=b_mask, char_input=b_char)
                hybrid_preds = torch.argmax(hybrid_out, dim=1).cpu().numpy()
            else:
                hybrid_preds = [-1] * len(b_ids)
            
            # Store batch results
            start_idx = i * BATCH_SIZE
            
            # Metadata from batch
            labels = batch['labels'].cpu().numpy()
            has_slang = batch['has_slang'].cpu().numpy()
            has_dota = batch['has_dota'].cpu().numpy()
            has_tox = batch['has_toxicity'].cpu().numpy()
            
            for j in range(len(b_ids)):
                global_idx = start_idx + j
                
                # Handle TF-IDF index safely
                t_pred = tfidf_preds[global_idx] if tfidf_model else -1
                
                results.append({
                    'index': global_idx,
                    'label': labels[j],
                    'pred_tfidf': t_pred,
                    'pred_bert': bert_preds[j],
                    'pred_char': char_preds[j],
                    'pred_hybrid': hybrid_preds[j],
                    'has_slang': has_slang[j],
                    'has_dota': has_dota[j],
                    'has_toxicity': has_tox[j]
                })

    # 4. Save Results
    df = pd.DataFrame(results)
    
    # Attach raw text for qualitative analysis
    df['text'] = test_ds.texts
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"--- SUCCESS: Results saved to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    evaluate()