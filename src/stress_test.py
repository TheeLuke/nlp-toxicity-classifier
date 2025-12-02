import torch
import torch.nn as nn
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# --- 1. THE OVERWATCH STRESS DATASET ---
# I designed these to test specific failures (Slang, Context, Hero Names)
data = [
    # --- EXPLICIT TOXICITY (Easy) ---
    ("genji you are trash", 1),
    ("kys hanzo", 1),
    ("useless mercy", 1),
    
    # --- GAMING SLANG (The Hybrid Test) ---
    ("gg ez", 1),                     # Classic slang
    ("dps diff", 1),                  # "Diff" implies skill gap
    ("widow is hacking", 1),          # Accusation
    ("nice C9 lol", 1),               # "C9" = leaving objective (Contextual slang)
    
    # --- IMPLICIT / SARCASM (The Hardest) ---
    ("report our rein for throwing", 1),
    ("thanks for feeding winston", 1), # "Feeding" = dying on purpose
    ("great ult mccree", 1),           # Likely sarcastic if reported
    
    # --- NON-TOXIC / GAME COMMS (False Positive Traps) ---
    ("can we get a shield?", 0),
    ("group up with me", 0),
    ("I need healing", 0),             # The classic meme (should be safe)
    ("enemy widow top right", 0),
    ("nice shot ashe", 0),
    ("gr", 0),                         # Good round
    ("gl hf", 0)                       # Good luck have fun
]

texts = [x[0] for x in data]
labels = [x[1] for x in data]

# --- 2. MODEL ARCHITECTURES (Must match saved weights) ---
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
    def __init__(self, num_classes=2):
        super(HybridBERTCharCNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.char_embedding = nn.Embedding(71, 128, padding_idx=0)
        self.char_cnn = nn.Sequential(
            nn.Conv1d(128, 256, 7), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(256, 256, 7), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(256, 256, 3), nn.ReLU(),
            nn.Conv1d(256, 256, 3), nn.ReLU(),
            nn.Conv1d(256, 256, 3), nn.ReLU(), nn.MaxPool1d(3)
        )
        # Option B Architecture (128 dim projection)
        self.cnn_projection = nn.Linear(256 * 34, 128)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 + 128, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, input_ids, attention_mask, char_input):
        bert_vec = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x_char = self.char_embedding(char_input).permute(0, 2, 1)
        x_char = self.char_cnn(x_char).view(x_char.size(0), -1)
        char_vec = self.cnn_projection(x_char)
        return self.classifier(torch.cat((bert_vec, char_vec), dim=1))

# --- 3. PREPROCESSING HELPERS ---
def get_char_tensor(text):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
    char_dict = {char: i + 1 for i, char in enumerate(alphabet)}
    indices = [char_dict.get(c, 0) for c in text.lower()]
    indices = indices[:1014] + [0]*(1014 - len(indices))
    return torch.tensor([indices], dtype=torch.long)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Overwatch Stress Test on {device}...")

    # Load Models
    print("Loading Models...")
    tfidf = joblib.load('saved_models/tfidf_baseline.pkl')
    
    bert = BertForSequenceClassification.from_pretrained('saved_models/bert_baseline_best')
    bert.to(device); bert.eval()
    
    char = CharCNN()
    char.load_state_dict(torch.load('saved_models/charcnn_baseline_best.pth', map_location=device))
    char.to(device); char.eval()
    
    hybrid = HybridBERTCharCNN()
    hybrid.load_state_dict(torch.load('saved_models/hybrid_model_best.pth', map_location=device))
    hybrid.to(device); hybrid.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Run Inference
    results = []
    print("\nProcessing Examples...")
    
    for i, text in enumerate(texts):
        # Prepare inputs
        inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)
        chars = get_char_tensor(text).to(device)
        
        # Predictions
        with torch.no_grad():
            p_tfidf = tfidf.predict([text])[0]
            p_bert = torch.argmax(bert(ids, attention_mask=mask).logits, dim=1).item()
            p_char = torch.argmax(char(chars), dim=1).item()
            p_hybrid = torch.argmax(hybrid(input_ids=ids, attention_mask=mask, char_input=chars), dim=1).item()
        
        results.append({
            'Message': text,
            'Label': labels[i],
            'TF-IDF': p_tfidf,
            'BERT': p_bert,
            'CharCNN': p_char,
            'Hybrid': p_hybrid
        })

    # Output Results
    df = pd.DataFrame(results)
    
    # Add a "Result" column: Did Hybrid win?
    # Win = Hybrid Correct AND BERT Incorrect
    df['Hybrid Win?'] = (df['Hybrid'] == df['Label']) & (df['BERT'] != df['Label'])
    
    print("\n" + "="*80)
    print("OVERWATCH STRESS TEST RESULTS")
    print("="*80)
    print(df[['Message', 'Label', 'BERT', 'Hybrid', 'Hybrid Win?']].to_string(index=False))
    
    # Save for Poster
    df.to_csv('overwatch_stress_test_results.csv', index=False)
    print("\nResults saved to 'overwatch_stress_test_results.csv'")

if __name__ == "__main__":
    main()