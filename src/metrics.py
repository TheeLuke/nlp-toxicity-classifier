import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# --- CONFIGURATION ---
INPUT_FILE = 'final_predictions.csv'

def calculate_metrics(df, model_col, label_col='label'):
    """
    Calculates standard metrics for a given model column against the labels.
    """
    if len(df) == 0:
        return 0, 0, 0, 0
    
    y_true = df[label_col]
    y_pred = df[model_col]
    
    # We use 'binary' average because we care about the Positive (Toxic) class
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return acc, prec, rec, f1

def print_row(name, metrics):
    acc, prec, rec, f1 = metrics
    print(f"{name:<15} | {acc:.4f}   | {prec:.4f}    | {rec:.4f} | {f1:.4f}")

def main():
    print(f"--- Loading Results from {INPUT_FILE} ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: File not found. Did you run 'evaluate_all.py'?")
        return

    # Define the models we want to compare
    models = {
        'TF-IDF': 'pred_tfidf',
        'BERT': 'pred_bert',
        'CharCNN': 'pred_char',
        'Hybrid': 'pred_hybrid'
    }

    print("\n" + "="*65)
    print(f"{'EXPERIMENT 1: OVERALL PERFORMANCE':^65}")
    print("="*65)
    print(f"{'Model':<15} | Accuracy | Precision | Recall | F1 (Toxic)")
    print("-" * 65)
    
    for name, col in models.items():
        metrics = calculate_metrics(df, col)
        print_row(name, metrics)

    # --- SUBSET ANALYSIS ---
    
    # Subset 1: Slang (The most important proof)
    df_slang = df[df['has_slang'] == 1]
    
    print("\n" + "="*65)
    print(f"{'EXPERIMENT 2: SLANG SUBSET (N=' + str(len(df_slang)) + ')':^65}")
    print("="*65)
    print(f"{'Model':<15} | Accuracy | Precision | Recall | F1 (Toxic)")
    print("-" * 65)
    
    for name, col in models.items():
        metrics = calculate_metrics(df_slang, col)
        print_row(name, metrics)
        
    # Subset 2: Dota/Game Jargon
    df_dota = df[df['has_dota'] == 1]
    
    print("\n" + "="*65)
    print(f"{'EXPERIMENT 3: DOTA JARGON SUBSET (N=' + str(len(df_dota)) + ')':^65}")
    print("="*65)
    print(f"{'Model':<15} | Accuracy | Precision | Recall | F1 (Toxic)")
    print("-" * 65)
    
    for name, col in models.items():
        metrics = calculate_metrics(df_dota, col)
        print_row(name, metrics)

    # --- WIN ANALYSIS (For your Poster Case Studies) ---
    # Find specific examples where BERT failed but Hybrid succeeded
    print("\n" + "="*65)
    print(f"{'CASE STUDIES: HYBRID WINS':^65}")
    print("="*65)
    
    # Filter: Actual Toxic, BERT said Safe (0), Hybrid said Toxic (1)
    wins = df[(df['label'] == 1) & (df['pred_bert'] == 0) & (df['pred_hybrid'] == 1)]
    
    print(f"Found {len(wins)} examples where Hybrid caught toxicity that BERT missed.")
    print("Top 5 IDs (Use these to look up the text in your CSV):")
    print(wins[['index', 'has_slang', 'has_dota']].head(5))

if __name__ == "__main__":
    main()