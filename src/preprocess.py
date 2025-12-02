import pandas as pd
import numpy as np
from config import Config

def create_label(row):
    """Maps CONDA intents to Binary Toxicity."""
    # E (Explicit) & I (Implicit) -> Toxic (1)
    # A (Action) & O (Other) -> Safe (0)
    if row['intentClass'] in ['E', 'I']:
        return 1
    return 0

def add_context_window(df):
    """
    Constructs conversation history for Context-Aware models.
    Output Format: [BEFORE] prev [CURRENT] curr [AFTER] next
    """
    print("  - Sorting by match and time...")
    df = df.sort_values(by=['matchId', 'conversationId', 'chatTime']).reset_index(drop=True)
    
    print("  - Shifting rows to find context...")
    df['prev_utterance'] = df.groupby('conversationId')['utterance'].shift(1).fillna("")
    df['next_utterance'] = df.groupby('conversationId')['utterance'].shift(-1).fillna("")
    
    def construct_window(row):
        prev = str(row['prev_utterance'])
        curr = str(row['utterance'])
        nxt = str(row['next_utterance'])
        
        context = f"[CURRENT] {curr}"
        if prev:
            context = f"[BEFORE] {prev} {context}"
        if nxt:
            context = f"{context} [AFTER] {nxt}"
        return context

    print("  - Building context strings...")
    df['context_text'] = df.apply(construct_window, axis=1)
    return df

def main():
    print("--- Starting Data Preprocessing Phase ---")
    
    # 1. Load
    print(f"Loading {Config.RAW_TRAIN_PATH} and {Config.RAW_VALID_PATH}...")
    try:
        train_df = pd.read_csv(Config.RAW_TRAIN_PATH)
        valid_df = pd.read_csv(Config.RAW_VALID_PATH)
    except FileNotFoundError:
        print("ERROR: Raw CSV files not found. Check paths in config.py")
        return

    # 2. Clean (Drop NaNs)
    train_df = train_df.dropna(subset=['utterance', 'intentClass']).reset_index(drop=True)
    valid_df = valid_df.dropna(subset=['utterance', 'intentClass']).reset_index(drop=True)

    # 3. Label
    print("Generating Labels...")
    train_df['label'] = train_df.apply(create_label, axis=1)
    valid_df['label'] = valid_df.apply(create_label, axis=1)

    # 4. Context Window
    print("Processing Train Set Context...")
    train_df = add_context_window(train_df)
    print("Processing Test Set Context...")
    valid_df = add_context_window(valid_df)

    # 5. Save
    print(f"Saving standardized datasets to:")
    print(f"  -> {Config.PROCESSED_TRAIN_PATH}")
    print(f"  -> {Config.PROCESSED_TEST_PATH}")
    
    train_df.to_csv(Config.PROCESSED_TRAIN_PATH, index=False)
    valid_df.to_csv(Config.PROCESSED_TEST_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()