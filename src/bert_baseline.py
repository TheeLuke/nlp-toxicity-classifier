import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import os
import time

from dataset_loader import CONDADataset 

def train_bert_baseline(train_path, val_path, save_dir='saved_models', epochs=4, batch_size=16):
    """
    Fine-tunes BERT on the CONDA dataset and saves the best performing model.
    """
    
    # --- 1. Setup & Configuration ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Tokenizer
    # We use 'bert-base-uncased' as per your report [cite: 58]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # --- 2. Load Data ---
    print("--- Loading Data for BERT ---")
    
    # CRITICAL: We enable use_context=True here.
    # This gives BERT the "Previous [SEP] Current" input structure 
    # to detect the "Implicit" toxicity your report said was missing.
    train_dataset = CONDADataset(train_path, model_type='bert', tokenizer=tokenizer, use_context=True)
    val_dataset = CONDADataset(val_path, model_type='bert', tokenizer=tokenizer, use_context=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 3. Initialize Model ---
    print("--- Initializing BERT Model ---")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2, # Binary: Toxic (1) vs Non-Toxic (0)
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    # --- 4. Optimizer & Scheduler ---
    # Standard hyperparameters for BERT fine-tuning
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    
    total_steps = len(train_loader) * epochs
    
    # Create the learning rate scheduler (linear warmup and decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )

    # --- 5. Training Loop ---
    print(f"--- Starting Training for {epochs} Epochs ---")
    
    best_val_loss = float('inf')
    
    for epoch_i in range(0, epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
        t0 = time.time()
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_loader):
            # Unpack batch and copy to GPU
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            model.zero_grad()        

            # Forward pass
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent exploding gradients
            optimizer.step()
            scheduler.step()

            if step % 100 == 0 and not step == 0:
                print(f'  Batch {step}  of  {len(train_loader)}.  Loss: {loss.item():.3f}')

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"  Average training loss: {avg_train_loss:.3f}")

        # --- Validation Phase ---
        print("  Running Validation...")
        model.eval()
        total_eval_loss = 0

        for batch in val_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
                
            loss = outputs.loss
            total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(val_loader)
        print(f"  Validation Loss: {avg_val_loss:.3f}")

        # --- Checkpointing (Prevent Overfitting) ---
        # If this model is better than the previous best, save it.
        if avg_val_loss < best_val_loss:
            print(f"  Validation Loss Improved ({best_val_loss:.3f} -> {avg_val_loss:.3f}). Saving model...")
            best_val_loss = avg_val_loss
            
            # Save model and tokenizer
            save_path = os.path.join(save_dir, 'bert_baseline_best')
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            print("  Validation Loss did not improve.")

    print("\nTraining complete.")

if __name__ == "__main__":
    # Update paths
    TRAIN_FILE = 'conda_train.csv' 
    VAL_FILE = 'conda_valid.csv'
    
    train_bert_baseline(TRAIN_FILE, VAL_FILE)