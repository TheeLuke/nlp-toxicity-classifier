import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import re

class CONDADataset(Dataset):
    """
    Custom Dataset class for loading and preprocessing CONDA data.
    Includes logic for Contextual Concatenation and Slot Annotation parsing (S, D, T).
    """
    def __init__(self, file_path, model_type='bert', tokenizer=None, max_len=128, use_context=False):
        """
        Args:
            file_path (str): Path to the CONDA csv file.
            model_type (str): 'bert', 'charcnn', 'hybrid', or 'tfidf'.
            tokenizer: Hugging Face tokenizer (required for bert/hybrid).
            max_len (int): Maximum sequence length.
            use_context (bool): If True, prepends the previous utterance in the conversation.
        """
        self.data = pd.read_csv(file_path)
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # --- 1. Label Mapping ---
        # Maps 'intentClass' to binary: Explicit(E)/Implicit(I) -> 1, Action(A)/Other(O) -> 0
        if 'intentClass' not in self.data.columns:
             if 'label' in self.data.columns:
                 self.label_col = 'label'
             else:
                 raise ValueError("Column 'intentClass' not found in CSV.")
        else:
             self.label_col = 'intentClass'
             
        self.data['binary_label'] = self.data[self.label_col].apply(
            lambda x: 1 if x in ['E', 'I'] else 0
        )
        
        # --- 2. Slot Annotation Parsing (Updated) ---
        # We look for 'S' (Slang), 'D' (Dota-specific), and 'T' (Toxic) tags.
        if 'slotClasses' in self.data.columns:
            # Ensure column is string, handle NaNs
            self.data['slotClasses'] = self.data['slotClasses'].astype(str).fillna('')
            
            # Create boolean flags
            self.data['has_slang'] = self.data['slotClasses'].apply(lambda x: 'S' in x)
            self.data['has_dota'] = self.data['slotClasses'].apply(lambda x: 'D' in x)
            self.data['has_toxicity'] = self.data['slotClasses'].apply(lambda x: 'T' in x)
        else:
            # Defaults if column is missing
            self.data['has_slang'] = False
            self.data['has_dota'] = False
            self.data['has_toxicity'] = False

        # --- 3. Context Engineering ---
        if use_context:
            self.data = self.data.sort_values(by=['matchId', 'conversationId', 'Id'])
            self.data['prev_utterance'] = self.data.groupby(['matchId', 'conversationId'])['utterance'].shift(1)
            self.data['prev_utterance'] = self.data['prev_utterance'].fillna("") 
            
            # Concatenate: "Previous [SEP] Current"
            self.data['full_text'] = self.data['prev_utterance'] + " [SEP] " + self.data['utterance'].astype(str)
            self.data['full_text'] = self.data['full_text'].str.replace(r"^ \[SEP\] ", "", regex=True)
            
            self.texts = self.data['full_text'].tolist()
        else:
            self.texts = self.data['utterance'].astype(str).tolist()

        self.labels = self.data['binary_label'].tolist()
        self.has_slang = self.data['has_slang'].tolist()
        self.has_dota = self.data['has_dota'].tolist()
        self.has_toxicity = self.data['has_toxicity'].tolist()

        # --- 4. CharCNN Setup ---
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        self.char_dict = {char: i + 1 for i, char in enumerate(self.alphabet)} 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Metadata flags for analysis
        has_slang = 1 if self.has_slang[idx] else 0
        has_dota = 1 if self.has_dota[idx] else 0
        has_toxicity = 1 if self.has_toxicity[idx] else 0

        # Output dictionary
        sample = {
            'labels': torch.tensor(label, dtype=torch.long),
            'has_slang': torch.tensor(has_slang, dtype=torch.long),
            'has_dota': torch.tensor(has_dota, dtype=torch.long),
            'has_toxicity': torch.tensor(has_toxicity, dtype=torch.long)
        }

        # --- A. TF-IDF Mode ---
        if self.model_type == 'tfidf':
            return text, label

        # --- B. BERT Mode ---
        elif self.model_type == 'bert':
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            sample['input_ids'] = encoding['input_ids'].flatten()
            sample['attention_mask'] = encoding['attention_mask'].flatten()

        # --- C. CharCNN Mode ---
        elif self.model_type == 'charcnn':
            char_indices = self._preprocess_char_cnn(text)
            sample['char_input'] = char_indices

        # --- D. Hybrid Mode ---
        elif self.model_type == 'hybrid':
            # 1. BERT tokens
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            sample['input_ids'] = encoding['input_ids'].flatten()
            sample['attention_mask'] = encoding['attention_mask'].flatten()
            
            # 2. Char indices
            char_indices = self._preprocess_char_cnn(text)
            sample['char_input'] = char_indices

        return sample

    def _preprocess_char_cnn(self, text):
        """
        Converts text string to a tensor of character indices.
        Truncates or pads to max_len.
        """
        text = text.lower()
        indices = [self.char_dict.get(c, 0) for c in text] 
        
        limit = 1014 if self.model_type in ['charcnn', 'hybrid'] else self.max_len
        
        if len(indices) > limit:
            indices = indices[:limit]
        else:
            indices = indices + [0] * (limit - len(indices))
            
        return torch.tensor(indices, dtype=torch.long)

# --- Helper for standard usage ---
def get_dataloaders(train_path, val_path, model_type, tokenizer=None, batch_size=32, use_context=False):
    train_ds = CONDADataset(train_path, model_type=model_type, tokenizer=tokenizer, use_context=use_context)
    val_ds = CONDADataset(val_path, model_type=model_type, tokenizer=tokenizer, use_context=use_context)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader