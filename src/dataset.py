import torch
from torch.utils.data import Dataset
import random
from config import Config

class CharTokenizer:
    def __init__(self):
        self.vocab = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{} "
        self.char2idx = {char: idx + 2 for idx, char in enumerate(self.vocab)}
        self.char2idx["<PAD>"] = 0
        self.char2idx["<UNK>"] = 1
        self.max_len = Config.MAX_LEN_CHAR

    def encode(self, text):
        text = str(text).lower()
        indices = [self.char2idx.get(char, 1) for char in text]
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long)

    def get_vocab_size(self):
        return len(self.char2idx)

def augment_text(text, prob=0.3):
    """Simulates gamer typos (spacing, deletions, swaps)."""
    if random.random() > prob:
        return text
    
    chars = list(text)
    try:
        aug_type = random.choice(['space', 'del', 'swap'])
        if aug_type == 'space' and len(chars) > 1:
            chars.insert(random.randint(1, len(chars)-1), ' ')
        elif aug_type == 'del' and len(chars) > 1:
            del chars[random.randint(0, len(chars)-1)]
        elif aug == 'swap' and len(chars) > 1:
            i = random.randint(0, len(chars)-2)
            chars[i], chars[i+1] = chars[i+1], chars[i]
    except:
        pass 
    return "".join(chars)

class HybridDataset(Dataset):
    def __init__(self, df, tokenizer_bert, tokenizer_char, augment=False):
        self.df = df.reset_index(drop=True)
        self.bert_tok = tokenizer_bert
        self.char_tok = tokenizer_char
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        
        # 1. Context for BERT
        context_text = str(row['context_text'])
        
        # 2. Target for CharCNN
        target_text = str(row['utterance'])
        
        # Augmentation: Apply only during training, only to Toxic class
        if self.augment and label == 1:
            target_text = augment_text(target_text, prob=Config.AUGMENT_PROB)
        
        # Tokenize
        bert_enc = self.bert_tok(
            context_text, 
            truncation=True, 
            padding='max_length', 
            max_length=Config.MAX_LEN_BERT, 
            return_tensors='pt'
        )
        char_ids = self.char_tok.encode(target_text)
        
        return {
            'b_ids': bert_enc['input_ids'].flatten(),
            'b_mask': bert_enc['attention_mask'].flatten(),
            'c_ids': char_ids,
            'labels': torch.tensor(label, dtype=torch.long),
            'text': target_text # Useful for error analysis
        }