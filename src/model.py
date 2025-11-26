import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class HybridBERT(nn.Module):
    def __init__(self, char_vocab_size, bert_name='bert-base-uncased', 
                 embed_dim=64, filters=128, kernels=[3, 4, 5]):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        # Note: Token embeddings must be resized in the training script after tokenizer load
        
        # CharCNN Branch
        self.char_emb = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, filters, k) for k in kernels
        ])
        
        # Calculate dimensions
        cnn_out_dim = filters * len(kernels)
        
        # Batch Norm for CNN branch (helps convergence)
        self.cnn_bn = nn.BatchNorm1d(cnn_out_dim)
        
        # Fusion & Classifier
        self.dropout = nn.Dropout(0.5)
        # Input: BERT (768) + CNN features
        self.fc = nn.Linear(768 + cnn_out_dim, 2)

    def forward(self, b_ids, b_mask, c_ids):
        # Path A: BERT Context
        b_out = self.bert(b_ids, attention_mask=b_mask).pooler_output
        
        # Path B: CharCNN Target
        x = self.char_emb(c_ids).permute(0, 2, 1) # [Batch, Emb, Seq]
        c_out = [F.max_pool1d(F.relu(conv(x)), conv(x).shape[2]).squeeze(2) for conv in self.convs]
        c_out = torch.cat(c_out, dim=1)
        c_out = self.cnn_bn(c_out)
        
        # Fusion
        combined = torch.cat((b_out, c_out), dim=1)
        return self.fc(self.dropout(combined))