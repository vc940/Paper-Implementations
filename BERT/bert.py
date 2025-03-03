from ENCODER import ENCODER_BLOCK
import torch
from torch import nn


class BERT(nn.Module):
    def __init__(self,vocab_size,embedding_dims=768,max_position_embed = 512 , max_vocab_size = 2,encoder_blocks= 12,Attention_heads = 12):
        super().__init__()
        encoders = [ENCODER_BLOCK(embeddings_dims=768,attention_heads=Attention_heads) for _ in range(encoder_blocks)]
        self.model = nn.Sequential(*encoders)
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dims)
        self.positional_embeddings = nn.Embedding(max_position_embed,embedding_dims)
        self.token_type_embeddings = nn.Embedding(max_vocab_size ,embedding_dims)
        self.LayerNorm = nn.LayerNorm(embedding_dims, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self,input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)  
        word_embeddings = self.word_embeddings(input_ids)
        pos_embeddings = self.positional_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeddings + pos_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return self.model(embeddings)
