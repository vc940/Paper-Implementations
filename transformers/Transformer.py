from ENCODER import ENCODER_BLOCK
from DECODER import Decoder_Block
from torch import nn
import torch

class Transformer_Block(nn.Module):

    def __init__(self,embeddings_dims=512,attention_heads=2,Encoders=5,Decoder = 5):
        super().__init__()
        self.encoder  = nn.ModuleList([ENCODER_BLOCK(embeddings_dims=embeddings_dims,attention_heads=attention_heads) for _ in range(Encoders)])
        self.decoder = nn.ModuleList([Decoder_Block(embeddings_dims=embeddings_dims,attention_heads=attention_heads) for _ in range(Decoder)])
        
        
    def forward(self,X):
        Y = X
        for layer in self.encoder:
            Y = layer(Y)
        for layer in self.decoder:
            X = layer(X,Y)
        return X
