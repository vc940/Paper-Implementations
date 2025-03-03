import torch
from torch import nn
from Add_Norm import AddandNorm
from feedforward import FFN
from attention import multi_head_attention

class ENCODER_BLOCK(nn.Module):
    def __init__(self,embeddings_dims = 512,attention_heads =2):
        super().__init__()
        assert embeddings_dims % attention_heads == 0 ,"Invalid  number of attention_heads select number such that it is divisible by embeddings dims"
        self.ffn = FFN(embeddings_dims=embeddings_dims)
        self.attention = multi_head_attention(no_of_heads = attention_heads,embeddings_dims=embeddings_dims)
        self.addnorm1 = AddandNorm(embeddings_dims=embeddings_dims) 
        self.addnorm2 = AddandNorm(embeddings_dims=embeddings_dims) 

    def forward(self,X):
        Y = self.attention(X)
        X = self.addnorm1(X,Y)
        Y = self.ffn(X)
        X = self.addnorm2(X,Y)
        return X


        