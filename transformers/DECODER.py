import torch
from torch import nn
from Add_Norm import AddandNorm
from attention import multi_head_attention
from feedforward import FFN
from positional_encoding import pos_encoding


class Decoder_Block(nn.Module):

    def __init__(self,embeddings_dims = 512,attention_heads = 2):
        super().__init__()
        assert embeddings_dims % attention_heads == 0 ,"Invalid  number of attention_heads select number such that it is divisible by embeddings dims"
        self.ffn = FFN(embeddings_dims=embeddings_dims)
        self.masked_attention = multi_head_attention(no_of_heads = attention_heads,embeddings_dims=embeddings_dims,Masked=True)
        self.cross_attention = multi_head_attention(no_of_heads = attention_heads,embeddings_dims=embeddings_dims,Cross=True)
        self.pos_encoding = pos_encoding(embeddings_dims=embeddings_dims)
        self.addnorm1 = AddandNorm(embeddings_dims=embeddings_dims) 
        self.addnorm2 = AddandNorm(embeddings_dims=embeddings_dims)
        self.addnorm3 = AddandNorm(embeddings_dims=embeddings_dims)  
    def forward(self,X,Encoder_output):
        X = self.pos_encoding(X)
        Y = self.masked_attention(X)
        X = self.addnorm1(X,Y)
        Y = self.cross_attention(X,Encoder_output)
        X = self.addnorm2(X,Y)
        Y = self.ffn(X)
        X = self.addnorm3(X,Y)
        return X
