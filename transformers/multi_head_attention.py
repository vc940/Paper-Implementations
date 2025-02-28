import torch
from torch import nn

class multi_head_attention(nn.Module):
    def __init__(self,no_of_heads,embeddings_dims=512):
        super().__init__( )
        self.heads = no_of_heads
        self.head_dims = embeddings_dims// no_of_heads
        self.embeddings_dims = embeddings_dims
        self.k = nn.Linear(embeddings_dims,embeddings_dims)
        self.q = nn.Linear(embeddings_dims,embeddings_dims)  
        self.v = nn.Linear(embeddings_dims,embeddings_dims)
        self.transform = nn.Linear(embeddings_dims,embeddings_dims)

    def forward( self,X):
        batch_size,sequence_length,_= X.shape
        K = self.k(X)
        Q = self.q(X)
        V = self.v(X)
        K = K.view(batch_size,sequence_length,self.heads,self.head_dims)
        Q = Q.view(batch_size,sequence_length,self.heads,self.head_dims)
        V = V.view(batch_size,sequence_length,self.heads,self.head_dims)
        K = K.permute(0, 2, 1, 3)
        Q = Q.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        KT = torch.transpose(K,-1,-2)
        Attention_matrix = torch.divide(torch.matmul(Q,KT),self.head_dims**0.5)
        Attention_matrix = torch.softmax(Attention_matrix,dim = -1)
        contextual_embeddings = torch.matmul(Attention_matrix,V)
        contextual_embeddings = contextual_embeddings.reshape(batch_size,sequence_length,self.embeddings_dims)
        contextual_embeddings = self.transform(contextual_embeddings)
        return contextual_embeddings
    