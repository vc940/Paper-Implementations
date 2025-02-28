import torch 
from torch import nn

class pos_encoding(nn.Module):
    def __init__(self,embeddings_dims = 512,max_len = 5000):
        super().__init__()
        self.positional_matrix = torch.zeros((max_len,embeddings_dims))
        row_index = torch.arange(0,max_len)
        even_index = torch.arange(0,embeddings_dims,2)
        odd_index = torch.arange(1,embeddings_dims,2)
        den = torch.pow(10000,2*torch.arange(0,embeddings_dims,2).float()/embeddings_dims)
        self.positional_matrix[:,even_index] = torch.sin(row_index[:,None]/den)
        self.positional_matrix[:,odd_index] = torch.cos(row_index[:,None]/den)
        self.register_buffer('positional_matrix_', self.positional_matrix) 
    def forward(self,X):
        return X + self.positional_matrix[:X.size(1), :].unsqueeze(0)
