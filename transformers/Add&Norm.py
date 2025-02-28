import torch
from torch import nn

class AddandNorm(nn.Module):

    def __init__(self,embeddings_dims = 512):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=(embeddings_dims))
    
    def forward(self,X,Y):
        Add =  torch.add(X,Y)
        result = self.layer_norm(Add)
        return result
