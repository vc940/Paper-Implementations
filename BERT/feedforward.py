import torch
from torch import nn

class FFN(nn.Module):

    def __init__(self,embeddings_dims = 512):
        super().__init__()
        self.embeddings_dims = embeddings_dims
        self.fc1 = nn.Linear(self.embeddings_dims,4*self.embeddings_dims)
        self.fc2 = nn.Linear(self.embeddings_dims*4,self.embeddings_dims)
        self.ReLU = nn.ReLU()
    
    def forward(self,X):
        X = self.fc1(X)
        X = self.ReLU(X)
        X = self.fc2(X)
        return X

        