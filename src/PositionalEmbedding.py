from torch import nn
import torch

class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.positional_embedding = nn.Embedding(block_size, embedding_dim)

    def forward(self, x):
        a = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return self.positional_embedding(a)