from torch.nn import Embedding
from torch import nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = self.initialize_embeddings()

    def initialize_embeddings(self):
        return Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
    
    def forward(self, x):
        return self.embeddings(x)
