from src.Embedding import EmbeddingLayer
from src.PositionalEmbedding import PositionalEmbedding
from src.TransformerBlock import TransformerBlock
from torch import nn
from torch.nn import ModuleList


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layers=6, n_heads=4, block_size=32):
        super(Model, self).__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(
            block_size=block_size, embedding_dim=embedding_dim
        )
        self.transformer_block = ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=n_heads,
                    block_size=block_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding_layer(x)
        positional = self.positional_embedding(x)

        embedded = embedded + positional

        output = embedded
        for block in self.transformer_block:
            output = block(output)

        output = self.final_layer(output)  # (B, S, E) -> (B, S, V)

        return output
