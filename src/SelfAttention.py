from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super(MultiHeadAttention, self).__init__()
        assert (
            embedding_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.head_size = embedding_dim // num_heads
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [
                SelfAttention(embedding_dim, self.head_size, block_size)
                for _ in range(num_heads)
            ]
        )
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        head_outputs = [head(x) for head in self.attention_heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(concatenated)
        return output


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.block_size = block_size
        self.query = nn.Linear(embedding_dim, head_size)
        self.key = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        Q = self.query(x)  # (B, S, E) -> (B, S, H)
        K = self.key(x)  # (B, S, E) -> (B, S, H)
        V = self.value(x)  # (B, S, E) -> (B, S, H)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.embedding_dim**0.5
        )  # (B, S, H) @ (B, H, S) -> (B, S, S)

        attention_scores = attention_scores.masked_fill(self.mask == 0, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        return output
