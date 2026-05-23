from torch import nn
from src.SelfAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(embedding_dim, num_heads, block_size)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.l1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, x):
        
        #out of MHA
        mha_output = self.mha(x)
        output_1 = x + mha_output
        output_1 = self.layer_norm(output_1) # (B, S, E) -> (B, S, E)
        
        # into feed forward network
        output_2 = self.l1(output_1) # (B, S, E) -> (B, S, E)
        output2 = output_1 + output_2 # (B, S, E) + (B, S, E) -> (B, S, E)
        output = self.layer_norm(output2) # (B, S, E) -> (B, S, E)
        
        return output