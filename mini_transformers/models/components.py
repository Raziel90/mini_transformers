import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, n_embeds) -> None:
        super().__init__()
        
        self.net = nn.Sequential([
            nn.Linear(n_embeds, 4 * n_embeds),
            nn.ReLU(),
            nn.Linear(4 * n_embeds, n_embeds) # projection for Residual connections
        ])


class AttentionHead(nn.Module):
    
    def __init__(self, context_len: int, n_embeds: int, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embeds, head_size, bias=True)
        self.query = nn.Linear(n_embeds, head_size, bias=True)
        self.value = nn.Linear(n_embeds, head_size, bias=True)
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)
        
        wei = q @ k.transpose(-2, -1) * C ** (-0.5) # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # B, T, T
        
        out = wei @ v
        return out
        
    
class MultiHeadAttention(nn.Module):
    def __init__(self, context_len: int, n_embeds: int, n_heads: int, head_size: int) -> None:
        super().__init__()
        
        self.heads = nn.ModuleList([AttentionHead(context_len, n_embeds, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embeds, n_embeds) # useful for residual connections
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
        

class ResidualBlock(nn.Module):
    def __init__(self, context_len: int, n_embeds: int, n_heads: int) -> None:
        super().__init__()
        
        self.attention = MultiHeadAttention(context_len=context_len, n_embeds=n_embeds, n_heads=n_heads, head_size=n_embeds // n_heads)
        self.ff_net = FeedForward(n_embeds=n_embeds)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(n_embeds), nn.LayerNorm(n_embeds)])
        
    def forward(self, x):
        
        x = x + self.attention(self.layer_norms[0](x))
        x = x + self.ff_net(self.layer_norms[1](x))
        
        return x