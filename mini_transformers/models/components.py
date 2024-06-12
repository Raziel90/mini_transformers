import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, cast


class FeedForward(nn.Module):
    def __init__(self, n_embeds: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embeds, 4 * n_embeds),
            nn.ReLU(),
            nn.Linear(4 * n_embeds, n_embeds),  # projection for Residual connections
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(Tensor, self.net(x))


class AttentionHead(nn.Module):

    def __init__(
        self, context_len: int, n_embeds: int, head_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embeds, head_size, bias=True)
        self.query = nn.Linear(n_embeds, head_size, bias=True)
        self.value = nn.Linear(n_embeds, head_size, bias=True)
        self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)

        wei = q @ k.transpose(-2, -1) * C ** (-0.5)  # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # B, T, T
        wei = self.dropout(wei)
        out = wei @ v
        return cast(Tensor, out)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        context_len: int,
        n_embeds: int,
        n_heads: int,
        head_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.n_embeds = n_embeds
        self.n_heads = n_heads
        self.head_size = head_size
        self.heads = nn.ModuleList(
            [AttentionHead(context_len, n_embeds, head_size) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(
            n_heads * head_size, n_embeds
        )  # useful for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        attention = torch.cat([head(x) for head in self.heads], dim=-1)
        return cast(Tensor, self.proj(attention))


class ResidualBlock(nn.Module):
    def __init__(
        self, context_len: int, n_embeds: int, n_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(
            context_len=context_len,
            n_embeds=n_embeds,
            n_heads=n_heads,
            head_size=n_embeds // n_heads,
            dropout=dropout,
        )
        self.ff_net = FeedForward(n_embeds=n_embeds, dropout=dropout)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(n_embeds), nn.LayerNorm(n_embeds)]
        )

    def forward(self, x: Tensor) -> Tensor:

        x = x + self.attention(self.layer_norms[0](x))
        x = x + self.ff_net(self.layer_norms[1](x))

        return x
