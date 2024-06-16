from torch import nn
from torch import Tensor
from typing import cast


from mini_transformers.models.components import MultiHeadAttention, ResidualBlock

import torch


class BaseEmbedding(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, idx: Tensor) -> Tensor:
        raise NotImplementedError("BaseEmbedding should not be instantiated!")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class SimpleEmbedding(BaseEmbedding):

    def __init__(self, vocab_size: int) -> None:
        super().__init__(vocab_size)

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.apply(self._init_weights)

    def forward(self, idx: Tensor) -> Tensor:

        logits = self.token_embedding_table(idx)

        return cast(Tensor, logits)


class HeadEmbedding(BaseEmbedding):

    def __init__(self, vocab_size: int, n_embeds: int) -> None:
        super().__init__(vocab_size)
        # self.vocab_size = vocab_size
        self.n_embeds = n_embeds
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)
        self.apply(self._init_weights)

    def forward(self, idx: Tensor) -> Tensor:
        tok_embeddings = self.token_embedding_table(idx)  # (B, T, vocab_size)
        logits = self.lm_head(tok_embeddings)  # (B, T, vocab_size)
        return cast(Tensor, logits)


class PositionHeadEmbedding(BaseEmbedding):

    def __init__(self, vocab_size: int, n_embeds: int, context_len: int) -> None:
        super().__init__(vocab_size)
        # self.vocab_size = vocab_size
        self.n_embeds = n_embeds
        self.context_len = context_len
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.position_embedding_table = nn.Embedding(context_len, n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)
        self.apply(self._init_weights)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = (idx_cond := idx[:, -self.context_len :]).shape
        tok_embeddings = self.token_embedding_table(idx_cond)  # (B, T, n_embeds)
        pos_embeddings = self.position_embedding_table(
            torch.arange(0, T).to(idx.device)
        )  # (T, n_embeds)
        x = (
            tok_embeddings + pos_embeddings
        )  # broadcast on the B dimension (B, T, n_embeds)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return cast(Tensor, logits)


class MultiHeadedAttentionEmbedding(BaseEmbedding):

    def __init__(
        self,
        vocab_size: int,
        n_embeds: int,
        n_heads: int,
        head_size: int,
        context_len: int,
    ) -> None:
        super().__init__(vocab_size)
        self.n_embeds = n_embeds
        self.context_len = context_len
        self.sa_attention_head = MultiHeadAttention(
            context_len=context_len,
            n_embeds=n_embeds,
            n_heads=n_heads,
            head_size=head_size,
        )
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.position_embedding_table = nn.Embedding(context_len, n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)
        self.apply(self._init_weights)

    def forward(self, idx: Tensor) -> Tensor:

        B, T = (idx_cond := idx[:, -self.context_len :]).shape
        tok_embeddings = self.token_embedding_table(idx_cond)  # (B, T, n_embeds)
        pos_embeddings = self.position_embedding_table(
            torch.arange(0, T).to(idx.device)
        )  # (T, n_embeds)
        x = (
            tok_embeddings + pos_embeddings
        )  # broadcast on the B dimension (B, T, n_embeds)
        x = self.sa_attention_head(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return cast(Tensor, logits)


class ResidualBlockAttentionEmbedding(BaseEmbedding):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        context_len: int,
        n_embeds: int,
        n_heads: int,
    ) -> None:
        super().__init__(vocab_size)
        self.n_embeds = n_embeds
        self.context_len = context_len

        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.position_embedding_table = nn.Embedding(context_len, n_embeds)

        self.residual_net = nn.Sequential(
            *[
                ResidualBlock(
                    context_len=context_len, n_embeds=n_embeds, n_heads=n_heads
                )
                for _ in range(n_layers)
            ],
            nn.LayerNorm(n_embeds),
            nn.Linear(n_embeds, vocab_size),
        )
        self.apply(self._init_weights)

    def forward(self, idx: Tensor) -> Tensor:

        B, T = (idx_cond := idx[:, -self.context_len :]).shape
        tok_embeddings = self.token_embedding_table(idx_cond)  # (B, T, n_embeds)
        pos_embeddings = self.position_embedding_table(
            torch.arange(0, T).to(idx.device)
        )  # (T, n_embeds)
        x = (
            tok_embeddings + pos_embeddings
        )  # broadcast on the B dimension (B, T, n_embeds)
        return cast(Tensor, self.residual_net(x))


class GPT(BaseEmbedding):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        context_len: int,
        n_embeds: int,
        n_heads: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(vocab_size)

        self.n_embeds = n_embeds
        self.context_len = context_len

        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.position_embedding_table = nn.Embedding(context_len, n_embeds)

        self.residual_net = nn.Sequential(
            *[
                ResidualBlock(
                    context_len=context_len,
                    n_embeds=n_embeds,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ],
            nn.LayerNorm(n_embeds),
        )
        self.model_head = nn.Linear(n_embeds, vocab_size)
        self.token_embedding_table.weight = self.model_head.weight
        self.apply(self._init_weights)

    def forward(self, idx: Tensor) -> Tensor:

        B, T = (idx_cond := idx[:, -self.context_len :]).shape
        tok_embeddings = self.token_embedding_table(idx_cond)  # (B, T, n_embeds)
        pos_embeddings = self.position_embedding_table(
            torch.arange(0, T).to(idx.device)
        )  # (T, n_embeds)
        x = (
            tok_embeddings + pos_embeddings
        )  # broadcast on the B dimension (B, T, n_embeds)
        x = self.residual_net(x)
        return cast(Tensor, self.model_head(x))
