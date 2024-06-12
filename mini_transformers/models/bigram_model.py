import torch
from typing import Optional, Tuple
from torch import nn
from torch.nn import functional as F
from mini_transformers.models.embedding_model import (
    BaseEmbedding,
)


class BigramModel(nn.Module):
    def __init__(self, embedding_model: BaseEmbedding) -> None:
        super().__init__()

        self.embedding_model = embedding_model

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        logits = self.embedding_model(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(
                logits,
                targets,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:

        idx = idx or torch.zeros((1, 1), dtype=torch.long)
        new_idx = idx.to(self.embedding_model.device)
        for _ in range(max_new_tokens):
            # execute on the last idx
            logits, _ = self(new_idx)
            # calculate probabilities on the last logits
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)  # dims -> B, C

            idx_next = torch.multinomial(probs, num_samples=1).to(
                self.embedding_model.device
            )  # get the next token

            new_idx = torch.cat([new_idx, idx_next], dim=1).to(
                self.embedding_model.device
            )  # attach the prediction to the corpus

        return new_idx
