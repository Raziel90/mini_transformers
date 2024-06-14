import lightning as pl
import torch.nn as nn
import torch
from mini_transformers.models.embedding_model import BaseEmbedding
from typing import Optional, Tuple, cast
import torch.nn.functional as F


class LightningBigram(pl.LightningModule):
    def __init__(self, embedding_model: BaseEmbedding, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_model"])
        self.learning_rate = learning_rate

        self.embedding = embedding_model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.embedding(idx))

    def generate(
        self,
        idx: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:

        idx = idx or torch.zeros((1, 1), dtype=torch.long)
        new_idx = idx.to(self.embedding.device)
        for _ in range(max_new_tokens):
            # execute on the last idx
            logits = self(new_idx)
            # calculate probabilities on the last logits
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)  # dims -> B, C
            idx_next = torch.multinomial(probs, num_samples=1)  # get the next token
            new_idx = torch.cat(
                [new_idx, idx_next], dim=1
            )  # attach the prediction to the corpus

        return new_idx

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, targets = batch
        logits = self(x)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(
            logits,
            targets,
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, targets = batch
        logits = self(x)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(
            logits,
            targets,
        )
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
