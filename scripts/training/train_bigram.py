#!/usr/bin/env python
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from importlib import resources
from torch.utils.data import DataLoader
import mini_transformers
from mini_transformers.models.litbigram import LightningBigram
from mini_transformers.models.embedding_model import GPT
from mini_transformers.data_load import ShakespeareDataset


CONTEXT_LEN = 256
BATCH_SIZE = 64
EPOCHS = 5


dataset = ShakespeareDataset(context_lenght=CONTEXT_LEN)
vocabulary = dataset.vocabulary
train_ds, valid_ds = dataset.train_valid_subsets()
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=5,
    persistent_workers=True,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=5,
    persistent_workers=True,
)

gpt = GPT(
    vocab_size=len(vocabulary),
    n_layers=6,
    n_embeds=384,
    n_heads=6,
    context_len=CONTEXT_LEN,
    dropout=0.2,
)

bigram = LightningBigram(embedding_model=gpt, learning_rate=1e-3)

with resources.path(mini_transformers, "logs") as log_path:
    logger = TensorBoardLogger(
        log_path,
        name="generative_bigram_model",
    )

with resources.path(mini_transformers, "checkpoints") as checkpoint_path:
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        every_n_epochs=1,
        save_top_k=2,
        monitor="validation_loss",
        filename="minigpt-epoch{epoch:02d}-val_loss{validation_loss:.2f}",
        auto_insert_metric_name=False,
    )

trainer = pl.Trainer(
    min_epochs=1,
    max_epochs=EPOCHS,
    overfit_batches=0.001,
    callbacks=[checkpoint_callback],
    logger=logger,
    log_every_n_steps=10,
    accelerator="mps",
    devices=1,
)

if __name__ == "__main__":

    trainer.fit(
        model=bigram, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
