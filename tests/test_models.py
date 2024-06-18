import pytest
import lightning as pl
from torch.utils.data import DataLoader
from torch import Tensor
from mini_transformers.data_load import ShakespeareDataset, Vocabulary
from mini_transformers.models.embedding_model import (
    BaseEmbedding,
    SimpleEmbedding,
    PositionHeadEmbedding,
    HeadEmbedding,
    MultiHeadedAttentionEmbedding,
    ResidualBlockAttentionEmbedding,
    GPT,
)
from mini_transformers.models.litbigram import LightningBigram
import torch

CONTEXT_LEN = 8
BATCH_SIZE = 64


@pytest.fixture(scope="module")
def train_val_dataloaders(
    request: pytest.FixtureRequest,
) -> tuple[Vocabulary, DataLoader[ShakespeareDataset], DataLoader[ShakespeareDataset]]:

    dataset = ShakespeareDataset(context_lenght=CONTEXT_LEN)
    vocabulary = dataset.vocabulary
    train_ds, valid_ds = dataset.train_valid_subsets()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )
    # train_loader.dataset.vocabulary
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=5,
        persistent_workers=True,
    )
    return (vocabulary, train_loader, valid_loader)


@pytest.mark.slow
def test_roll_train_dataset(
    train_val_dataloaders: tuple[
        Vocabulary, DataLoader[ShakespeareDataset], DataLoader[ShakespeareDataset]
    ]
) -> None:
    vocabulary, train_loader, val_loader = train_val_dataloaders

    for X, y in iter(train_loader):
        assert X.size(0) <= BATCH_SIZE
        assert X.size(1) == CONTEXT_LEN
        assert y.size(0) <= BATCH_SIZE
        assert y.size(1) == CONTEXT_LEN

    for _, _ in iter(val_loader):
        assert X.size(0) <= BATCH_SIZE
        assert X.size(1) == CONTEXT_LEN
        assert y.size(0) <= BATCH_SIZE
        assert y.size(1) == CONTEXT_LEN


def test_base_embedding_error() -> None:
    embedding = BaseEmbedding(10)
    with pytest.raises(NotImplementedError):
        embedding(torch.tensor([1, 2, 3]))


@pytest.mark.slow
def test_simple_embedding_training(
    train_val_dataloaders: tuple[
        Vocabulary, DataLoader[ShakespeareDataset], DataLoader[ShakespeareDataset]
    ]
) -> None:
    vocabulary, train_loader, val_loader = train_val_dataloaders

    bigram = LightningBigram(SimpleEmbedding(len(vocabulary)), 1e-3)
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=1,
        callbacks=[],
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(
        model=bigram, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


@pytest.mark.parametrize("topk", [None, 5], ids=lambda d: f"generate_topk_{d}")
def test_generate_simple_embedding(
    train_val_dataloaders: tuple[
        Vocabulary, DataLoader[ShakespeareDataset], DataLoader[ShakespeareDataset]
    ],
    topk: int,
) -> None:

    vocabulary, train_loader, val_loader = train_val_dataloaders
    bigram = LightningBigram(SimpleEmbedding(len(vocabulary)), 1e-3)
    generated = bigram.generate(torch.tensor([0]), top_k=topk)
    vocabulary.decode(generated.squeeze().tolist())


@pytest.mark.slow
@pytest.mark.parametrize(
    "embedding_type",
    [
        (
            HeadEmbedding.__name__,
            lambda vocab_size: HeadEmbedding(vocab_size, n_embeds=100),
        ),
        (
            PositionHeadEmbedding.__name__,
            lambda vocab_size: PositionHeadEmbedding(
                vocab_size, n_embeds=100, context_len=CONTEXT_LEN
            ),
        ),
        (
            MultiHeadedAttentionEmbedding.__name__,
            lambda vocab_size: MultiHeadedAttentionEmbedding(
                vocab_size,
                n_embeds=100,
                n_heads=4,
                head_size=10,
                context_len=CONTEXT_LEN,
            ),
        ),
        (
            ResidualBlockAttentionEmbedding.__name__,
            lambda vocab_size: ResidualBlockAttentionEmbedding(
                vocab_size, 2, context_len=CONTEXT_LEN, n_embeds=100, n_heads=4
            ),
        ),
        (
            GPT.__name__,
            lambda vocab_size: GPT(
                vocab_size,
                n_embeds=100,
                n_heads=4,
                n_layers=2,
                context_len=CONTEXT_LEN,
            ),
        ),
    ],
    ids=lambda e: f"{e[0]}",
)
def test_models_bigram(
    train_val_dataloaders: tuple[
        Vocabulary, DataLoader[ShakespeareDataset], DataLoader[ShakespeareDataset]
    ],
    embedding_type: BaseEmbedding,
) -> None:
    vocabulary, train_loader, val_loader = train_val_dataloaders
    embedding_model = embedding_type[1](len(vocabulary))
    bigram = LightningBigram(embedding_model, 1e-3)
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=1,
        overfit_batches=1,
        callbacks=[],
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(
        model=bigram, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
