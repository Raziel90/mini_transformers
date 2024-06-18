from typing import Any, List, Optional, Tuple, Union
from torch.utils.data import Dataset, random_split, Subset
from importlib import resources
import mini_transformers
import torch
from enum import Enum
from pathlib import Path


class TextDatasets(Enum):

    shakespeare = "shakespeare.txt"


def get_text_file(filename: Union[TextDatasets, str]) -> Path:
    filename = filename.value if isinstance(filename, TextDatasets) else filename
    with resources.path(mini_transformers, "data") as data_path:
        data_file = data_path / str(filename)
        return data_file


def load_text_data(filepath: Union[str, Path]) -> str:

    with open(filepath, "r", encoding="utf-8") as data_file:
        text = data_file.read()

    return text


class Vocabulary:
    def __init__(self, text: str) -> None:
        self.vocabulary = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.vocabulary)}
        self.itos = {i: c for i, c in enumerate(self.vocabulary)}

    def __len__(self) -> int:
        return len(self.vocabulary)

    def encode(self, text: str) -> List[Union[int, None]]:
        return [self.stoi.get(c) for c in text]

    def decode(self, code: List[int]) -> str:
        return "".join([self.itos.get(c, "") for c in code])


class ShakespeareDataset(Dataset["ShakespeareDataset"]):
    def __init__(self, context_lenght: int = 8, train_split: float = 0.9) -> None:
        super().__init__()
        self.filepath = get_text_file(TextDatasets.shakespeare)
        self.datatext = load_text_data(self.filepath)
        self.context_length = context_lenght

        self.vocabulary = Vocabulary(self.datatext)

        self.data = torch.tensor(
            self.vocabulary.encode(text=self.datatext), dtype=torch.long
        )
        self.train_split = train_split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:

        x = self.data[index : index + self.context_length]
        y = self.data[index + 1 : index + self.context_length + 1]
        if y.size(0) < self.context_length:
            x = torch.hstack(
                [
                    x,
                    torch.zeros(
                        (self.context_length - x.size(0)),
                        dtype=torch.long,
                        device=x.device,
                    ),
                ]
            )
            y = torch.hstack(
                [
                    y,
                    torch.zeros(
                        (self.context_length - y.size(0)),
                        dtype=torch.long,
                        device=y.device,
                    ),
                ]
            )
        return x, y

    def train_valid_subsets(
        self, train_split: Optional[float] = None
    ) -> Tuple[Subset["ShakespeareDataset"], Subset["ShakespeareDataset"]]:
        train_split = train_split or self.train_split
        train_len = int(train_split * len(self.data))
        training_set, validation_set = random_split(
            self, [train_len, len(self.data) - train_len]
        )
        return training_set, validation_set
