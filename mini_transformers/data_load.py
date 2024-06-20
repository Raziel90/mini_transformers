from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset, random_split, Subset
from torch import Tensor
from importlib import resources
import mini_transformers
import torch
from enum import Enum
from pathlib import Path


class TextDatasets(Enum):

    shakespeare = "shakespeare.txt"


def get_text_file(filename: Union[TextDatasets, str]) -> Path:
    """Loads file of name `filename` from from the dataset path

    Args:
        filename (Union[TextDatasets, str]): name of the file or dataset to load.

    Returns:
        Path: full path of the dataset file.
    """
    filename = filename.value if isinstance(filename, TextDatasets) else filename
    with resources.path(mini_transformers, "data") as data_path:
        data_file = data_path / str(filename)
        return data_file


def load_text_data(filepath: Union[str, Path]) -> str:
    """loads text data from a filepath

    Args:
        filepath (Union[str, Path]): filepath of the text file.

    Returns:
        str: text content of the file.
    """

    with open(filepath, "r", encoding="utf-8") as data_file:
        text = data_file.read()

    return text


class Vocabulary:
    """
    Vocabulary class to encode and decode text data
    """

    def __init__(self, text: str) -> None:
        """
        Args:
            text (str): text data to create vocabulary from.
        """

        self.vocabulary = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.vocabulary)}
        self.itos = {i: c for i, c in enumerate(self.vocabulary)}

    def __len__(self) -> int:
        return len(self.vocabulary)

    def encode(self, text: str) -> List[Union[int, None]]:
        return [self.stoi.get(c) for c in text]

    def decode(self, code: List[int]) -> str:
        return "".join([self.itos.get(c, "") for c in code])


class ShakespeareDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Shakespeare dataset class
    """

    def __init__(self, context_lenght: int = 8, train_split: float = 0.9) -> None:
        """Initialises the Shakespeare dataset with a specific context length and train split ratio.

        The dataset is loaded from a text file and then encoded into integer tokens using the
        Vocabulary class.

        Args:
            context_lenght (int, optional): _description_. Defaults to 8.
            train_split (float, optional): _description_. Defaults to 0.9.
        """
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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Returns a tuple of (x, y) where x is the input sequence and y is the target sequence.

        The input sequence is the context length long, and the target sequence is the same length
        as the context length. If the target sequence is shorter than the context length, it is
        padded with zeros.

        Args:
            index (int): index of the data sample to retrieve.

        Returns:
            Any: tuple of (x, y) where x is the input sequence and y is the target sequence.
        """

        x: Tensor = self.data[index : index + self.context_length]
        y: Tensor = self.data[index + 1 : index + self.context_length + 1]
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
        return (x, y)

    def train_valid_subsets(
        self, train_split: Optional[float] = None
    ) -> Tuple[Subset[Tuple[Tensor, Tensor]], Subset[Tuple[Tensor, Tensor]]]:
        train_split = train_split or self.train_split
        train_len = int(train_split * len(self.data))
        training_set, validation_set = random_split(
            self, [train_len, len(self.data) - train_len]
        )
        return training_set, validation_set
