from typing import Any, List, Optional, Tuple
from torch.utils.data import Dataset, random_split, Subset
from importlib import resources
import mini_transformers
import torch
from enum import auto, Enum
from dataclasses import dataclass, field

class TextDatasets(Enum):
    
    def _generate_next_value_(name, start, count, last_values):
         return name + '.txt'
    
    shakespeare = auto()


def get_text_file(filename: str):
    
    with resources.path(mini_transformers, 'data') as data_path:
        data_file = data_path / filename
        return data_file
    
    
def load_text_data(filepath: str):
    
    with open(filepath, 'r', encoding='utf-8') as data_file:
        text = data_file.read()
        
    return text


class Vocabulary():
    def __init__(self, text: str) -> None:
        self.vocabulary = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.vocabulary)}
        self.itos = {i: c for i, c in enumerate(self.vocabulary)}
        
    def __len__(self):
        return len(self.vocabulary)
        
    def encode(self, text: str) -> List: 
        return [self.stoi.get(c) for c in text]
    
    def decode(self, code: List) -> str: 
        return ''.join([self.itos.get(c) for c in code])
    

class ShakespeareDataset(Dataset):
    def __init__(self, context_lenght=8, train_split: float = .9) -> None:
        super().__init__()
        self.filepath = get_text_file(TextDatasets.shakespeare.value)
        self.datatext = load_text_data(self.filepath)
        self.context_length = context_lenght
        
        self.vocabulary = Vocabulary(self.datatext)
        # self.vocabulary = sorted(list(set(self.datatext)))
        # self.stoi = {c: i for i, c in enumerate(self.vocabulary)}
        # self.itos = {i: c for i, c in enumerate(self.vocabulary)}
        
        self.data = torch.tensor(self.vocabulary.encode(text=self.datatext), dtype=torch.long)
        self.train_split = train_split
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        
        
        x = self.data[index: index + self.context_length]
        y = self.data[index+1: index + self.context_length +1]
        
        # if x.ndimension() == 0:
        #     x = torch.tensor([x])
        #     y = torch.tensor([y])
        
        if y.size(0) < self.context_length:
            x = torch.hstack([x, torch.zeros((self.context_length - x.size(0)), dtype=torch.long)])
            y = torch.hstack([y, torch.zeros((self.context_length - y.size(0)), dtype=torch.long)])
            
        
        return x, y
    # def encode(self, text) -> List: 
    #     return [self.stoi.get(c) for c in text]
    
    # def decode(self, code) -> str: 
    #     return ''.join([self.itos.get(c) for c in code])
    
    def train_valid_subsets(self, train_split: Optional[float]=None) -> Tuple[Subset, Subset]:
        train_split = (train_split or self.train_split)
        train_len = int(train_split * len(self))
        return random_split(self, [train_len, len(self) - train_len])