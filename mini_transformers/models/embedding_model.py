from torch import nn
from torchtyping import TensorType
from mini_transformers.models.components import MultiHeadAttention, ResidualBlock

import torch


class BaseEmbedding(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def forward(self, idx: TensorType):
        raise NotImplementedError()
    

class SimpleEmbedding(BaseEmbedding):
    
    def __init__(self, vocab_size: int) -> None:
        super().__init__(vocab_size)
        
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx: TensorType["batch", "time", torch.long]) -> TensorType["batch", "time", "channels", torch.float32]:
        
        logits = self.token_embedding_table(idx)
        
        return logits
     
    
class HeadEmbedding(BaseEmbedding):
    
    def __init__(self, vocab_size: int, n_embeds: int) -> None:
        super().__init__(vocab_size)
        # self.vocab_size = vocab_size
        self.n_embeds = n_embeds
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)
        
    def forward(self, idx: TensorType["batch", "time", torch.long]) -> TensorType["batch", "time", "channels", torch.float32]:
        tok_embeddings = self.token_embedding_table(idx) # (B, T, vocab_size)
        logits = self.lm_head(tok_embeddings) # (B, T, vocab_size)
        return logits
    
    
    
class PositionHeadEmbedding(BaseEmbedding):
    
    def __init__(self, vocab_size: int, n_embeds: int, context_len: int) -> None:
        super().__init__(vocab_size)
        # self.vocab_size = vocab_size
        self.n_embeds = n_embeds
        self.context_len = context_len
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.position_embedding_table = nn.Embedding(context_len, n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)
        
    def forward(self, idx: TensorType["batch", "time", torch.long]) -> TensorType["batch", "time", "channels", torch.float32]:
        B, T = (idx_cond := idx[:, -self.context_len:]).shape
        tok_embeddings = self.token_embedding_table(idx_cond) # (B, T, n_embeds)
        pos_embeddings = self.position_embedding_table(torch.arange(0, T) % self.context_len) # (T, n_embeds)
        x = tok_embeddings + pos_embeddings # broadcast on the B dimension (B, T, n_embeds)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    
    
class MultiHeadedAttentionEmbedding(BaseEmbedding):
    
    def __init__(self, vocab_size: int, n_embeds: int, n_heads: int, head_size: int, context_len: int) -> None:
        super().__init__(vocab_size)
        # self.vocab_size = vocab_size
        self.n_embeds = n_embeds
        self.context_len = context_len
        self.sa_attention_head = MultiHeadAttention(context_len=context_len, n_embeds=n_embeds, n_heads=n_heads, head_size=head_size)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.position_embedding_table = nn.Embedding(context_len, n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)
        
    def forward(self, idx: TensorType["batch", "time", torch.long]) -> TensorType["batch", "time", "channels", torch.float32]:
        
        B, T = (idx_cond := idx[:, -self.context_len:]).shape
        tok_embeddings = self.token_embedding_table(idx_cond) # (B, T, n_embeds)
        pos_embeddings = self.position_embedding_table(torch.arange(0, T)) # (T, n_embeds)
        x = tok_embeddings + pos_embeddings # broadcast on the B dimension (B, T, n_embeds)
        x = self.sa_attention_head(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    
    
class ResidualBlockAttentionEmbedding(BaseEmbedding):
    def __init__(self, vocab_size: int) -> None:
        super().__init__(vocab_size)
        
        self.net = nn.Sequential([
            ResidualBlockAttentionEmbedding
        ])
        
    