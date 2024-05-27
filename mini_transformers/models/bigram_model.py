import torch
from torch import nn
from torch.nn import functional as F
from mini_transformers.models.embedding_model import SimpleEmbedding, HeadEmbedding, BaseEmbedding

class BigramModel(nn.Module):
    def __init__(self, embedding_model: BaseEmbedding) -> None:
        super().__init__()
        
        self.embedding_model = embedding_model
        
    def forward(self, idx, targets=None):
        
        logits = self.embedding_model(idx)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, )
        
        return logits, loss
            
        
    def generate(self, idx=None, max_new_tokens: int = 100):
        
        idx = idx or torch.zeros((1, 1), dtype=torch.long)
        new_idx = idx
        for _ in range(max_new_tokens):
            # execute on the last idx
            logits, _ = self(new_idx)
            # calculate probabilities on the last logits
            probs = F.softmax(logits[:, -1, :], dim=-1) # dims -> B, C
            
            idx_next = torch.multinomial(probs, num_samples=1) # get the next token
            
            new_idx = torch.cat([new_idx, idx_next], dim=1) # attach the prediction to the corpus
            
        return new_idx