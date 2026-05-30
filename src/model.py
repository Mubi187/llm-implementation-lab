import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # each token looks up the next token scores directly
        # shape: (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx     : (B, T) input token indices
        targets : (B, T) target token indices, optional
        """
        # logits = raw scores for next token, shape: (B, T, vocab_size)
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # cross_entropy expects (N, C) — flatten batch and time
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        idx : (B, T) context tokens
        Autoregressively generates max_new_tokens new tokens
        """
        for _ in range(max_new_tokens):
            # get predictions
            logits, _ = self(idx)

            # focus only on last time step → (B, C)
            logits = logits[:, -1, :]

            # softmax → probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append to sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx
