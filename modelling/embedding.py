import torch
from torch import nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, padding_idx=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.embed_dim = embed_size  # Store for easy access

    def forward(self, x):
        return self.embedding(x)