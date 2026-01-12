import torch
import torch.nn as nn
from modelling.encoder import BaseTransformerLayer
from modelling.decoder import TransformerDecoderLayer
from modelling.embedding import EmbeddingLayer
from modelling.positionalencoding import PositionalEncoding
from modelling.ffn import FFN
from typing import Optional, List

class Transformer(nn.Module):
    def __init__(self, vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 max_len: int = 512,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            BaseTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        # Output projection layer to convert d_model to vocab_size
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Scale embeddings by sqrt(d_model) and add positional encoding
        x = self.embedding(src) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=src_mask)
        return x
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Scale embeddings by sqrt(d_model) and add positional encoding
        x = self.embedding(tgt) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, encoder_attention_mask=memory_mask, attention_mask=tgt_mask)
        return x
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, memory_mask)
        # Project to vocabulary size to get logits
        logits = self.output_projection(output)
        return logits