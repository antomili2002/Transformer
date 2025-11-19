import torch
import torch.nn as nn
from typing import Optional
from modelling.Attention import MultiHeadAttention
from modelling.embedding import EmbeddingLayer
from modelling.positionalencoding import PositionalEncoding
from modelling.ffn import FFN

class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim=512, num_heads=2, feature_dim=6, dropout=0.0):
        super().__init__()
        self.self_attention = MultiHeadAttention(input_dim, 
                                            num_heads, 
                                            mask_future=False, 
                                            attn_dropout=dropout, 
                                            proj_dropout=dropout)
        self.feature_transformation = FFN(input_dim, feature_dim, input_dim, dropout)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x is input sequence already embedded and with positional encoding
        # x: [B, seq_len, d_model]
        # attention_mask: [B, seq_len] 1=Keep 0=pad
        attn_out = self.self_attention(
            query = x,
            keys = x,
            values = x,
            attention_mask = attention_mask,
            future_mask = False,
        )
        x = x + attn_out
        x = self.layer_norm_1(x)
        
        ffn_out = self.feature_transformation(x)
        x = x + ffn_out
        x = self.layer_norm_2(x)
        
        return x