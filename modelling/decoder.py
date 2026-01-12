import torch
import torch.nn as nn
from typing import Optional
from modelling.Attention import MultiHeadAttention
from modelling.ffn import FFN

class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim=512, num_heads=2, feature_dim=6, dropout=0.0):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            input_dim,
            num_heads,
            mask_future=True,
            attn_dropout=dropout,
            proj_dropout=dropout
        )
        self.encoder_attention = MultiHeadAttention(
            input_dim,
            num_heads=num_heads,
            mask_future=False,
            attn_dropout=dropout,
            proj_dropout=dropout
        )
        self.feature_transformation = FFN(input_dim, feature_dim, input_dim, dropout)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)
        
    def forward(self, input: torch.Tensor,
                encoder: torch.Tensor, 
                encoder_attention_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None):
        x = input
        #print("Decoder input shape:", x.shape)
        #print("Encoder output shape:", encoder.shape)
        #print("Attention mask shape:", attention_mask.shape if attention_mask is not None else None)
        #print("Encoder attention mask shape:", encoder_attention_mask.shape if encoder_attention_mask is not None else None)
        
        
        tgt_mask = encoder_attention_mask
        src_mask = attention_mask
        
        attn_out = self.self_attention(
            query = x,
            keys = x,
            values = x,
            attention_mask = src_mask,
            future_mask = True,
        )
        x = x + attn_out
        x = self.layer_norm_1(x)
        
        cross_attn_out = self.encoder_attention(
            query = x,
            keys = encoder,
            values = encoder,
            attention_mask = tgt_mask,
            future_mask = False,
        )
        x = x + cross_attn_out
        x = self.layer_norm_2(x)
        
        ffn_out = self.feature_transformation(x)
        x = x + ffn_out
        x = self.layer_norm_3(x)
        
        return x