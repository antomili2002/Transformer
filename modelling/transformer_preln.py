import torch
import torch.nn as nn
from typing import Optional
from modelling.Attention import MultiHeadAttention
from modelling.ffn import FFN
from modelling.embedding import EmbeddingLayer
from modelling.positionalencoding import PositionalEncoding


class PreLNTransformerEncoderLayer(nn.Module):
    """Encoder layer with Pre-LayerNorm"""
    def __init__(self, input_dim=512, num_heads=8, feature_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            input_dim,
            num_heads,
            mask_future=False,
            attn_dropout=dropout,
            proj_dropout=dropout
        )
        self.feature_transformation = FFN(input_dim, feature_dim, input_dim, dropout)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

    def forward(self, input: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        x = input

        # Pre-LN: LayerNorm BEFORE self-attention
        norm_x = self.layer_norm_1(x)
        attn_out = self.self_attention(
            query=norm_x,
            keys=norm_x,
            values=norm_x,
            attention_mask=attention_mask,
            future_mask=False,
        )
        x = x + attn_out

        # Pre-LN: LayerNorm BEFORE FFN
        norm_x = self.layer_norm_2(x)
        ffn_out = self.feature_transformation(norm_x)
        x = x + ffn_out 

        return x


class PreLNTransformerDecoderLayer(nn.Module):
    """Decoder layer with Pre-LayerNorm"""
    def __init__(self, input_dim=512, num_heads=8, feature_dim=2048, dropout=0.1):
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

        # Pre-LN: LayerNorm BEFORE self-attention
        norm_x = self.layer_norm_1(x)
        attn_out = self.self_attention(
            query=norm_x,
            keys=norm_x,
            values=norm_x,
            attention_mask=attention_mask,
            future_mask=True,
        )
        x = x + attn_out 

        # Pre-LN: LayerNorm BEFORE cross-attention
        norm_x = self.layer_norm_2(x)
        cross_attn_out = self.encoder_attention(
            query=norm_x,
            keys=encoder,
            values=encoder,
            attention_mask=encoder_attention_mask,
            future_mask=False,
        )
        x = x + cross_attn_out

        # Pre-LN: LayerNorm BEFORE FFN
        norm_x = self.layer_norm_3(x)
        ffn_out = self.feature_transformation(norm_x)
        x = x + ffn_out  

        return x


class PreLNTransformer(nn.Module):
    """Complete Transformer with Pre-LayerNorm architecture"""
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
            PreLNTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_final_norm = nn.LayerNorm(d_model)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            PreLNTransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_final_norm = nn.LayerNorm(d_model)

        self.output_projection = nn.Linear(d_model, vocab_size)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(src) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, attention_mask=src_mask)

        # PRE-LN!
        x = self.encoder_final_norm(x)
        return x

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(tgt) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, memory, encoder_attention_mask=memory_mask, attention_mask=tgt_mask)

        # PRE-LN!
        x = self.decoder_final_norm(x)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        memory = self.encode(src, src_mask)

        output = self.decode(tgt, memory, tgt_mask, memory_mask)

        logits = self.output_projection(output)
        return logits
