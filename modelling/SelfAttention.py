import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self,  
                 mask_future: bool = True,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.mask_future = mask_future
        self.attn_drop = nn.Dropout(attn_dropout)
    
    def forward(self, query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                attention_mask: bool = True ) -> torch.Tensor: # 1=keep, 0=mask; shape [B, Tk]
        # x: [B, seq_len, emb]
        B, Tq, emb = query.shape
        Tk = key.shape[1]
        
        #Q = self.q(query)  # [B, seq_len, emb]
        #K = self.k(key)
        #V = self.v(value)
        
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(emb)  # [B, seq_len, seq_len]
        
        # causal mask to prevent attending to future tokens
        if self.mask_future and Tq == Tk:
            mask = torch.triu(
                torch.ones(Tq, Tk, device=query.device, dtype=torch.bool), 
                diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
        
        # Key padding mask: attention_mask [B, Tk], 1=keep, 0=mask
        if attention_mask is not None:
            key_pad = (attention_mask == 0).to(torch.bool)  # [B, Tk]
            key_pad = key_pad.unsqueeze(1)                  # [B, 1, Tk]
            scores = scores.masked_fill(key_pad, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ value  # [B, seq_len, emb]
        return out
