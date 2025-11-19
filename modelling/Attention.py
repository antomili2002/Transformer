import torch
import torch.nn as nn
import math

class Attention(nn.Module):
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

class MultiHeadAttention(nn.Module):
    def __init__(self,  
                 emb: int,
                 num_heads: int,
                 mask_future: bool = True,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0):
        super().__init__()
        assert emb % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.emb = emb
        self.num_head = num_heads
        self.mask_future = mask_future
        self.head_dim = emb // num_heads
        
        self.query_transform = nn.Linear(emb, emb, bias=False)
        self.key_transform = nn.Linear(emb, emb, bias=False)
        self.value_transform = nn.Linear(emb, emb, bias=False)
        self.output_transform = nn.Linear(emb, emb, bias=False)
        
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.proj_dropout = nn.Dropout(p=proj_dropout)
        
    def forward(self, 
                query: torch.Tensor, 
                keys: torch.Tensor, 
                values: torch.Tensor, 
                attention_mask: bool = True,
                future_mask: bool = True) -> torch.Tensor:
        B, Tq, _ =  query.shape
        _, Tk, _ = keys.shape
        
        query = self.query_transform(query)      # [B, seq_len, d_model]
        keys = self.key_transform(keys)        # [B, seq_len, d_model]
        values = self.value_transform(values)    # [B, seq_len, d_model]
        
        # split into head dimensions
        query = query.view(B, Tq, self.num_head, self.head_dim)
        query = query.permute(0, 2, 1, 3) # [B, h, T, d_heads]
        
        keys = keys.view(B, Tk, self.num_head, self.head_dim)
        keys = keys.permute(0, 2, 1, 3) # [B, h, T, d_heads]
        
        values = values.view(B, Tk, self.num_head, self.head_dim)
        values = values.permute(0, 2, 1, 3) # [B, h, T, d_heads]
        
        # compute score
        # Q: [B, h, T, d_heads] K.T: [B, h, d_heads, T]
        scores = (query @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # causal mask to mask out future tokens
        if self.mask_future and Tq == Tk:
            mask = torch.triu(
                torch.ones(Tq, Tk, device=query.device, dtype=torch.bool), 
                diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
        
        # Key padding mask: attention_mask [B, Tk], 1=keep, 0=mask
        if attention_mask is not None:
            key_pad = (attention_mask == 0).to(torch.bool)  # [B, Tk]
            key_pad = key_pad.unsqueeze(1).unsqueeze(1)     # [B, 1, 1, Tk]
            scores = scores.masked_fill(key_pad, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # attn: [B, h, Tq, Tk], values: [B, h, Tk, d_k]
        out_heads = attn @ values  # out_heads: [B, h, Tq, d_k]
        
        # now merge heads
        out_heads = out_heads.permute(0, 2, 1, 3) # [B, Tq, h, d_k] 
        # [B, Tq, h*d_k] where h*d_k = emb
        out = out_heads.contiguous().view(B, Tq, self.emb)
        
        out = self.output_transform(out)
        out = self.proj_dropout(out)
        return out