import torch
import torch.nn as nn
import torch.functional as F

class FFN(nn.Module):
    def __init__(self, in_features=512, hid_features=2048, out_features=512, proj_dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hid_features, bias=True)
        self.linear2 = nn.Linear(hid_features, out_features, bias=True)
        self.dropout = nn.Dropout(proj_dropout)
        
    def forward(self, x):
        return self.dropout(self.linear2(torch.relu(self.linear1(x))))