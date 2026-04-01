import torch
import torch.nn as nn

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        return self.norm(x + sublayer(x))