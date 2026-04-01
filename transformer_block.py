import torch
import torch.nn as nn
from attention import SelfAttention
from residual import ResidualLayerNorm
from ffn import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.attn = SelfAttention(d_model)
        self.res_attn = ResidualLayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim)
        self.res_ffn = ResidualLayerNorm(d_model)

    def forward(self, x):
        print(f"      [Block] 输入形状: {x.shape}")
        x = self.res_attn(x, self.attn)
        print(f"      [Block] Self-Attention + Residual 后的形状: {x.shape}")
        x = self.res_ffn(x, self.ffn)
        print(f"      [Block] FFN + Residual 后的形状: {x.shape}")
        return x