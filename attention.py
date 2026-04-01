import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        print(f"         [Attention] Q/K/V 矩阵形状: {Q.shape}")
        
        # attention score
        scores = Q @ K.transpose(-2, -1) / (self.d_model ** 0.5)
        print(f"         [Attention] 注意力分数矩阵形状 (seq_len x seq_len): {scores.shape}")
        
        attn = F.softmax(scores, dim=-1)
        # 简单打印注意力权重
        print(f"         [Attention] 第一行注意力权重: {attn[0, 0, :].tolist()}")
        
        out = attn @ V
        print(f"         [Attention] 输出形状: {out.shape}")
        return out