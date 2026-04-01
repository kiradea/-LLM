import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        print(f"      [Embedding] 初始化: vocab_size={vocab_size}, d_model={d_model}")


    def forward(self, x):
        out = self.embedding(x)
        print(f"      [Embedding] 输入 IDs: {x.tolist()}")
        print(f"      [Embedding] 输出形状: {out.shape}")
        return out  # [batch_size, seq_len, d_model]