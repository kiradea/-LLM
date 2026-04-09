import torch
import torch.nn as nn
from embedding import Embedding
from transformer_block import TransformerBlock

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_dim, n_layers):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, hidden_dim) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)  # 输出每个token概率
        print(f"   [Model Init] embed: {self.embed}")
        print(f"   [Model Init] blocks: {self.blocks}")
        print(f"   [Model Init] head: {self.head}")

    def forward(self, x):
        print(f"   [Model] 输入 Token IDs: {x}")
        x = self.embed(x)
        print(f"   [Model] Embedding 后的形状: {x.shape}")
        
        for i, block in enumerate(self.blocks):
            print(f"   [Model] 进入第 {i} 个 Transformer Block...")
            x = block(x)
            
        logits = self.head(x)
        print(f"   [Model] 输出 Logits 形状: {logits.shape}")
        return logits
