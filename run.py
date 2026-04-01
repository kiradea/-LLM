import torch
from tokenizer import SimpleTokenizer
from model import MiniTransformer

tokenizer = SimpleTokenizer()
vocab_size = len(tokenizer.vocab)
print(f"字典大小: {vocab_size}")

# 初始化模型
model = MiniTransformer(vocab_size, d_model=32, hidden_dim=32, n_layers=1) # 减为1层方便观察
model.eval() # 开启评估模式

text = input("请输入一段文字 (仅限英文和空格): ")
print(f"\n1. 原始输入: '{text}'")

# 1. Tokenization
encoded_tokens = tokenizer.encode(text)
tokens = torch.tensor([encoded_tokens], dtype=torch.long)
print(f"2. Token 编码 (IDs): {encoded_tokens}")
print(f"3. 输入 Tensor 形状: {tokens.shape} (batch_size, seq_len)")

# 2. Model Forward
print("\n--- 进入模型内部数据流 ---")
with torch.no_grad():
    logits = model(tokens)
print("--- 离开模型内部数据流 ---\n")

# 3. Output
print(f"4. Logits 形状: {logits.shape} (batch_size, seq_len, vocab_size)")
pred_token = torch.argmax(logits, dim=-1)
print(f"5. 预测的 Token IDs: {pred_token[0].tolist()}")

decoded_text = tokenizer.decode(pred_token[0].tolist())
print(f"6. 解码预测结果: '{decoded_text}'")

# 说明：模型对输入序列的每一个 token 都会预测其下一个 token
# 如果输入 "hi"，则第一个字符对应 'h' 的预测，第二个对应 'i' 的预测
last_pred_id = pred_token[0, -1].item()
print(f"\n结论：根据最后一个字符 '{text[-1]}', 模型预测的下一个字符是: '{tokenizer.decode([last_pred_id])}'")