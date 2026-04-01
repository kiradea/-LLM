# tokenizer.py
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {ch:i for i,ch in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        self.inv_vocab = {i:ch for ch,i in self.vocab.items()}
        print(f"[Tokenizer] 初始化完成，词表大小: {len(self.vocab)}")

    def encode(self, text):
        print(f"[Tokenizer] 正在编码文本: '{text}'")
        res = [self.vocab[ch] for ch in text.lower() if ch in self.vocab]
        print(f"[Tokenizer] 编码结果: {res}")
        return res

    def decode(self, tokens):
        print(f"[Tokenizer] 正在解码 Tokens: {tokens}")
        res = "".join([self.inv_vocab[t] for t in tokens])
        print(f"[Tokenizer] 解码结果: '{res}'")
        return res
    