import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, head_size, block_size):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.block_size = block_size
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size, dtype = torch.bool))
        )


    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.mask[:T, :T], float('-inf'))
        attn = torch.softmax(scores,dim=-1)
        return attn @ v



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.head_size = d_model // n_head
        self.heads = nn.ModuleList([SingleHeadAttention(d_model, self.head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(d_model, d_model)


    def forward(self, x):
        B, T, C = x.shape
        out = [h(x) for h in self.heads]
        out = torch.cat(out, dim=-1)
        out = self.proj(out)
        return out


class FNN(nn.Module):
    def __init__(self, d_model, o_model):
        super().__init__()
        self.d_model = d_model
        self.o_model = o_model
        self.fc1 = nn.Linear(d_model, o_model)
        self.fc2 = nn.Linear(o_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

class Block(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_head, block_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FNN(d_model, 4 * d_model)

    def forward(self, x):
        x = self.att(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size

        # ① 词嵌入：每个 token id → 一个 d_model 维向量
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # ② 位置嵌入：每个位置号 → 一个 d_model 维向量（可学习）
        self.pos_emb = nn.Embedding(block_size, d_model)

        # ③ N 个 Block 堆起来
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, block_size) for _ in range(n_layer)
        ])

        # ④ 最后的 LayerNorm（Pre-LN 结构必须补这一步）
        self.ln_f = nn.LayerNorm(d_model)

        # ⑤ 输出层：d_model → 词表大小
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ⑥ 初始化：GPT-2 风格（默认初始化在 d_model 大时会让 logits 爆炸）
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        # 词嵌入 + 位置嵌入，直接相加
        tok = self.token_emb(idx)                              # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=idx.device)) # (T, d_model)
        x = tok + pos                                          # (B, T, d_model)

        # 过所有 Block
        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                               # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """给定 prompt ids，逐 token 生成，返回 (B, T + max_new_tokens)。"""
        for _ in range(max_new_tokens):
            # ① 截断：只保留最近 block_size 个 token
            idx_cond = idx[:, -self.block_size:]

            # ② 前向，只要 logits
            logits, _ = self(idx_cond)

            # ③ 只取最后一个位置的预测
            logits = logits[:, -1, :] / temperature

            # ④ 可选：只保留分数最高的 top_k 个
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # ⑤ 变成概率，采样一个 token
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)

            # ⑥ 接到序列末尾
            idx = torch.cat([idx, nxt], dim=1)

        return idx
