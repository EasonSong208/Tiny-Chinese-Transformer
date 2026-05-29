# 在这个项目中，我将尝试
# 训练一个能生成短文本的小型 GPT-like Transformer
# 本py是第一阶段，那么现在开始

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



## 提前设定下参数量，不能太大
n_embd = 256
n_heads = 4
n_layers = 4
block_size = 128
batch_size = 32
learning_rate = 3e-4
max_iters = 5000
# 这个大小是为了让模型能在我的GPU上训练，太大了就不行了
# 256维的嵌入，4个头，4层，128的块大小，32的批量大小，3e-4的学习率，5000次迭代

## 先定义一个简单的Transformer模型
class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * n_embd
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_heads, hidden_dim) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        token_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = token_emb + pos_emb  # (B, T, n_embd)
        for layer in self.layers:
            x = layer(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
    
# 这个模型包含了一个词嵌入层，一个位置嵌入层，多个Transformer块，
# 一个层归一化层和一个线性层来输出词汇表大小的logits。

# 目标是在 4060 上训练一个能生成短文本的小型 GPT-like Transformer，想要在这一阶段实现这些类：
# class TokenEmbedding(nn.Module) 
# class PositionalEncoding(nn.Module) 
# class MultiHeadSelfAttention(nn.Module) 
# class FeedForward(nn.Module) 
# class TransformerBlock(nn.Module) 
# class TransformerLM(nn.Module)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx):
        return self.embedding(idx)  # (B, T, n_embd)
    # 这个类实现了一个词嵌入层，输入是一个索引张量，输出是对应的嵌入向量。

class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, n_embd)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]  # (1, T, n_embd)
    # 这个类实现了位置编码，使用正弦和余弦函数来生成位置编码向量，并将其注册为缓冲区。

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3 * n_embd)
        qkv = qkv.view(B, T, self.n_heads, 3 * self.head_dim).permute(0, 2, 1, 3)  # (B, n_heads, T, head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # 各自的形状都是 (B, n_heads, T, head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, n_heads, T, T)

        attn_output = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, C)  # (B, T, n_embd)
        output = self.out_proj(attn_output)  # (B, T, n_embd)
        return output
    # 这个类实现了多头自注意力机制，输入是一个张量，输出是经过注意力计算后的张量。

class FeedForward(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embd)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # (B, T, hidden_dim)
        x = self.fc2(x)  # (B, T, n_embd)
        return x
    # 这个类实现了一个前馈神经网络，包含两个线性层和一个ReLU激活函数。

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * n_embd
        self.attn = MultiHeadSelfAttention(n_embd, n_heads)
        self.ffn = FeedForward(n_embd, hidden_dim)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_output = self.attn(self.ln1(x))  # (B, T, n_embd)
        x = x + attn_output  # 残差连接
        ffn_output = self.ffn(self.ln2(x))  # (B, T, n_embd)
        x = x + ffn_output  # 残差连接
        return x
    # 这个类实现了一个Transformer块，包含一个多头自注意力层和一个前馈神经网络，并使用层归一化和残差连接。

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * n_embd
        self.token_embedding = TokenEmbedding(vocab_size, n_embd)
        self.position_embedding = PositionalEncoding(n_embd)
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_heads, hidden_dim) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        token_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(token_emb)  # (1, T, n_embd)
        x = token_emb + pos_emb  # (B, T, n_embd)
        for layer in self.layers:
            x = layer(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
    # 这个类实现了一个Transformer语言模型，包含词嵌入、位置编码、多个Transformer块、层归一化和输出层。

# 一个transfomer的本质是一个由自注意力机制和前馈神经网络形成的结构
# 它有编码器，解码器，和编码器-解码器结构
# 这里我们实现的是一个语言模型，所以我们只需要解码器部分
# 这个模型可以用来生成文本，输入是一个索引序列，输出是下一个词的概率分布。


# 现在我们已经定义了这些类，下一步就是准备数据，训练模型，并评估它的性能。

# 训练一个能生成短文本的小型 GPT-like Transformer 的第一阶段已经完成了，我们定义了模型的结构和组件。

# 我们再做一个训练的文件夹，里面放入数据和训练代码，来训练这个模型。
