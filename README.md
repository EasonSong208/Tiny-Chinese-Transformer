## 项目文件说明

本项目是一个从零实现的小型 GPT-like Transformer 中文字符级语言模型。项目目标不是训练一个可实用的大语言模型，而是完整体验 Transformer 语言模型从数据读取、模型构建、训练、保存 checkpoint 到文本生成的基本流程。

---

## 目录结构

本项目的主要结构如下：

```text
Tiny-Chinese-Transformer/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── __init__.py
├── DataLoader.py
├── TransformsFor4060.py
├── TrainingForTransformer.py
├── TrainingForTransformer_progress.py
├── TrainingForTransformer_epoch.py
├── generate.py
├── experiment.md
├── TrainingData/
│   └── README.md
└── checkpoints/
    └── README.md
```

其中，`TrainingData/` 用于存放训练文本，`checkpoints/` 用于保存训练得到的模型权重。

出于隐私和版权考虑，本仓库默认不提供原始训练语料，也不提供基于私人语料训练得到的模型权重。

---

## 1. `DataLoader.py`

`DataLoader.py` 负责把原始文本转换成模型可以训练的数据格式。

核心类是：

```python
class TextDataset(Dataset):
```

它主要完成以下工作：

1. 根据输入文本构建字符表 `vocab`；
2. 建立字符到编号的映射 `char_to_idx`；
3. 建立编号到字符的映射 `idx_to_char`；
4. 把整段文本转换成 token id 序列；
5. 每次返回一段长度为 `block_size` 的输入序列 `x`，以及对应右移一位的目标序列 `y`。

也就是说，模型学习的任务是：

```text
给定前面的若干字符，预测下一个字符。
```

例如：

```text
输入 x: 今 天 天 气 很
目标 y: 天 天 气 很 好
```

这就是最基本的自回归语言模型训练方式。

---

## 2. `TransformsFor4060.py`

`TransformsFor4060.py` 是模型结构文件，定义了小型 Transformer 语言模型的主体。

其中包括：

```python
class TokenEmbedding(nn.Module)
class PositionalEncoding(nn.Module)
class MultiHeadSelfAttention(nn.Module)
class FeedForward(nn.Module)
class TransformerBlock(nn.Module)
class TransformerLM(nn.Module)
class Transformer(nn.Module)
```

本项目使用的是 decoder-only 的 GPT-like 结构，主要包含：

1. token embedding；
2. position embedding / positional encoding；
3. masked multi-head self-attention；
4. feed forward network；
5. residual connection；
6. layer normalization；
7. linear language modeling head。

模型默认配置大致为：

```python
n_embd = 256
n_heads = 4
n_layers = 4
block_size = 128
batch_size = 32
learning_rate = 3e-4
```

这个规模是为了能在 RTX 4060 这类消费级显卡上完成训练。

---

## 训练脚本说明

本项目保留了三个版本的训练脚本，代表项目逐步改进的过程。

---

## 3. `TrainingForTransformer.py`

这是最早期的训练版本，主要用于验证整个训练流程是否能跑通。

它完成的事情包括：

1. 从 `TrainingData/` 文件夹读取所有 `.txt` 文件；
2. 合并训练文本；
3. 构建 `TextDataset` 和 `DataLoader`；
4. 初始化 Transformer 模型；
5. 使用 AdamW 和 CrossEntropyLoss 进行训练；
6. 训练结束后保存模型权重。

运行方式：

```bash
python -m Transformers.TrainingForTransformer
```

注意：这个版本中 `max_iters` 实际上控制的是完整 dataloader 训练轮次，而不是单个 batch step。因此如果训练文本较长，训练时间会非常久。这个版本主要作为早期实验记录保留，不推荐作为最终训练入口。

---

## 4. `TrainingForTransformer_progress.py`

这是第二版训练脚本，在第一版基础上加入了更清晰的训练进度显示和定期采样。

相比第一版，它主要改进了：

1. 修正项目根目录导入问题；
2. 使用 `tqdm` 显示训练进度条；
3. 按真实 step 控制训练，而不是误把完整 epoch 当作一次迭代；
4. 定期生成样本文本，方便观察模型训练效果；
5. 定期保存 checkpoint；
6. 增加梯度裁剪，提升训练稳定性。

运行方式：

```bash
python -m Transformers.TrainingForTransformer_progress
```

这个版本适合观察模型在训练过程中的变化，比如 loss 下降和生成文本逐渐从乱码变得通顺。

---

## 5. `TrainingForTransformer_epoch.py`

这是当前推荐使用的训练版本。

它在前两个版本基础上进一步整理为按 epoch 训练，并加入了验证集评估和最佳模型保存。

主要功能包括：

1. 自动读取 `TrainingData/` 下所有 `.txt` 文件；
2. 合并文本并构建字符级数据集；
3. 划分训练集和验证集；
4. 使用 DataLoader 批量训练；
5. 使用 AdamW 优化器；
6. 使用 CosineAnnealingLR 学习率调度；
7. 每个 epoch 后计算验证集 loss；
8. 自动保存验证集表现最好的模型；
9. 定期保存 checkpoint；
10. 定期生成样本文本。

推荐运行方式：

```bash
python -m Transformers.TrainingForTransformer_epoch
```

这是目前最稳定、最完整的训练入口。

---

## 6. `generate.py`

`generate.py` 是独立的推理脚本，用于加载训练好的模型并生成文本。

它会：

1. 读取 `TrainingData/` 中的文本，重建字符表；
2. 加载 `checkpoints/transformer_best.pth`；
3. 根据用户输入的 prompt 生成后续文本；
4. 支持 temperature 和 top-k 采样；
5. 支持单次生成和交互式生成。

基本运行方式：

```bash
python -m Transformers.generate --prompt "今天" --max_new_tokens 300
```

指定采样参数：

```bash
python -m Transformers.generate \
  --prompt "大学生活" \
  --max_new_tokens 300 \
  --temperature 0.8 \
  --top_k 40
```

进入交互模式：

```bash
python -m Transformers.generate --interactive
```

交互模式下可以连续输入多个 prompt，观察模型的不同生成结果。

---

## 如何从头体验本项目

### 1. 克隆项目

```bash
git clone https://github.com/EasonSong208/Tiny-Chinese-Transformer.git
cd Tiny-Chinese-Transformer
```

---

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果需要使用 GPU，请根据自己的 CUDA 版本安装对应的 PyTorch。

---

### 3. 准备训练数据

在项目中的 `TrainingData/` 文件夹下放入若干 `.txt` 文件：

```text
TrainingData/
├── text1.txt
├── text2.txt
└── text3.txt
```

本仓库不提供原始训练语料。使用者需要自行准备公开文本、自己有权使用的文本，或经过脱敏处理的文本。

---

### 4. 开始训练

推荐使用当前最完整版本：

```bash
python -m Transformers.TrainingForTransformer_epoch
```

训练过程中会显示：

1. 读取到的 `.txt` 文件数量；
2. 总字符数；
3. vocab size；
4. train / val 样本数量；
5. 每个 epoch 的 loss；
6. 验证集 loss；
7. 定期生成的样本文本；
8. checkpoint 保存位置。

训练得到的模型会保存在：

```text
checkpoints/
```

其中较重要的是：

```text
transformer_best.pth
transformer_model.pth
```

---

### 5. 生成文本

训练完成后，可以运行：

```bash
python -m Transformers.generate --prompt "今天" --max_new_tokens 300
```

也可以尝试：

```bash
python -m Transformers.generate --prompt "人工智能" --max_new_tokens 300
```

或者进入交互模式：

```bash
python -m Transformers.generate --interactive
```

---

## 推荐体验顺序

如果是第一次体验，建议按下面顺序：

```bash
# 1. 准备数据
mkdir -p TrainingData
# 将若干 .txt 文本放入 TrainingData/

# 2. 训练模型
python -m Transformers.TrainingForTransformer_epoch

# 3. 单次生成
python -m Transformers.generate --prompt "今天" --max_new_tokens 300

# 4. 调整采样参数
python -m Transformers.generate --prompt "大学生活" --temperature 0.8 --top_k 40 --max_new_tokens 300

# 5. 交互式体验
python -m Transformers.generate --interactive
```

---

## 关于生成效果的说明

这个模型是一个小型字符级语言模型，不是指令微调模型，也不是聊天机器人。

它的能力主要体现在：

1. 能学习训练语料中的局部语言模式；
2. 能生成一定程度通顺的中文文本；
3. 能复现训练语料中的部分风格；
4. 可以帮助理解 Transformer 语言模型的基本训练流程。

但它也有明显局限：

1. 数据规模较小时容易过拟合；
2. 容易记忆训练语料；
3. 长文本逻辑不稳定；
4. 对 prompt 的语义理解能力较弱；
5. 生成内容可能发生主题跳转；
6. 不适合用于真实生产环境。

因此，本项目更适合作为学习 Transformer、理解语言模型训练过程的小型实验项目。

---

## 注意事项

### 1. 不要上传私人训练语料

如果使用自己的课程论文、实验报告、笔记等作为训练数据，请不要将 `TrainingData/` 中的原始 `.txt` 文件上传到公开仓库。

---

### 2. 谨慎公开 checkpoint

如果模型是在私人语料上训练得到的，不建议公开 `.pth` 权重文件。小模型在小数据集上容易记住训练文本，公开权重可能导致训练内容被间接泄露。

---

### 3. 生成脚本依赖原始字符表

由于本项目是字符级模型，`generate.py` 需要通过训练文本重建字符表。如果训练后修改、删除或更换了 `TrainingData/` 中的文本，可能导致 checkpoint 和当前字符表不匹配，从而无法正确加载模型。

更完善的做法是在训练时额外保存 vocab 信息，这可以作为后续改进方向。

---

## 项目定位

本项目不是为了训练一个真正可用的大语言模型，而是为了理解 Transformer 语言模型的基本训练流程。

在 RTX 4060 这样的消费级显卡上，小规模字符级模型已经可以生成局部通顺的中文文本，但同时也会暴露出数据规模不足、过拟合、语义控制弱、上下文窗口短等问题。

因此，本项目更适合作为 Transformer 学习、课程展示和小模型实验复现项目。
