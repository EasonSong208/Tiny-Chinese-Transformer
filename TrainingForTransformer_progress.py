import os
import sys
from pathlib import Path
from itertools import cycle

# 关键：先修正项目根目录，再导入 Transformers 包
# 假设本文件位于 /home/song_eason/Transformers/TrainingForTransformer.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Transformers.TransformsFor4060 import Transformer
from Transformers.DataLoader import TextDataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def count_parameters(model):
    # 统计可训练参数量（不含冻结参数）
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def generate_sample(model, dataset, device, prompt="今天", max_new_tokens=120):
    # 采样阶段不需要梯度，先切到 eval 模式以禁用 dropout 等
    model.eval()

    # 只保留词表里存在的字符，避免生僻 prompt 报错
    ids = [dataset.char_to_idx[ch] for ch in prompt if ch in dataset.char_to_idx]
    if not ids:
        # 如果 prompt 全是 OOV（词表外），用 0 作为兜底
        ids = [0]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # 模型只看最近 block_size 个 token，避免超长序列
        idx_cond = idx[:, -dataset.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        # softmax 得到下一步的采样分布
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    text = "".join(dataset.idx_to_char[int(i)] for i in idx[0].tolist())
    # 采样完恢复训练模式
    model.train()
    return text


def train_one_step(model, batch, optimizer, criterion, device):
    inputs, targets = batch
    # 非阻塞拷贝在 pin_memory=True 时更有效
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    # 交叉熵需要 [N, C] 与 [N] 的形状
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
    loss.backward()
    # 裁剪梯度，避免训练初期梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def main():
    # 路径不再依赖你从哪个目录启动
    data_dir = PROJECT_ROOT / "Transformers" / "TrainingData"
    save_dir = PROJECT_ROOT / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Folder '{data_dir}' not found! Please create it and add .txt files.")

    txt_files = sorted([p for p in data_dir.iterdir() if p.suffix == ".txt"])
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{data_dir}'!")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {data_dir}")
    print(f"Found {len(txt_files)} .txt files:")
    for p in txt_files:
        print(f"  - {p.name}")

    # 合并所有训练文本
    all_text = ""
    for p in txt_files:
        all_text += p.read_text(encoding="utf-8") + "\n"

    print(f"Total text length: {len(all_text)} characters")

    # 超参数（根据显存与数据规模自行调整）
    block_size = 128
    batch_size = 32
    n_embd = 256
    n_heads = 4
    n_layers = 4
    learning_rate = 3e-4

    # 现在这个是真正的“参数更新步数”，不是 epoch 数
    max_steps = 5000
    log_interval = 50
    sample_interval = 500
    save_interval = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 构建字符级数据集
    dataset = TextDataset(all_text, block_size)
    vocab_size = len(dataset.vocab)

    # 随机打乱 + drop_last 保证 batch 形状一致
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    steps_per_epoch = len(dataloader)
    print(f"Vocab size: {vocab_size}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Max update steps: {max_steps}")
    print(f"Equivalent epochs: {max_steps / max(1, steps_per_epoch):.2f}")

    # 初始化模型
    model = Transformer(vocab_size, n_embd, n_heads, n_layers).to(device)
    print(f"Trainable parameters: {count_parameters(model) / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    running_loss = 0.0

    # tqdm 可选：未安装时仍可运行
    if tqdm is not None:
        progress = tqdm(total=max_steps, desc="Training", dynamic_ncols=True)
    else:
        progress = None
        print("Tip: pip install tqdm 可以显示更漂亮的进度条。")

    # cycle 让 dataloader 无限循环，按 step 控制训练长度
    data_iter = cycle(dataloader)

    for step in range(1, max_steps + 1):
        batch = next(data_iter)
        loss = train_one_step(model, batch, optimizer, criterion, device)
        running_loss += loss

        if progress is not None:
            progress.update(1)
            progress.set_postfix(loss=f"{loss:.4f}")

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            running_loss = 0.0
            print(f"Step {step:5d}/{max_steps}, loss = {avg_loss:.4f}")

        # 定期生成样本文本，观察训练效果
        if step % sample_interval == 0:
            sample = generate_sample(model, dataset, device, prompt="今天", max_new_tokens=120)
            print("\n===== sample =====")
            print(sample)
            print("==================\n")

        # 定期保存 checkpoint，便于中断恢复
        if step % save_interval == 0:
            ckpt_path = save_dir / f"transformer_step_{step}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    if progress is not None:
        progress.close()

    final_path = save_dir / "transformer_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
