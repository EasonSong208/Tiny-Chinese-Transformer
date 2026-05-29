"""
TrainingForTransformer_epoch.py
================================
基于 TrainingForTransformer_progress.py 改进：
  1. 改按 epoch 训练（可配置训练轮数）
  2. 保留 tqdm 进度条
  3. 保留定期生成样本文本
  4. 保留 checkpoint 保存
  5. 增加验证集评估 & 最佳模型保存
  6. 增加学习率调度（CosineAnnealing）
"""

import os
import sys
from pathlib import Path

# ---------- 修正项目根目录 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from Transformers.TransformsFor4060 import Transformer, TransformerLM
from Transformers.DataLoader import TextDataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ======================== 工具函数 ========================

def count_parameters(model):
    """统计可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def generate_sample(model, dataset, device, prompt="今天", max_new_tokens=120):
    """从给定 prompt 出发，用模型生成一段文本"""
    model.eval()

    ids = [dataset.char_to_idx[ch] for ch in prompt if ch in dataset.char_to_idx]
    if not ids:
        ids = [0]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -dataset.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    text = "".join(dataset.idx_to_char[int(i)] for i in idx[0].tolist())
    model.train()
    return text


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, max_batches=20):
    """在验证集上计算平均 loss"""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


# ======================== 主函数 ========================

def main():
    # ---------- 超参数 ----------
    block_size = 128
    batch_size = 32
    n_embd = 256
    n_heads = 4
    n_layers = 4
    learning_rate = 3e-4
    num_epochs = 20                # 训练轮数
    val_split = 0.05               # 验证集比例
    log_interval = 50              # 每 N 步输出一次 loss
    sample_epochs = 2              # 每 N 个 epoch 生成一次样本
    save_epochs = 2                # 每 N 个 epoch 保存一次 checkpoint

    # ---------- 路径准备 ----------
    data_dir = PROJECT_ROOT / "Transformers" / "TrainingData"
    # 保存目录放在 Transformers 目录下，便于 WSL/Windows 访问
    save_dir = Path(__file__).resolve().parent / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Folder '{data_dir}' not found! Please create it and add .txt files.")

    txt_files = sorted([p for p in data_dir.iterdir() if p.suffix == ".txt"])
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{data_dir}'!")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir:    {data_dir}")
    print(f"Found {len(txt_files)} .txt files:")
    for p in txt_files:
        print(f"  - {p.name}")

    # 合并所有训练文本
    all_text = ""
    for p in txt_files:
        all_text += p.read_text(encoding="utf-8") + "\n"
    print(f"Total text length: {len(all_text)} characters")

    # ---------- 设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------- 数据集 & DataLoader ----------
    full_dataset = TextDataset(all_text, block_size)
    vocab_size = len(full_dataset.vocab)

    # 划分训练集 / 验证集
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs

    print(f"Vocab size:          {vocab_size}")
    print(f"Train samples:       {train_size}")
    print(f"Val samples:         {val_size}")
    print(f"Steps per epoch:     {steps_per_epoch}")
    print(f"Num epochs:          {num_epochs}")
    print(f"Total train steps:   {total_steps}")

    # ---------- 模型 ----------
    model = Transformer(vocab_size, n_embd, n_heads, n_layers).to(device)
    # 也可用 torch.compile 加速（PyTorch >= 2.0）
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    print(f"Trainable params:    {count_parameters(model) / 1e6:.2f} M")

    # ---------- 优化器 & 调度器 & 损失函数 ----------
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    # 优化器使用 AdamW，带权重衰减有助于正则化
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    # 这里使得学习率在训练过程中逐渐降低，帮助模型更好地收敛
    criterion = nn.CrossEntropyLoss()
    # 交叉熵损失适用于语言建模任务

    # ---------- 训练状态 ----------
    model.train()
    global_step = 0
    best_val_loss = float("inf")

    print(f"\n{'='*50}")
    print(f"Start training for {num_epochs} epochs")
    print(f"{'='*50}\n")

    # ---------- Epoch 循环 ----------
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        batch_count = 0

        # 每个 epoch 一个进度条
        if tqdm is not None:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{num_epochs}", dynamic_ncols=True, leave=True)
        else:
            pbar = train_loader

        for batch in pbar:
            global_step += 1
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            batch_count += 1

            # 更新进度条显示
            if tqdm is not None and isinstance(pbar, tqdm):
                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_now:.2e}")

            # 每隔 log_interval 步输出详细信息
            if global_step % log_interval == 0:
                avg_recent = epoch_loss / batch_count
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  Step {global_step:6d} | Epoch {epoch:3d}/{num_epochs} | "
                      f"loss = {avg_recent:.4f} | lr = {lr_now:.2e}")

        # ---- 每个 epoch 结束后的处理 ----
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        print(f"\n>>> Epoch {epoch:3d}/{num_epochs} finished, avg loss = {avg_epoch_loss:.4f}")

        # 验证集评估
        if len(val_loader) > 0:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"    Val loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = save_dir / "transformer_best.pth"
                torch.save(model.state_dict(), best_path)
                print(f"    New best model saved: {best_path}")

        # 定期生成样本
        if epoch % sample_epochs == 0:
            sample = generate_sample(model, full_dataset, device, prompt="今天", max_new_tokens=120)
            print("\n  ===== sample =====")
            print("  " + sample.replace("\n", "\n  "))
            print("  ==================\n")

        # 定期保存 checkpoint
        if epoch % save_epochs == 0:
            ckpt_path = save_dir / f"transformer_epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        print()  # 空行分隔 epoch

    # ---------- 训练完成 ----------
    final_path = save_dir / "transformer_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()