import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Transformers.TransformsFor4060 import Transformer
from Transformers.DataLoader import TextDataset
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 确保可以正确导入 Transformers 模块中的内容


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    # 1. 收集 trainingdata 文件夹下的所有 .txt 文件
    data_dir = "Transformers/TrainingData"
    all_text = ""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Folder '{data_dir}' not found! Please create it and add .txt files.")

    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{data_dir}'!")

    print(f"Found {len(txt_files)} .txt files: {txt_files}")
    for filename in txt_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"

    print(f"Total text length: {len(all_text)} characters")

    # 2. 超参数 — block_size 必须和 Transformer.py 中的 block_size 一致
    block_size = 128
    batch_size = 32
    n_embd = 256
    n_heads = 4
    n_layers = 4
    learning_rate = 3e-4
    max_iters = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 预计在4060上训练时间约为1-2小时，具体取决于数据量和模型复杂度

    # 3. 构建数据集和 DataLoader
    dataset = TextDataset(all_text, block_size)
    vocab_size = len(dataset.vocab)  # 动态获取词汇表大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Vocab size: {vocab_size}")

    # 4. 实例化模型、优化器、损失函数
    model = Transformer(vocab_size, n_embd, n_heads, n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环
    for i in range(max_iters):
        avg_loss = train(model, dataloader, optimizer, criterion, device)
        if i % 100 == 0:
            print(f"Iter {i}, Average Loss: {avg_loss:.4f}")

    # 6. 保存模型
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "transformer_model.pth"))
    print(f"Training complete! Model saved to {save_dir}/transformer_model.pth")

if __name__ == "__main__":
    main()
