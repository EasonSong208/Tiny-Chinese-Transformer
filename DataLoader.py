# 在这个py中，主要执行就是定义一个数据加载器来加载训练数据，进行预处理，
# 并将其转换为适合Transformer模型输入的格式。

import torch
from torch.utils.data import Dataset, DataLoader
class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        self.vocab = sorted(set(text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y
    
    @staticmethod
    # 这个静态方法用来创建一个DataLoader实例，输入是文本数据，块大小和批量大小。
    def get_dataloader(text, block_size, batch_size):
        dataset = TextDataset(text, block_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
# 这个TextDataset类实现了一个简单的文本数据集，输入是一个字符串文本和块大小，输出是一个索引张量。