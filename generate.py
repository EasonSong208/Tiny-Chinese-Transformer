"""
独立文本生成脚本
================

运行方式：
    cd /home/song_eason
    python -m Transformers.generate --prompt "今天" --max_new_tokens 300

也可以不写 --prompt，让脚本在终端里等待你输入：
    python -m Transformers.generate --max_new_tokens 300

如果想连续测试多个 prompt：
    python -m Transformers.generate --interactive

这个文件只负责“推理/生成”，不负责训练。它会：
  1. 读取 Transformers/TrainingData 里的训练文本，重建训练时使用的字符表；
  2. 加载 Transformers/checkpoints/transformer_best.pth；
  3. 根据 prompt 自回归地生成新字符；
  4. 把完整文本打印到终端。

注意：你的模型是字符级语言模型，所以 vocab 是从训练文本里的“字符集合”
构造出来的。生成脚本必须用和训练时相同的文本、相同的排序方式来重建字符表，
否则 checkpoint 里的 token id 和实际字符会对不上。
"""

import argparse
import sys
from pathlib import Path

import torch

# 本文件路径：/home/song_eason/Transformers/generate.py
# PROJECT_ROOT 应该是 /home/song_eason，这样才能用 python -m Transformers.generate
# 的包导入方式稳定找到 Transformers 目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Transformers.DataLoader import TextDataset
from Transformers.TransformsFor4060 import Transformer


# 这些默认超参数必须和 TrainingForTransformer_epoch.py 训练时保持一致。
# checkpoint 只保存了权重，没有保存完整配置，所以推理时需要重新创建同结构模型。
DEFAULT_BLOCK_SIZE = 128
DEFAULT_N_HEADS = 4
DEFAULT_N_LAYERS = 4
DEFAULT_N_EMBD = 256

DEFAULT_DATA_DIR = PROJECT_ROOT / "Transformers" / "TrainingData"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "Transformers" / "checkpoints" / "transformer_best.pth"


def parse_args():
    """解析命令行参数，让同一个脚本可以方便测试不同 prompt 和采样设置。"""
    parser = argparse.ArgumentParser(description="Generate text with the trained small GPT-like Transformer.")
    parser.add_argument("--prompt", type=str, default=None, help="生成的起始文本；不传时会在终端交互输入。")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="最多新生成多少个字符。")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="要加载的模型权重路径。")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="训练文本所在目录，用于重建字符表。")
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE, help="上下文窗口长度，需和训练一致。")
    parser.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS, help="注意力头数量，需和训练一致。")
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS, help="Transformer block 层数，需和训练一致。")
    parser.add_argument("--n_embd", type=int, default=DEFAULT_N_EMBD, help="embedding 维度，需和训练一致。")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度；越低越保守，越高越发散。")
    parser.add_argument("--top_k", type=int, default=40, help="只从概率最高的 k 个字符中采样；设为 0 表示不限制。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子；设置后可复现同样的生成结果。")
    parser.add_argument("--interactive", action="store_true", help="进入连续交互模式，可以反复输入 prompt。")
    return parser.parse_args()


def read_training_text(data_dir):
    """按训练脚本相同方式合并所有 .txt 文件，以重建完全一致的字符表。"""
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data folder not found: {data_dir}")

    txt_files = sorted(p for p in data_dir.iterdir() if p.suffix == ".txt")
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {data_dir}")

    # TrainingForTransformer_epoch.py 中也是按排序后的文件列表逐个 read_text，
    # 并在每个文件后额外加一个换行符。这里保持完全一致。
    all_text = ""
    for path in txt_files:
        all_text += path.read_text(encoding="utf-8") + "\n"
    return all_text


def clean_state_dict(state_dict):
    """兼容 torch.compile 保存出来的 _orig_mod. 前缀。"""
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        cleaned[key] = value
    return cleaned


def load_model(args, dataset, device):
    """创建模型并加载 checkpoint。"""
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 兼容两种常见保存格式：
    #   1. torch.save(model.state_dict(), path)
    #   2. torch.save({"model_state_dict": model.state_dict(), ...}, path)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    state_dict = clean_state_dict(state_dict)

    vocab_size = len(dataset.vocab)

    # checkpoint 的 embedding 第一维就是训练时的 vocab_size。
    # 如果这里和当前 TrainingData 重建出的 vocab_size 不一致，通常说明：
    #   1. 训练后增删/修改了 TrainingData 里的文本；
    #   2. 或者正在加载另一次训练产生的 checkpoint。
    ckpt_vocab_size = state_dict["token_embedding.weight"].shape[0]
    if ckpt_vocab_size != vocab_size:
        raise ValueError(
            f"Vocab size mismatch: checkpoint has {ckpt_vocab_size}, "
            f"but current TrainingData builds {vocab_size}. "
            "Use the exact same training text files as the checkpoint, "
            "or save/load vocab metadata during training."
        )

    model = Transformer(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint. Please check that --n_embd, --n_heads, "
            "--n_layers, and --block_size match the training script."
        ) from exc

    model.eval()
    return model


def encode_prompt(prompt, dataset):
    """把 prompt 中模型认识的字符转成 token id；不认识的字符会被跳过。"""
    ids = [dataset.char_to_idx[ch] for ch in prompt if ch in dataset.char_to_idx]

    # 如果 prompt 里没有任何字符出现在训练词表里，就用 id=0 作为起点。
    # 这样脚本不会直接崩掉，但生成内容会更随机。
    if not ids:
        print("Warning: prompt has no known characters; falling back to token id 0.")
        ids = [0]
    return ids


def sample_next_id(logits, temperature, top_k):
    """根据最后一个位置的 logits 采样下一个 token id。"""
    if temperature <= 0:
        # temperature <= 0 时不随机采样，直接取概率最大的字符。
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k and top_k > 0:
        # 只保留概率最高的 top_k 个候选，其余候选设为 -inf。
        # 这样能减少完全乱跳的字符，让短文本更稳定。
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        cutoff = values[:, [-1]]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_text(model, dataset, device, prompt, max_new_tokens, temperature, top_k):
    """自回归生成文本：每次预测 1 个新字符，再把它接回输入继续预测。"""
    ids = encode_prompt(prompt, dataset)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # 模型训练时最长只看 block_size 个字符；生成时也只把最近的上下文喂给模型。
        idx_cond = idx[:, -dataset.block_size:]

        # logits 形状是 (B, T, vocab_size)，这里只需要最后一个位置的预测结果。
        logits = model(idx_cond)
        last_logits = logits[:, -1, :]

        next_id = sample_next_id(last_logits, temperature=temperature, top_k=top_k)
        idx = torch.cat([idx, next_id], dim=1)

    return "".join(dataset.idx_to_char[int(token_id)] for token_id in idx[0].tolist())


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = read_training_text(args.data_dir)
    dataset = TextDataset(text, args.block_size)
    model = load_model(args, dataset, device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    if args.interactive:
        print("Interactive mode. 输入空行或 Ctrl-D 退出。")
        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except EOFError:
                print()
                break

            if not prompt:
                break

            generated = generate_text(
                model=model,
                dataset=dataset,
                device=device,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            print("=" * 60)
            print(generated)
        return

    if args.prompt is None:
        args.prompt = input("Prompt> ").strip()
        if not args.prompt:
            raise ValueError("Prompt cannot be empty.")

    generated = generate_text(
        model=model,
        dataset=dataset,
        device=device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(f"Prompt: {args.prompt}")
    print("=" * 60)
    print(generated)


if __name__ == "__main__":
    main()
