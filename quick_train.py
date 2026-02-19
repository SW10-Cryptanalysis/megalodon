import time
import torch
from torch.utils.data import DataLoader, Dataset

# Monkeypatch distributed helpers for single-process runs
import megalodon.distributed as _mdist

_mdist.get_model_parallel_world_size = lambda: 1
_mdist.get_model_parallel_rank = lambda: 0
_mdist.get_model_parallel_group = lambda: object()
_mdist.get_data_parallel_group = lambda: object()
_mdist.get_chunk_parallel_group = lambda: object()
_mdist.get_chunk_parallel_world_size = lambda: 1
_dist_funcs = [
    "get_chunk_parallel_rank",
    "get_chunk_parallel_prev_rank",
    "get_chunk_parallel_next_rank",
]
for fn in _dist_funcs:
    setattr(_mdist, fn, lambda: 0)

from megalodon.config import ModelConf
from megalodon.model.mega import Mega


class RandSeqDataset(Dataset):
    def __init__(self, n_samples: int, seq_len: int, vocab_size: int):
        self.n = n_samples
        self.seq_len = seq_len
        self.vocab = vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seq = torch.randint(0, self.vocab, (self.seq_len,), dtype=torch.long)
        return {"input_ids": seq, "labels": seq.clone()}


def get_tiny_cfg(vocab_size: int, seq_len: int) -> ModelConf:
    cfg = ModelConf()
    cfg.num_layers = 2
    cfg.model_dim = 64
    cfg.z_dim = 32
    cfg.value_dim = 64
    cfg.num_heads = 1
    cfg.ffn_hidden_dim = 128
    cfg.chunk_size = max(seq_len, 128)
    cfg.vocab_size = vocab_size
    cfg.norm_num_groups = 1
    return cfg


def train_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = 256
    seq_len = 64

    cfg = get_tiny_cfg(vocab, seq_len)
    print("Building Mega model (tiny)...")
    model = Mega(cfg)
    model.to(device)
    model.train()

    dataset = RandSeqDataset(200, seq_len, vocab)
    dl = DataLoader(dataset, batch_size=2, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_f = torch.nn.CrossEntropyLoss()

    it = iter(dl)
    print("Starting a few training steps...")
    for step in range(6):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        logits, _ = model(x)
        # logits: (B, S, V)
        B, S, V = logits.shape
        loss = loss_f(logits.view(B * S, V), y.view(B * S))

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"step {step} loss={loss.item():.4f}")
        time.sleep(0.1)


if __name__ == "__main__":
    train_smoke()
