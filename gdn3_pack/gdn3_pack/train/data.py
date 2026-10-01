"""Token sources for LM training. All return a 1-D integer array; Sampler draws random windows; val blocks are fixed."""
import zipfile, urllib.request
from pathlib import Path
import numpy as np


def load_enwik8(path="enwik8.zip"):
    if not Path(path).exists():
        urllib.request.urlretrieve("https://mattmahoney.net/dc/enwik8.zip", path)
    with zipfile.ZipFile(path) as z, z.open("enwik8") as f:
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return {"train": d[:90_000_000], "val": d[90_000_000:95_000_000], "test": d[95_000_000:]}


def load_source(spec):
    """'enwik8' | 'bytes:/path/file.txt' (byte-level, 95/5 split) | 'npy:/path/tokens.npy' (uint16/uint32 BPE ids, 95/5 split)"""
    if spec == "enwik8":
        return load_enwik8()
    kind, _, path = spec.partition(":")
    d = np.frombuffer(open(path, "rb").read(), dtype=np.uint8) if kind == "bytes" else np.load(path, mmap_mode="r")
    cut = int(len(d) * 0.95)
    return {"train": d[:cut], "val": d[cut:], "test": d[cut:]}


class Sampler:
    def __init__(self, tokens, L, bsz, seed=0):
        self.t, self.L, self.bsz, self.rng = tokens, L, bsz, np.random.default_rng(seed)
        assert len(tokens) > L + 1

    def next(self):
        st = self.rng.integers(0, len(self.t) - self.L, self.bsz)
        return np.stack([np.asarray(self.t[s:s + self.L]) for s in st]).astype(np.int32)


def val_blocks(tokens, L, n_blocks):
    n = min(n_blocks, len(tokens) // L)
    return np.asarray(tokens[:n * L]).reshape(n, L).astype(np.int32)
