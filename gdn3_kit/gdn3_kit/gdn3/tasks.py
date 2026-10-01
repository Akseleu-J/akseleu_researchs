"""Data + evaluation tasks.

MQAR (multi-query associative recall) is the capacity benchmark: it can SEE memory (enwik8 with a few hundred
steps cannot).  Layout of a sequence of fixed length L:

    [ P facts: ka kb val ] [ F filler ] [ n_q queries: SEP ka kb val ] 

Tokens: keys 0..127 (pair key = (ka,kb)), values 128..191, filler 192..249, SEP 250, PAD 251.  The target of a
query is its `val` token; loss/accuracy are measured there only.  Fixed L => one jit compilation for all P.
"""
from __future__ import annotations
import os
import urllib.request
import zipfile

import numpy as np

NK, NV = 128, 64
VAL0, FIL0, NFIL, SEP, PAD = NK, NK + NV, 58, 250, 251
VOCAB = 256


def make_mqar(rng: np.random.Generator, B: int, P: int, n_q: int, L: int):
    """Returns ids (B,L) int32, mask (B,L) bool (target positions), qseg_pos (B,n_q) fact index per query."""
    F = L - 3 * P - 4 * n_q
    if F < 0:
        raise ValueError(f"L={L} too short for P={P}, n_q={n_q} (need >= {3 * P + 4 * n_q})")
    ids = np.empty((B, L), np.int32)
    mask = np.zeros((B, L), bool)
    qi_all = np.empty((B, n_q), np.int32)
    base = 3 * P + F
    for b in range(B):
        pairs = rng.choice(NK * NK, size=P, replace=False)
        ka, kb = pairs // NK, pairs % NK
        val = VAL0 + rng.integers(0, NV, P)
        body = np.stack([ka, kb, val], 1).reshape(-1)
        fill = FIL0 + rng.integers(0, NFIL, F)
        qi = rng.integers(0, P, n_q)
        qs = np.stack([np.full(n_q, SEP), ka[qi], kb[qi], val[qi]], 1).reshape(-1)
        ids[b] = np.concatenate([body, fill, qs])
        mask[b, base + 3::4] = True
        qi_all[b] = qi
    return ids, mask, qi_all


def mqar_batch_fn(P_list, n_q: int, L: int, B: int):
    """Closure for train(): samples P from P_list per batch (all batches share shape (B,L))."""
    P_list = list(P_list)

    def fn(rng: np.random.Generator, step: int):
        P = int(P_list[rng.integers(len(P_list))])
        ids, mask, _ = make_mqar(rng, B, P, n_q, L)
        return dict(ids=ids, mask=mask)
    return fn


def oracle_weights(qi, P: int, n_q: int, L: int, C: int, n: int):
    """Oracle router input (B,L,n): one-hot of the segment holding the queried fact (3 tokens per fact) on the
    4 tokens of each query.  Diagnostics: oracle >> normal => fix the router; oracle ~ normal => capacity."""
    B = qi.shape[0]
    F = L - 3 * P - 4 * n_q
    base = 3 * P + F
    A = np.zeros((B, L, n), np.float32)
    for b in range(B):
        for i, p_ in enumerate(qi[b]):
            j = (3 * int(p_)) // C
            A[b, base + 4 * i: base + 4 * i + 4, j] = 1.0
    return A


# ----------------------------------------------------------------------------- bytes LM
def load_enwik8(split: str = "train", path: str = "enwik8.zip") -> np.ndarray:
    if not os.path.exists(path):
        urllib.request.urlretrieve("https://mattmahoney.net/dc/enwik8.zip", path)
    with zipfile.ZipFile(path) as z, z.open("enwik8") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    a, b = {"train": (0, 90_000_000), "val": (90_000_000, 95_000_000), "test": (95_000_000, 100_000_000)}[split]
    return data[a:b]


def load_bytes_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


def synthetic_bytes(n: int = 4_000_000, seed: int = 0) -> np.ndarray:
    """Offline fallback (no internet): a Markov-ish byte stream with long-range repeats. For smoke tests only."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, np.uint8)
    words = [rng.integers(97, 123, rng.integers(3, 9)).astype(np.uint8) for _ in range(2000)]
    i = 0
    while i < n:
        w = words[min(int(rng.zipf(1.3)) - 1, len(words) - 1)]
        m = min(len(w), n - i)
        out[i:i + m] = w[:m]
        i += m
        if i < n:
            out[i] = 32
            i += 1
    return out


def lm_batch_fn(data: np.ndarray, L: int, B: int):
    def fn(rng: np.random.Generator, step: int):
        starts = rng.integers(0, len(data) - L - 1, B)
        ids = np.stack([data[s:s + L] for s in starts]).astype(np.int32)
        return dict(ids=ids, mask=np.ones((B, L), bool))
    return fn


def get_bytes(name: str, split: str = "train") -> np.ndarray:
    if name == "enwik8":
        return load_enwik8(split)
    if name == "synthetic":
        d = synthetic_bytes()
        cut = int(len(d) * 0.95)
        return d[:cut] if split == "train" else d[cut:]
    d = load_bytes_file(name)
    cut = int(len(d) * 0.95)
    return d[:cut] if split == "train" else d[cut:]
