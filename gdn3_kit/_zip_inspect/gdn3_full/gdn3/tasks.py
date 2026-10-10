"""Data + evaluation tasks.

MQAR (multi-query associative recall) layout of a sequence of fixed length L:
    [ P facts: ka kb val ] [ F filler ] [ n_q queries: SEP ka kb val ]        F = L - 3P - 4 n_q
DISJOINT vocabulary ranges (the NB35 failure was overlapping ranges): keys 0..127 (pair key = (ka,kb) -> 16384 keys),
values 128..191, filler 192..249, SEP 250, PAD 251.  Loss / accuracy only at the answer (value) tokens.
"""
import os
import urllib.request
import zipfile

import numpy as np

NK, NV = 128, 64
VAL0, FIL0, NFIL, SEP, PAD = NK, NK + NV, 58, 250, 251
VOCAB = 256


def make_mqar(rng, B, P, n_q, L):
    """-> ids (B,L) int32, mask (B,L) bool (answer positions), qi (B,n_q) fact index asked by each query."""
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


def oracle_weights(qi, P, n_q, L, seg):
    """TRUE oracle router for MQAR (fixed segmentation): gate weights (B,L,n) with n = L//seg banks.  For query i the weight
    sits on the bank(s) that hold the asked fact's tokens (3*qi .. 3*qi+2; split evenly if the fact straddles a boundary),
    on the 4 positions of that query; everywhere else 0 (no archive read).  Causality is enforced by the mixer's `valid` mask."""
    if L % seg:
        raise ValueError("oracle needs L % seg == 0")
    B, n = qi.shape[0], L // seg
    w = np.zeros((B, L, n), np.float32)
    base = L - 4 * n_q
    for b in range(B):
        for i in range(n_q):
            f = int(qi[b, i])
            banks = sorted({(3 * f) // seg, (3 * f + 2) // seg})
            for t in range(4):
                for bk in banks:
                    w[b, base + 4 * i + t, bk] = 1.0 / len(banks)
    return w


def mqar_batch_fn(P_list, n_q, L, B, oracle_seg=None):
    P_list = list(P_list)

    def fn(rng, step):
        P = int(P_list[rng.integers(len(P_list))])
        ids, mask, qi = make_mqar(rng, B, P, n_q, L)
        out = dict(ids=ids, mask=mask)
        if oracle_seg:
            out["oracle"] = oracle_weights(qi, P, n_q, L, oracle_seg)
        return out
    return fn


def load_enwik8(split="train", path="enwik8.zip"):
    if not os.path.exists(path):
        urllib.request.urlretrieve("https://mattmahoney.net/dc/enwik8.zip", path)
    with zipfile.ZipFile(path) as z, z.open("enwik8") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    a, b = {"train": (0, 90_000_000), "val": (90_000_000, 95_000_000), "test": (95_000_000, 100_000_000)}[split]
    return data[a:b]


def synthetic_bytes(n=2_000_000, seed=0):
    """Offline fallback: zipf word stream with long-range repeats (smoke tests only)."""
    rng = np.random.default_rng(seed)
    words = [rng.integers(97, 123, rng.integers(3, 9)).astype(np.uint8) for _ in range(2000)]
    out, i = np.empty(n, np.uint8), 0
    while i < n:
        w = words[min(int(rng.zipf(1.3)) - 1, len(words) - 1)]
        m = min(len(w), n - i)
        out[i:i + m] = w[:m]
        i += m
        if i < n:
            out[i] = 32
            i += 1
    return out


def get_bytes(name, split="train"):
    if name == "enwik8":
        return load_enwik8(split)
    d = synthetic_bytes() if name == "synthetic" else np.frombuffer(open(name, "rb").read(), dtype=np.uint8)
    cut = int(len(d) * 0.95)
    return d[:cut] if split == "train" else d[cut:]


def lm_batch_fn(data, L, B):
    def fn(rng, step):
        st = rng.integers(0, len(data) - L - 1, B)
        return dict(ids=np.stack([data[s:s + L] for s in st]).astype(np.int32), mask=np.ones((B, L), bool))
    return fn
