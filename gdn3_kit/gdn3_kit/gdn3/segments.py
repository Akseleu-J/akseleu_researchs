"""Segmentation of a sequence into independent banks (batch-axis folding) + the dynamic-count planner."""
from __future__ import annotations
import math

import jax.numpy as jnp


def round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def pad_to_multiple(t, mult: int, axis: int = 2):
    """Right-pad with zeros along `axis`. Zero tokens (k=0, beta=0, w=0, g=0) are exact no-ops for the state."""
    r = (-t.shape[axis]) % mult
    if r == 0:
        return t
    pad = [(0, 0)] * t.ndim
    pad[axis] = (0, r)
    return jnp.pad(t, pad)


def fold_overlap(t, n: int, C: int, W: int):
    """(B,H,n*C,D) -> (B*n,H,W+C,D).  Segment j covers tokens [j*C - W, (j+1)*C); the first W tokens of segment 0
    are zero padding (no-ops).  W=0 is a pure reshape.  Batch index of segment j of example b is b*n + j."""
    B, H, L, D = t.shape
    assert L == n * C, (L, n, C)
    if W == 0:
        return t.reshape(B, H, n, C, D).transpose(0, 2, 1, 3, 4).reshape(B * n, H, C, D)
    tp = jnp.pad(t, ((0, 0), (0, 0), (W, 0), (0, 0)))
    idx = jnp.arange(n)[:, None] * C + jnp.arange(W + C)[None, :]          # (n, W+C)
    seg = tp[:, :, idx, :]                                                  # (B,H,n,W+C,D)
    return seg.transpose(0, 2, 1, 3, 4).reshape(B * n, H, W + C, D)


def unfold_out(o, B: int, n: int, W: int):
    """(B*n,H,W+C,D) -> (B,H,n*C,D); the warm-up outputs (first W) are discarded."""
    BN, H, T, D = o.shape
    o = o[:, :, W:].reshape(B, n, H, T - W, D)
    return o.transpose(0, 2, 1, 3, 4).reshape(B, H, n * (T - W), D)


def unfold_states(S, B: int, n: int):
    """(B*n,H,D,E) -> (B,H,n,D,E)"""
    BN, H, D, E = S.shape
    return S.reshape(B, n, H, D, E).transpose(0, 2, 1, 3, 4)


def split_segments(t, n: int, C: int):
    """(B,H,n*C,D) -> (n,B,H,C,D)  (scan layout for the cumulative snapshot pass)."""
    B, H, L, D = t.shape
    return t.reshape(B, H, n, C, D).transpose(2, 0, 1, 3, 4)


def merge_segments(t):
    """(n,B,H,C,D) -> (B,H,n*C,D)"""
    n, B, H, C, D = t.shape
    return t.transpose(1, 2, 0, 3, 4).reshape(B, H, n * C, D)


def plan_segments(L: int, c_target: int, n_max: int | None = None, align: int = 128) -> dict:
    """Dynamic number of matrices: choose the bank size C (multiple of `align`) and count n for a length L.

    Without a budget: n = ceil(L / C).  With a budget n_max, C grows so that n <= n_max (capacity per bank
    then grows with L: use n_max only when memory, not quality, is the constraint).
    """
    C = max(align, round_up(c_target, align))
    n = -(-L // C)
    if n_max is not None and n > n_max:
        C = round_up(-(-L // n_max), align)
        n = -(-L // C)
    return dict(C=C, n=n, L_padded=n * C, pad=n * C - L)


def capacity_pairs(d_head: int, frac: float = 1 / 3) -> int:
    """Rule of thumb: a D x D delta-rule bank is reliably searchable up to ~D/3 stored key-value pairs
    (numpy simulation in docs/DESIGN.md); beyond ~D/2 retrieval collapses."""
    return max(1, int(d_head * frac))


def state_bytes(cfg_heads: int, d_head: int, n_banks: int, layers: int, dtype_bytes: int = 4) -> int:
    return layers * cfg_heads * d_head * d_head * n_banks * dtype_bytes


def segments_for(L: int, C: int) -> int:
    return math.ceil(L / C)
