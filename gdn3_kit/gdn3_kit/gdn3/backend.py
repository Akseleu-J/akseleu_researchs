"""Kernel backends with a common interface.

    call(q, k, v, w, b, g, h0=None) -> (o, h_final)
        q,k,v,w,b,g : (B, H, L, D) float32, head-major;  g <= 0 (log-decay per key channel)
        h0, h_final : (B, H, D, D) float32, indexed [key, value]
        o           : (B, H, L, D)

`serial`  - pure-JAX token recurrence (ground truth; any D, any L; CPU-friendly; slow). Used by the unit tests.
`pallas`  - atomic_gdn2.make_blr_trainable (TPU v5e kernels; L % 128 == 0).
"""
from __future__ import annotations
import dataclasses as dc
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp

HIGHEST = jax.lax.Precision.HIGHEST


@dc.dataclass
class Backend:
    name: str
    call: Callable
    bt: int            # kernel chunk (sequence lengths must be multiples of it); 1 for serial
    g_max: float
    kcfg: Any = None


def gdn_step(h, q, k, v, w, b, g, scale):
    """One token of the GDN-2 recurrence (same math as atomic_gdn2.reference.token_serial_ref).

    h (...,D,D)[key,value]; q,k,v,w,b,g (...,D).  Returns (h_new, o)."""
    a = jnp.exp(jnp.minimum(g, 0.0))
    h = h * a[..., :, None]
    erase = jnp.einsum("...d,...de->...e", b * k, h, precision=HIGHEST)
    v_new = w * v - erase
    h = h + k[..., :, None] * v_new[..., None, :]
    o = jnp.einsum("...de,...d->...e", h, q * scale, precision=HIGHEST)
    return h, o


def serial_backend(d_head: int, g_max: float = 5.0) -> Backend:
    scale = 1.0 / math.sqrt(d_head)

    def call(q, k, v, w, b, g, h0=None):
        q, k, v, w, b, g = (t.astype(jnp.float32) for t in (q, k, v, w, b, g))
        B, H, L, D = q.shape
        h0 = jnp.zeros((B, H, D, D), jnp.float32) if h0 is None else h0.astype(jnp.float32)

        def step(h, xs):
            return gdn_step(h, *xs, scale)

        xs = tuple(jnp.moveaxis(t, 2, 0) for t in (q, k, v, w, b, g))
        hf, o = jax.lax.scan(step, h0, xs)
        return jnp.moveaxis(o, 0, 2), hf

    return Backend("serial", call, 1, g_max)


def pallas_backend(d_head: int, interpret=None, **cfg_kw) -> Backend:
    from .env import ensure_kernel
    ag = ensure_kernel()
    kw = dict(bt=128, score_bs=128, bs2=32, mb=8, diag="btl", dot_mode="highest",
              solve_dot_mode="bf16x3", b3_dot_mode="bf16x3", vmem_mb=150.0)
    kw.update(cfg_kw)
    if interpret is not None:
        kw["interpret"] = interpret
    kcfg = ag.make_cfg(**kw)
    core = ag.make_blr_trainable(kcfg, 1.0 / math.sqrt(d_head))

    def call(q, k, v, w, b, g, h0=None):
        q, k, v, w, b, g = (t.astype(jnp.float32) for t in (q, k, v, w, b, g))
        if h0 is not None:
            h0 = h0.astype(jnp.float32)
        return core(q, k, v, w, b, g, h0)

    return Backend("pallas", call, kcfg.bt, kcfg.g_max, kcfg)


def make_backend(kind: str, d_head: int, **kw) -> Backend:
    if kind == "serial":
        return serial_backend(d_head, **kw)
    if kind in ("pallas", "tpu"):
        return pallas_backend(d_head, **kw)
    raise ValueError(f"unknown backend {kind!r}; use 'serial' or 'pallas'")
