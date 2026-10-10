"""Kernel backends with one interface:   call(q,k,v,w,b,g,h0=None) -> (o, h_final)
   q..g: (B,H,L,D) f32 head-major, g<=0 per key channel; h0/h_final (B,H,D,D) [key,value]; o uses scale 1/sqrt(D).
   serial : pure-JAX token scan (ground truth, any L, any D, CPU-friendly).
   pallas : atomic_gdn2 kernels (TPU v5e; L % 128 == 0).   Rewriting later in Pallas only touches this file + fwd/bwd kernels."""
import dataclasses as dc
import math
from typing import Callable

import jax
import jax.numpy as jnp

from . import ops


@dc.dataclass
class Backend:
    name: str
    call: Callable
    bt: int
    g_max: float


def serial_backend(d_head: int, g_max: float = 5.0) -> Backend:
    scale = 1.0 / math.sqrt(d_head)

    def call(q, k, v, w, b, g, h0=None):
        q, k, v, w, b, g = (t.astype(jnp.float32) for t in (q, k, v, w, b, g))
        B, H, L, D = q.shape
        h0 = jnp.zeros((B, H, D, D), jnp.float32) if h0 is None else h0.astype(jnp.float32)

        def step(h, xs):
            h2, _ = ops.gdn_core(h, None, *xs[1:])
            return h2, scale * ops._vec_h(xs[0], h2)

        hf, o = jax.lax.scan(step, h0, tuple(jnp.moveaxis(t, 2, 0) for t in (q, k, v, w, b, g)))
        return jnp.moveaxis(o, 0, 2), hf

    return Backend("serial", call, 1, g_max)


def pallas_backend(d_head: int, **cfg_kw) -> Backend:
    from .env import ensure_kernel
    ag = ensure_kernel()
    kw = dict(bt=128, score_bs=128, bs2=32, mb=8, diag="btl", dot_mode="highest",
              solve_dot_mode="bf16x3", b3_dot_mode="bf16x3", vmem_mb=150.0)
    kw.update(cfg_kw)
    kcfg = ag.make_cfg(**kw)
    core = ag.make_blr_trainable(kcfg, 1.0 / math.sqrt(d_head))

    def call(q, k, v, w, b, g, h0=None):
        q, k, v, w, b, g = (t.astype(jnp.float32) for t in (q, k, v, w, b, g))
        return core(q, k, v, w, b, g, None if h0 is None else h0.astype(jnp.float32))

    return Backend("pallas", call, kcfg.bt, kcfg.g_max)


def make_backend(kind: str, d_head: int, **kw) -> Backend:
    if kind == "serial":
        return serial_backend(d_head, **kw)
    if kind in ("pallas", "tpu"):
        return pallas_backend(d_head, **kw)
    raise ValueError(f"unknown backend {kind!r}; use 'serial' or 'pallas'")
