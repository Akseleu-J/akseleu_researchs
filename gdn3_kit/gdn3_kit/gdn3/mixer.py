"""GDN-3 token mixer (parallel forward).

Modes (see config.py):
  gdn2   : o = GDN2(all heads)
  hybrid : o = GDN2(all heads) + tanh(g) * ArchiveRead(independent banks of archive heads)
  local  : archive heads: independent segment states (+warm-up) + ArchiveRead ; remaining heads: GDN2
  snap   : o = GDN2(all heads) + tanh(g) * ArchiveRead(snapshots of the cumulative state)
"""
from __future__ import annotations
import math

import jax
import jax.numpy as jnp

from .read import archive_read
from .segments import fold_overlap, unfold_out, unfold_states, split_segments, merge_segments


def rmsnorm(x, w, eps: float = 1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w


def l2n(t, eps: float = 1e-6):
    return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps)


def causal_conv(u, w, b):
    """Depthwise causal conv. u (B,L,P), w (P,K), b (P,).  Tap i multiplies u[t-(K-1)+i]."""
    L = u.shape[1]
    K = w.shape[1]
    up = jnp.pad(u, ((0, 0), (K - 1, 0), (0, 0)))
    acc = 0.0
    for i in range(K):
        acc = acc + up[:, i:i + L] * w[:, i]
    return acc + b


def token_features(p, h, cfg):
    """h (...,d) normalised input -> beta, wg, g of shape (...,H,D)."""
    H, D = cfg.n_heads, cfg.d_head
    sh = h.shape[:-1] + (H, D)
    beta = jax.nn.sigmoid(h @ p["wb"] + p["bb"]).reshape(sh)
    wg = jax.nn.sigmoid(h @ p["ww"] + p["bw"]).reshape(sh)
    df = h @ p["wd"] + p["bd"]                                       # (...,H)
    g = -jnp.minimum(jnp.exp(p["decay_a"]) * jax.nn.softplus(df)[..., None], cfg.g_max)
    return beta, wg, g


def _cumulative_scan(be, q, k, v, w, b, g, n, C):
    """Cumulative GDN-2 evaluated segment by segment under lax.scan (one kernel instance); also returns the
    snapshot of the state after every segment (gradient flows through the carried state)."""
    B, H, L, D = q.shape
    xs = tuple(split_segments(t, n, C) for t in (q, k, v, w, b, g))

    def body(h, x):
        o, h2 = be.call(*x, h)
        return h2, (o, h2)

    h0 = jnp.zeros((B, H, D, D), jnp.float32)
    hf, (os_, hs) = jax.lax.scan(body, h0, xs)
    return merge_segments(os_), hf, hs.transpose(1, 2, 0, 3, 4)       # o (B,H,L,D), hf, snaps (B,H,n,D,D)


def mixer_fwd(p, x, cfg, be, oracle=None, want_cache: bool = False, want_stats: bool = False):
    """x (B,L,d) -> (y (B,L,d) residual delta, extras for the streaming cache, stats)."""
    B, L, _ = x.shape
    H, D, P, K, C = cfg.n_heads, cfg.d_head, cfg.proj, cfg.d_conv, cfg.seg
    Ha, Wb, mode = cfg.ha, cfg.bank_warm, cfg.mode
    if L % cfg.align(be.bt):
        raise ValueError(f"L={L} must be a multiple of {cfg.align(be.bt)} (pad the sequence)")

    h = rmsnorm(x, p["norm1"])
    raw = jnp.stack([h @ p["wq"], h @ p["wk"], h @ p["wv"]])                         # (3,B,L,P)
    conv = jnp.stack([causal_conv(raw[i], p["conv_w"][i], p["conv_b"][i]) for i in range(3)])
    act = jax.nn.silu(conv).reshape(3, B, L, H, D)
    q, k, v = l2n(act[0]), l2n(act[1]), act[2]
    beta, wg, g = token_features(p, h, cfg)
    hm = lambda t: t.transpose(0, 2, 1, 3)
    q, k, v, wg, beta, g = map(hm, (q, k, v, wg, beta, g))                           # (B,H,L,D)
    args = (q, k, v, wg, beta, g)

    n = L // C
    hc = None            # cumulative final state (B,H,D,D) for the cache
    S = None             # bank states (B,Ha,n,D,D)
    A_next = None
    h_init = p["h_init"] if Ha else None

    if mode == "gdn2":
        o, hc = be.call(*args)

    elif mode == "hybrid":
        o, hc = be.call(*args)
        fa = lambda t: fold_overlap(t[:, :Ha], n, C, 0)
        h0 = jnp.broadcast_to(h_init[None], (B * n, Ha, D, D))
        _, S_fin = be.call(*map(fa, args), h0)
        S = unfold_states(S_fin, B, n)

    elif mode == "snap":
        o, hc, snaps = _cumulative_scan(be, *args, n, C)
        S = snaps[:, :Ha]

    else:  # local
        fa = lambda t: fold_overlap(t[:, :Ha], n, C, Wb)
        h0 = jnp.broadcast_to(h_init[None], (B * n, Ha, D, D))
        o_s, S_fin = be.call(*map(fa, args), h0)
        o_a = unfold_out(o_s, B, n, Wb)
        S = unfold_states(S_fin, B, n)
        if Ha < H:
            o_c, hc_rest = be.call(*(t[:, Ha:] for t in args))
            o = jnp.concatenate([o_a, o_c], axis=1)
            hc = jnp.concatenate([jnp.zeros((B, Ha, D, D), jnp.float32), hc_rest], axis=1)
        else:
            o = o_a
            hc = jnp.zeros((B, H, D, D), jnp.float32)

    st = {}
    if Ha:
        o_arch, st = archive_read(q[:, :Ha], S, p, cfg, C, oracle=oracle, want_stats=want_stats)
        gate = jnp.tanh(p["arch_gate"])[None, :, None, None]
        o = o + jnp.pad(gate * o_arch, ((0, 0), (0, H - Ha), (0, 0), (0, 0)))
        if want_stats:
            st["gate"] = jnp.mean(jnp.abs(gate))

    ex = {}
    if want_cache:
        ex["conv"] = raw[:, :, L - (K - 1):, :]                                      # (3,B,K-1,P)
        ex["hc"] = hc
        if Ha:
            ex["S"] = S
            if Wb > 0:
                tail = lambda t: t[:, :Ha, L - Wb:]
                h0w = jnp.broadcast_to(h_init[None], (B, Ha, D, D))
                _, A_next = be.call(*map(tail, args), h0w)
            else:
                A_next = jnp.broadcast_to(h_init[None], (B, Ha, D, D))
            ex["A"] = A_next

    o = o.transpose(0, 2, 1, 3).reshape(B, L, P)
    o = rmsnorm(o, p["onorm"])
    y = (o * jax.nn.silu(h @ p["wog"])) @ p["wo"]
    return y, ex, st
