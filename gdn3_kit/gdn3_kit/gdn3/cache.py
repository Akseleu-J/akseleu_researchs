"""Streaming inference: O(1) recurrent step + a growing / budgeted archive of bank matrices.

Per-layer cache (all arrays batch-leading):
  conv (3,B,K-1,P)  last K-1 pre-conv q/k/v projections
  hc   (B,H,D,D)    cumulative state (gdn2/hybrid/snap; cumulative heads of local)
  A    (B,Ha,D,D)   state of the CURRENT independent segment (local/hybrid)
  Bw   (B,Ha,D,D)   warm-up state of the NEXT segment (local with warm > 0)
  S    (B,Ha,N,D,D) archive banks (N = max_banks slots)         T (B,Ha,N,D,m) sketches  Sf (B,Ha,N) ||S||_F^2
  seg  (N,) int32   segment id of each slot (-1 = empty)        cnt () int32  number of filled slots
Global: pos (int32) = number of tokens processed.

The archive is DYNAMIC: a bank is appended every `seg` tokens.  If max_banks is exceeded the policy decides:
  fifo  - drop the oldest bank
  merge - add the two oldest banks into one 'sink' bank (superposition; lossy but keeps a trace of everything)
"""
from __future__ import annotations
import math

import jax
import jax.numpy as jnp

from .backend import gdn_step
from .mixer import rmsnorm, l2n, token_features
from .routing import prec, make_sketch, sketch_bank, frob2, log_ratio, bank_gate


# ----------------------------------------------------------------------------- archive helpers
def finalize_archive(S, R):
    Sf = frob2(S)
    T = sketch_bank(S, R) if R is not None else jnp.zeros(S.shape[:3] + (1, 1), S.dtype)
    return T, Sf


def fill_archive(S, N: int, policy: str):
    """Fit n prefill banks S (B,Ha,n,D,D) into N slots (static python logic). Returns (S_N, seg, cnt)."""
    n = S.shape[2]
    segs = list(range(n))
    if n > N:
        if policy == "fifo":
            S = S[:, :, n - N:]
            segs = segs[n - N:]
        else:
            sink = jnp.sum(S[:, :, :n - N + 1], axis=2, keepdims=True)
            S = jnp.concatenate([sink, S[:, :, n - N + 1:]], axis=2)
            segs = [segs[n - N]] + segs[n - N + 1:]
        n = N
    pad = N - n
    seg = jnp.asarray(segs + [-1] * pad, jnp.int32)
    if pad:
        S = jnp.pad(S, ((0, 0), (0, 0), (0, pad), (0, 0), (0, 0)))
    return S, seg, jnp.int32(n)


def append_bank(S, seg, cnt, S_new, seg_new, policy: str):
    """Append one closed bank (jit-friendly; shapes static)."""
    N = S.shape[2]

    def free(_):
        S2 = jax.lax.dynamic_update_slice(S, S_new[:, :, None], (0, 0, cnt, 0, 0))
        return S2, seg.at[cnt].set(seg_new), cnt + 1

    def full(_):
        if N == 1:
            S2 = S + S_new[:, :, None] if policy == "merge" else S_new[:, :, None]
            return S2, seg.at[0].set(seg_new), cnt
        if policy == "fifo":
            S2 = jnp.concatenate([S[:, :, 1:], S_new[:, :, None]], axis=2)
            seg2 = jnp.concatenate([seg[1:], seg_new[None]])
        else:
            sink = S[:, :, 0] + S[:, :, 1]
            S2 = jnp.concatenate([sink[:, :, None], S[:, :, 2:], S_new[:, :, None]], axis=2)
            seg2 = jnp.concatenate([seg[1:2], seg[2:], seg_new[None]])
        return S2, seg2, cnt

    return jax.lax.cond(cnt < N, free, full, None)


# ----------------------------------------------------------------------------- init / build
def init_layer_cache(p, cfg, B: int, N: int):
    H, D, P, K, Ha = cfg.n_heads, cfg.d_head, cfg.proj, cfg.d_conv, cfg.ha
    z = lambda *s: jnp.zeros(s, jnp.float32)
    c = dict(conv=z(3, B, K - 1, P), hc=z(B, H, D, D))
    if Ha:
        m = cfg.sketch_m or 1
        hi = jnp.broadcast_to(p["h_init"][None], (B, Ha, D, D))
        c.update(A=hi, Bw=hi, S=z(B, Ha, N, D, D),
                 T=z(B, Ha, N, D, m) if cfg.sketch_m else z(B, Ha, N, 1, 1),
                 Sf=z(B, Ha, N), seg=-jnp.ones((N,), jnp.int32), cnt=jnp.int32(0))
    else:
        c.update(A=z(B, 0, D, D), Bw=z(B, 0, D, D), S=z(B, 0, 1, D, D), T=z(B, 0, 1, 1, 1),
                 Sf=z(B, 0, 1), seg=-jnp.ones((1,), jnp.int32), cnt=jnp.int32(0))
    return c


def init_cache(params, cfg, B: int, max_banks: int):
    return dict(pos=jnp.int32(0),
                layers=[init_layer_cache(p, cfg, B, max_banks) for p in params["layers"]])


def build_layer_cache(ex, p, cfg, B: int, N: int):
    """Turn the parallel-forward exports of one layer into a streaming cache (prefix length = multiple of seg)."""
    c = init_layer_cache(p, cfg, B, N)
    c["conv"] = ex["conv"]
    c["hc"] = ex["hc"]
    if cfg.ha:
        R = make_sketch(cfg.d_head, cfg.sketch_m)
        S, seg, cnt = fill_archive(ex["S"], N, cfg.bank_policy)
        T, Sf = finalize_archive(S, R)
        c.update(A=ex["A"], Bw=jnp.broadcast_to(p["h_init"][None], ex["A"].shape), S=S, T=T, Sf=Sf, seg=seg, cnt=cnt)
    return c


# ----------------------------------------------------------------------------- step
def step_read(q, c, p, cfg, pos, sparse: bool = True):
    """Archive read for ONE token. q (B,Ha,D) -> (B,Ha,D)."""
    B, Ha, D = q.shape
    C = cfg.seg
    pr = prec(cfg.precision)
    R = make_sketch(D, cfg.sketch_m)
    S = c["S"]
    T = c["T"] if R is not None else S
    r = jnp.einsum("bhd,bhndm->bhnm", q, T, precision=pr)
    e = jnp.sum(r * r, axis=-1)                                           # (B,Ha,N)
    logr = log_ratio(e, c["Sf"], jnp.sum(q * q, -1)[..., None], D)
    cur = pos // C
    age = jnp.maximum(cur - c["seg"] - 1, 0).astype(jnp.float32)          # (N,)
    valid = c["seg"] >= 0
    tau = p["route_tau"][None, :, None]
    lam = p["route_lam"][None, :, None]
    scores = tau * logr - lam * jnp.log1p(age)
    a = bank_gate(scores, valid, tau * p["route_null"][None, :, None], cfg.top_k)    # (B,Ha,N)
    scale = 1.0 / math.sqrt(D)
    N = S.shape[2]
    if sparse and cfg.top_k < N:
        w_, idx = jax.lax.top_k(a, cfg.top_k)                              # (B,Ha,k)
        Ssel = jnp.take_along_axis(S, idx[..., None, None], axis=2)        # (B,Ha,k,D,D)
        rd = jnp.einsum("bhd,bhkde->bhke", q, Ssel, precision=pr)
        return scale * jnp.einsum("bhk,bhke->bhe", w_, rd)
    rd = jnp.einsum("bhd,bhnde->bhne", q, S, precision=pr)
    return scale * jnp.einsum("bhn,bhne->bhe", a, rd)


def mixer_step(p, c, x, pos, cfg, sparse: bool = True):
    """x (B,d) one token -> (y (B,d), new layer cache)."""
    B = x.shape[0]
    H, D, P, K, C = cfg.n_heads, cfg.d_head, cfg.proj, cfg.d_conv, cfg.seg
    Ha, Wb, mode = cfg.ha, cfg.bank_warm, cfg.mode
    scale = 1.0 / math.sqrt(D)

    h = rmsnorm(x, p["norm1"])
    raw = jnp.stack([h @ p["wq"], h @ p["wk"], h @ p["wv"]])                       # (3,B,P)
    buf = jnp.concatenate([c["conv"], raw[:, :, None, :]], axis=2)                 # (3,B,K,P)
    conv = jnp.einsum("sbkp,spk->sbp", buf, p["conv_w"]) + p["conv_b"][:, None, :]
    act = jax.nn.silu(conv).reshape(3, B, H, D)
    q, k, v = l2n(act[0]), l2n(act[1]), act[2]
    beta, wg, g = token_features(p, h, cfg)                                         # (B,H,D)
    tok = (q, k, v, wg, beta, g)

    nc = dict(c)
    nc["conv"] = buf[:, :, 1:]
    u = pos % C
    hc = c["hc"]
    orst = None

    if mode in ("gdn2", "hybrid", "snap"):
        hc2, o = gdn_step(hc, *tok, scale)
    else:  # local: heads [Ha:] cumulative, heads [:Ha] use the independent state
        if Ha < H:
            hr, orst = gdn_step(hc[:, Ha:], *(t[:, Ha:] for t in tok), scale)
            hc2 = jnp.concatenate([hc[:, :Ha], hr], axis=1)
        else:
            hc2 = hc
        o = None
    nc["hc"] = hc2

    if Ha:
        tokA = tuple(t[:, :Ha] for t in tok)
        A2, oA = gdn_step(c["A"], *tokA, scale)
        hinit = jnp.broadcast_to(p["h_init"][None], A2.shape)
        if Wb > 0:
            Bin = jnp.where(u == C - Wb, hinit, c["Bw"])
            B2, _ = gdn_step(Bin, *tokA, scale)
            B2 = jnp.where(u >= C - Wb, B2, c["Bw"])
        else:
            B2 = c["Bw"]
        if mode == "local":
            o = oA if Ha == H else jnp.concatenate([oA, orst], axis=1)

        o_arch = step_read(q[:, :Ha], c, p, cfg, pos, sparse)
        gate = jnp.tanh(p["arch_gate"])[None, :, None]
        o = o + jnp.pad(gate * o_arch, ((0, 0), (0, H - Ha), (0, 0)))

        closed = u == C - 1
        bank = hc2[:, :Ha] if mode == "snap" else A2
        nc["A"] = jnp.where(closed, B2 if Wb > 0 else hinit, A2)
        nc["Bw"] = jnp.where(closed, hinit, B2)
        R = make_sketch(D, cfg.sketch_m)
        seg_new = (pos // C).astype(jnp.int32)

        def do_app(_):
            S2, seg2, cnt2 = append_bank(c["S"], c["seg"], c["cnt"], bank, seg_new, cfg.bank_policy)
            T2, Sf2 = finalize_archive(S2, R)
            return S2, seg2, cnt2, T2, Sf2

        def no_app(_):
            return c["S"], c["seg"], c["cnt"], c["T"], c["Sf"]

        S2, seg2, cnt2, T2, Sf2 = jax.lax.cond(closed, do_app, no_app, None)
        nc.update(S=S2, seg=seg2, cnt=cnt2, T=T2, Sf=Sf2)

    o = o.reshape(B, P)
    o = rmsnorm(o, p["onorm"])
    y = (o * jax.nn.silu(h @ p["wog"])) @ p["wo"]
    return y, nc


def layer_step(p, c, x, pos, cfg, sparse: bool = True):
    y, nc = mixer_step(p, c, x, pos, cfg, sparse)
    x = x + y
    h2 = rmsnorm(x, p["norm2"])
    x = x + (jax.nn.silu(h2 @ p["w_gate"]) * (h2 @ p["w_up"])) @ p["w_down"]
    return x, nc


def decode_step(params, cache, tok, cfg, sparse: bool = True):
    """tok (B,) int -> (logits (B,V), new cache)."""
    x = params["embed"][tok]
    pos = cache["pos"]
    layers = []
    for p, c in zip(params["layers"], cache["layers"]):
        x, nc = layer_step(p, c, x, pos, cfg, sparse)
        layers.append(nc)
    logits = rmsnorm(x, params["final_norm"]) @ params["embed"].T
    return logits, dict(pos=pos + 1, layers=layers)


def prefill(params, ids, cfg, be, max_banks: int, sparse: bool = True):
    """Process a prompt: parallel over the aligned prefix, recurrent steps over the (< align) tail.
    Returns (logits (B,L,V), cache)."""
    from .model import forward
    B, L = ids.shape
    al = cfg.align(be.bt)
    La = (L // al) * al
    outs = []
    if La > 0:
        logits_a, aux = forward(params, ids[:, :La], cfg, be, want_cache=True)
        layers = [build_layer_cache(ex, p, cfg, B, max_banks) for ex, p in zip(aux["extras"], params["layers"])]
        cache = dict(pos=jnp.int32(La), layers=layers)
        outs.append(logits_a)
    else:
        cache = init_cache(params, cfg, B, max_banks)
    if La < L:
        def body(cch, tok):
            lg, c2 = decode_step(params, cch, tok, cfg, sparse)
            return c2, lg
        cache, lg = jax.lax.scan(body, cache, ids[:, La:].T)
        outs.append(lg.transpose(1, 0, 2))
    return jnp.concatenate(outs, axis=1), cache
