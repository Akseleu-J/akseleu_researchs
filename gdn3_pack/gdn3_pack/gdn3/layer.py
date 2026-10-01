"""GDN-3 layer = GDN-2 layer (+ optional bank archive). Functional JAX (params are plain dict pytrees).

  layer_pre   : x -> q,k,v,w,beta,g (head-major) with explicit conv state (so batch and streaming share one code path)
  mixer_batch : exact GDN-2 over the whole sequence (main stream) + [banks: n independent segment states, content-routed read]
  layer_post  : output norm/gate/proj + SwiGLU MLP
With banks disabled (or gate==0) the layer is exactly GDN-2.
"""
import math
import jax, jax.numpy as jnp
from .segments import fold, unfold_states
from .read import read_banks
from . import landmarks as lm


def rmsnorm(x, scale, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * scale


def _dense(key, din, dout, bias=False, mult=1.0):
    p = {"w": jax.random.normal(key, (din, dout), jnp.float32) * (mult / math.sqrt(din))}
    if bias:
        p["b"] = jnp.zeros((dout,), jnp.float32)
    return p


def _lin(p, x):
    y = x @ p["w"]
    return y + p["b"] if "b" in p else y


def _conv_init(key, dim, K):
    w = 0.02 * jax.random.normal(key, (dim, K), jnp.float32)
    return {"w": w.at[:, -1].add(1.0), "b": jnp.zeros((dim,), jnp.float32)}   # ~identity at init


def init_layer(key, cfg, banks):
    ks = jax.random.split(key, 16); d, P, H, D = cfg.d_model, cfg.P, cfg.n_heads, cfg.d_head
    p = {"norm1": jnp.ones((d,)), "norm2": jnp.ones((d,)), "mix_norm": jnp.ones((P,)),
         "q": _dense(ks[0], d, P), "k": _dense(ks[1], d, P), "v": _dense(ks[2], d, P),
         "conv_q": _conv_init(ks[3], P, cfg.d_conv), "conv_k": _conv_init(ks[4], P, cfg.d_conv),
         "conv_v": _conv_init(ks[5], P, cfg.d_conv),
         "erase": _dense(ks[6], d, P, True), "write": _dense(ks[7], d, P, True),
         "decay_proj": _dense(ks[8], d, H, True, mult=0.1),
         "out_gate": _dense(ks[9], d, P), "out": _dense(ks[10], P, d, mult=1.0 / math.sqrt(2 * cfg.n_layers)),
         "gate_proj": _dense(ks[11], d, cfg.mlp_mult * d), "up_proj": _dense(ks[12], d, cfg.mlp_mult * d),
         "down_proj": _dense(ks[13], cfg.mlp_mult * d, d, mult=1.0 / math.sqrt(2 * cfg.n_layers))}
    if cfg.decay_mode == "head":
        p["decay_a"] = jnp.full((H,), math.log(math.expm1(cfg.decay_init)), jnp.float32)
    else:
        lam = [math.exp(math.log(math.log(2) / cfg.hl_min) + (math.log(math.log(2) / cfg.hl_max) -
               math.log(math.log(2) / cfg.hl_min)) * i / max(D - 1, 1)) for i in range(D)]
        p["decay_a"] = jnp.broadcast_to(jnp.log(jnp.asarray(lam, jnp.float32) / 0.6931471805599453), (H, D))
    if banks:
        p["log_tau"] = jnp.full((H,), math.log(cfg.tau_init), jnp.float32)
        p["theta"] = jnp.full((H,), cfg.theta_init, jnp.float32)
        p["gate_raw"] = jnp.full((H,), math.atanh(min(max(cfg.gate_init, 0.0), 0.999)), jnp.float32)
        if cfg.n_landmarks > 0:
            p["log_sharp"] = jnp.full((H,), math.log(cfg.landmark_sharp), jnp.float32)
            p["lnull"] = jnp.full((H,), cfg.landmark_null, jnp.float32)
            p["lgate_raw"] = jnp.full((H,), math.atanh(0.1), jnp.float32)
    return p


def causal_conv(u, w, bias, state):
    """u (B,L,P); w (P,K); state (B,K-1,P) or None. Returns (y, new_state)."""
    B, L, Pd = u.shape; K = w.shape[1]
    if state is None:
        state = jnp.zeros((B, K - 1, Pd), u.dtype)
    xx = jnp.concatenate([state, u], 1)
    y = sum(w[:, i] * xx[:, i:i + L] for i in range(K)) + bias
    return y, xx[:, L:]


def _l2(t):
    return t * jax.lax.rsqrt(jnp.sum(t * t, -1, keepdims=True) + 1e-6)


def layer_pre(p, cfg, x, conv_state=None):
    B, L, _ = x.shape; H, D = cfg.n_heads, cfg.d_head
    xn = rmsnorm(x, p["norm1"]) if cfg.pre_norm else x
    st = conv_state or {}
    new = {}

    def proj(nm):
        y, new[nm] = causal_conv(_lin(p[nm], xn), p["conv_" + nm]["w"], p["conv_" + nm]["b"], st.get(nm))
        return jax.nn.silu(y).reshape(B, L, H, D)

    q, k, v = proj("q"), proj("k"), proj("v")
    v = jnp.clip(v, -50.0, 50.0); q, k = _l2(q), _l2(k)
    beta = jax.nn.sigmoid(_lin(p["erase"], xn)).reshape(B, L, H, D)
    wg = jax.nn.sigmoid(_lin(p["write"], xn)).reshape(B, L, H, D)
    sp = jax.nn.softplus(_lin(p["decay_proj"], xn).astype(jnp.float32))              # (B,L,H)
    rate = jnp.exp(jnp.clip(p["decay_a"], -20.0, 20.0))
    rate = rate[None, None, :, None] if rate.ndim == 1 else rate[None, None]       # (1,1,H,1) | (1,1,H,D)
    g = -jnp.minimum(rate * sp[..., None], cfg.g_max)
    g = jnp.broadcast_to(g, (B, L, H, D))
    hm = lambda t: t.transpose(0, 2, 1, 3)
    return xn, tuple(map(hm, (q, k, v, wg, beta, g))), new


def layer_post(p, cfg, x, xn, o):
    """o (B,H,L,D) -> new residual stream."""
    B, H, L, D = o.shape
    o = rmsnorm(o.transpose(0, 2, 1, 3).reshape(B, L, H * D), p["mix_norm"])
    x = x + _lin(p["out"], o * jax.nn.silu(_lin(p["out_gate"], xn)))
    h2 = rmsnorm(x, p["norm2"])
    return x + _lin(p["down_proj"], jax.nn.silu(_lin(p["gate_proj"], h2)) * _lin(p["up_proj"], h2))


def mixer_batch(p, cfg, backend, q, k, v, w, b, g, banks, force_gate=None):
    """q..g (B,H,L,D). Main stream = exact GDN-2. Banks: independent zero-init segment states (one batched kernel call)."""
    lmk = banks and cfg.n_landmarks > 0
    out = backend(q, k, v, w, b, g, None, return_vnew=lmk)
    o = out[0]
    if not banks:
        return o
    B, H, L, D = q.shape; C = cfg.seg; n = L // C
    if n < 1:
        return o
    Ln = n * C
    kf, vf, wf, bf, gf = (fold(t[:, :, :Ln], n, C) for t in (k, v, w, b, g))
    h0 = jnp.zeros((B * n, H, D, D), jnp.float32)
    Sf = backend(jnp.zeros_like(kf), kf, vf, wf, bf, gf, h0)[1]            # only the final state of each segment is used
    S = unfold_states(Sf, B, n)
    Gtok = jnp.cumsum(g.astype(jnp.float32), axis=2)
    Gend = Gtok[:, :, :Ln].reshape(B, H, n, C, D)[:, :, :, -1]
    seg_idx = jnp.arange(L) // C
    r = read_banks(q, S, Gtok, Gend, seg_idx, jnp.exp(p["log_tau"]), p["theta"], 1.0 / math.sqrt(D),
                   cfg.bank_decay, cfg.read_precision)
    gate = jnp.tanh(p["gate_raw"]) if force_gate is None else jnp.full((H,), force_gate, jnp.float32)
    o = o + gate[None, :, None, None] * r
    if lmk:
        lk, lv = lm.pick_landmarks(k[:, :, :Ln], (w * v)[:, :, :Ln], out[2][:, :, :Ln], b[:, :, :Ln], n, C, cfg.n_landmarks)
        ro = lm.landmark_read(q, lk, lv, jnp.exp(p["log_sharp"]), p["lnull"], C)
        lg = jnp.tanh(p["lgate_raw"]) if force_gate is None else jnp.full((H,), force_gate, jnp.float32)
        o = o + lg[None, :, None, None] * ro
    return o


def layer_forward(p, cfg, backend, x, banks, force_gate=None):
    xn, (q, k, v, w, b, g), _ = layer_pre(p, cfg, x)
    o = mixer_batch(p, cfg, backend, q, k, v, w, b, g, banks, force_gate)
    return layer_post(p, cfg, x, xn, o)
