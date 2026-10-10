"""GDN-3 layer pieces (functional, params = plain dict pytree).
  init_layer   parameters (backbone + cell extras + archive/router + dla/hola/ccq/adaptive-lambda extras)
  layer_pre    x -> features (B,L,H,D) with explicit conv state (one code path for batch and streaming)
  substeps     cell -> list of delta sub-steps per token (SAME function feeds the scan engine and the kernel interleaving)
  layer_post   output norm / gate / proj + SwiGLU MLP
"""
import math
import jax
import jax.numpy as jnp

from . import ops

LN2 = math.log(2.0)


def rmsnorm(x, scale, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * scale


def l2n(t, eps=1e-6):
    return t * jax.lax.rsqrt(jnp.sum(t * t, -1, keepdims=True) + eps)


def _lin(key, din, dout, gain=1.0):
    return jax.random.normal(key, (din, dout), jnp.float32) * (gain / math.sqrt(din))


def decay_init(cfg):
    """log-rate grid: exp(a)*softplus(0) = ln2/half_life, half-lives log-spaced hl_min..hl_max (per channel)."""
    D, H = cfg.d_head, cfg.n_heads
    if cfg.decay_mode == "head":
        return jnp.full((H,), math.log(cfg.decay_init / LN2), jnp.float32)
    lam = jnp.exp(jnp.linspace(math.log(LN2 / cfg.hl_min), math.log(LN2 / cfg.hl_max), D))
    return jnp.broadcast_to(jnp.log(lam / LN2)[None, :], (H, D)).astype(jnp.float32)


def init_layer(key, cfg):
    d, P, H, D, K, Ha = cfg.d_model, cfg.proj, cfg.n_heads, cfg.d_head, cfg.d_conv, cfg.ha
    ks = iter(jax.random.split(key, 40))
    og = 1.0 / math.sqrt(2 * cfg.n_layers)
    conv_w = jnp.zeros((3, P, K), jnp.float32).at[:, :, K - 1].set(1.0) + 0.02 * jax.random.normal(next(ks), (3, P, K))
    dd = P if cfg.decay_mode == "kda" else H
    p = dict(norm1=jnp.ones((d,)), norm2=jnp.ones((d,)), mix_norm=jnp.ones((P,)),
             wq=_lin(next(ks), d, P), wk=_lin(next(ks), d, P), wv=_lin(next(ks), d, P),
             conv_w=conv_w, conv_b=jnp.zeros((3, P)),
             wb=_lin(next(ks), d, P, 0.5), bb=jnp.zeros((P,)), ww=_lin(next(ks), d, P, 0.5), bw=jnp.zeros((P,)),
             wd=_lin(next(ks), d, dd, 0.1), bd=jnp.zeros((dd,)), decay_a=decay_init(cfg),
             wog=_lin(next(ks), d, P), wo=_lin(next(ks), P, d, og),
             w_gate=_lin(next(ks), d, cfg.mlp_mult * d), w_up=_lin(next(ks), d, cfg.mlp_mult * d),
             w_down=_lin(next(ks), cfg.mlp_mult * d, d, og))
    if cfg.cell in ("qed", "eda"):
        p["wgam"] = _lin(next(ks), d, H, 0.1)
        p["bgam"] = jnp.full((H,), cfg.gamma_init)
    if cfg.cell == "eda":
        p["we"] = _lin(next(ks), d, P)
    if cfg.cell == "dp":
        E = cfg.dp_extra
        p.update(wke=_lin(next(ks), d, E * P), wve=_lin(next(ks), d, E * P),
                 wbe=_lin(next(ks), d, E * P, 0.5), bbe=jnp.zeros((E * P,)))
    if cfg.cell == "osdn":
        p["log_eta"] = jnp.full((H,), math.log(cfg.osdn_eta))
    if Ha:
        p.update(h_init=ops.init_h_init(next(ks), Ha, D, cfg.h_init_scale) if cfg.h_init_scale else jnp.zeros((Ha, D, D)),
                 route_tau=jnp.full((Ha,), cfg.tau_init), route_lam=jnp.full((Ha,), cfg.lam_init),
                 route_null=jnp.full((Ha,), cfg.null_init), arch_gate=jnp.full((Ha,), cfg.gate_init))
        if cfg.adaptive_lam:                                 # lambda_l(x) = softplus(MLP(x)) ; init ~ lam_init
            p.update(lam_w1=_lin(next(ks), d, cfg.lam_hidden), lam_b1=jnp.zeros((cfg.lam_hidden,)),
                     lam_w2=_lin(next(ks), cfg.lam_hidden, Ha * cfg.n_lv, 0.1),
                     lam_b2=jnp.full((Ha * cfg.n_lv,), math.log(math.expm1(cfg.lam_init))))
        if cfg.ccq:
            p.update(ccq_w=jnp.zeros((Ha, D)), ccq_b=jnp.full((Ha,), -2.0))
        if cfg.seg_mode == "dla":
            p["dla_theta"] = jnp.full((Ha,), cfg.dla_theta)
    if cfg.hola:
        p.update(hola_gate=jnp.full((H,), 0.1), hola_lgamma=jnp.full((H,), math.log(cfg.hola_gamma)))
    return p


def causal_conv(u, w, bias, state):
    """u (B,L,P); w (P,K); state (B,K-1,P) or None. Returns (y, new_state)."""
    B, L, P = u.shape
    K = w.shape[1]
    if state is None:
        state = jnp.zeros((B, K - 1, P), u.dtype)
    xx = jnp.concatenate([state, u], 1)
    return sum(w[:, i] * xx[:, i:i + L] for i in range(K)) + bias, xx[:, L:]


def layer_pre(p, cfg, x, conv_state=None):
    """x (B,L,d) -> (xn, feat, new_conv_state).  feat tensors are (B,L,H,D) unless noted."""
    B, L, _ = x.shape
    H, D, Ha = cfg.n_heads, cfg.d_head, cfg.ha
    xn = rmsnorm(x, p["norm1"])
    new_conv = []

    def proj(i, nm):
        y, s = causal_conv(xn @ p["w" + nm], p["conv_w"][i], p["conv_b"][i], None if conv_state is None else conv_state[i])
        new_conv.append(s)
        return jax.nn.silu(y).reshape(B, L, H, D)

    q, k, v = l2n(proj(0, "q")), l2n(proj(1, "k")), proj(2, "v")
    b = jax.nn.sigmoid(xn @ p["wb"] + p["bb"]).reshape(B, L, H, D)
    w = jax.nn.sigmoid(xn @ p["ww"] + p["bw"]).reshape(B, L, H, D)
    z = xn @ p["wd"] + p["bd"]
    if cfg.decay_mode == "kda":
        g = ops.kda_gate(z.reshape(B, L, H, D), jnp.clip(p["decay_a"], -20.0, 20.0), cfg.g_max)
    else:
        rate = jnp.exp(jnp.clip(p["decay_a"], -20.0, 20.0))
        rate = rate[None, None, :, None] if rate.ndim == 1 else rate[None, None]
        g = -jnp.minimum(rate * jax.nn.softplus(z)[..., None], cfg.g_max)
    g = jnp.broadcast_to(g, (B, L, H, D))
    f = dict(q=q, k=k, v=v, w=w, b=b, g=g)
    if cfg.cell in ("qed", "eda"):
        f["gam"] = jax.nn.sigmoid(xn @ p["wgam"] + p["bgam"])                       # (B,L,H)
    if cfg.cell == "eda":
        f["e"] = l2n((xn @ p["we"]).reshape(B, L, H, D))
    if cfg.cell == "dp":
        E = cfg.dp_extra
        f["ke"] = l2n((xn @ p["wke"]).reshape(B, L, H, E, D))
        f["ve"] = (xn @ p["wve"]).reshape(B, L, H, E, D)
        f["be"] = jax.nn.sigmoid(xn @ p["wbe"] + p["bbe"]).reshape(B, L, H, E, D)
    if Ha and cfg.adaptive_lam:
        f["lamvec"] = ops.adaptive_lambda(xn, p["lam_w1"], p["lam_b1"], p["lam_w2"], p["lam_b2"]).reshape(B, L, Ha, cfg.n_lv)
    if Ha and cfg.ccq:
        f["ccql"] = jax.nn.sigmoid(jnp.einsum("blhd,hd->blh", q[:, :, :Ha], p["ccq_w"]) + p["ccq_b"])
    return xn, f, jnp.stack(new_conv)


def substeps(cfg, q, k, v, w, b, g, ex):
    """Delta sub-steps of one token as [(k,v,w,b,g), ...] and the index of the main write step.
    Works for any leading dims (per-token (B,H,D) or head-major sequences (B,H,L,D)); the readout is taken after the last one.
      qed : (1) erase along q_perp  = delta step with key q_perp, v=w=0, b=gamma, decay g ; (2) main delta, no decay
      eda : (1) erase along e       = delta step with key e,      v=w=0, b=gamma/|e|^2    ; (2) main delta, no decay
      dp  : main step (with decay) then dp_extra undecayed delta steps (own k_i,v_i,b_i; shared w)
      osdn: main step with preconditioned key k*d (d from the carry)"""
    zg = jnp.zeros_like(g)
    if cfg.cell == "gdn2":
        return [(k, v, w, b, g)], 0
    if cfg.cell == "osdn":
        return [(k * ex["d"], v, w, b, g)], 0
    z = jnp.zeros_like(v)
    if cfg.cell == "qed":
        qp = q - (jnp.sum(q * k, -1, keepdims=True) / (jnp.sum(k * k, -1, keepdims=True) + 1e-6)) * k
        bg = jnp.broadcast_to(ex["gam"][..., None], k.shape)
        return [(qp, z, z, bg, g), (k, v, w, b, zg)], 1
    if cfg.cell == "eda":
        e = ex["e"]
        bg = jnp.broadcast_to(ex["gam"][..., None] / (jnp.sum(e * e, -1, keepdims=True) + 1e-6), k.shape)
        return [(e, z, z, bg, g), (k, v, w, b, zg)], 1
    subs = [(k, v, w, b, g)]
    for i in range(cfg.dp_extra):
        subs.append((ex["ke"][..., i, :], ex["ve"][..., i, :], w, ex["be"][..., i, :], zg))
    return subs, 0


def token_update(h, subs):
    """Apply the sub-steps sequentially; returns (h_new, v_new of the FIRST step, [v_new per step])."""
    vns = []
    for (k, v, w, b, g) in subs:
        h, vn = ops.gdn_core(h, None, k, v, w, b, g)
        vns.append(vn)
    return h, vns


def interleave(cfg, f):
    """Head-major features (B,H,L,D) -> kernel streams (B,H,L*m,D): sub-steps of a token are adjacent."""
    extra = {k_: f[k_] for k_ in ("gam", "e", "ke", "ve", "be") if k_ in f}
    subs, _ = substeps(cfg, f["q"], f["k"], f["v"], f["w"], f["b"], f["g"], extra)
    B, H, L, D = f["q"].shape
    st = lambda i: jnp.stack([s[i] for s in subs], axis=3).reshape(B, H, L * len(subs), D)
    q = jnp.stack([f["q"]] * len(subs), axis=3).reshape(B, H, L * len(subs), D)
    return q, st(0), st(1), st(2), st(3), st(4)


def layer_post(p, cfg, x, xn, o):
    """o (B,L,H,D) -> new residual stream."""
    B, L = o.shape[:2]
    o = rmsnorm(o.reshape(B, L, cfg.proj), p["mix_norm"])
    x = x + (o * jax.nn.silu(xn @ p["wog"])) @ p["wo"]
    h2 = rmsnorm(x, p["norm2"])
    return x + (jax.nn.silu(h2 @ p["w_gate"]) * (h2 @ p["w_up"])) @ p["w_down"]
