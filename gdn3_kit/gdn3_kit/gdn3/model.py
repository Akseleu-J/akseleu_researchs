"""Pure-JAX GDN-3 language model (params are a plain dict; no flax dependency)."""
from __future__ import annotations
import math

import jax
import jax.numpy as jnp

from .config import G3Config
from .mixer import mixer_fwd, rmsnorm


def _lin(key, din, dout, gain=1.0):
    return (jax.random.normal(key, (din, dout)) * (gain / math.sqrt(din))).astype(jnp.float32)


def decay_init(cfg: G3Config):
    """Per-channel log-spaced half-lives hl_min .. hl_max -> decay_a (H,D) with exp(a)*softplus(0)=ln2/hl."""
    D, H = cfg.d_head, cfg.n_heads
    lam = jnp.exp(jnp.linspace(math.log(math.log(2) / cfg.hl_min), math.log(math.log(2) / cfg.hl_max), D))
    a = jnp.log(lam / math.log(2.0))
    return jnp.broadcast_to(a[None, :], (H, D)).astype(jnp.float32)


def init_layer(key, cfg: G3Config):
    d, P, H, D, K = cfg.d_model, cfg.proj, cfg.n_heads, cfg.d_head, cfg.d_conv
    ks = jax.random.split(key, 12)
    og = 1.0 / math.sqrt(2 * cfg.n_layers)
    conv_w = jnp.zeros((3, P, K), jnp.float32).at[:, :, K - 1].set(1.0)
    conv_w = conv_w + 0.02 * jax.random.normal(ks[8], (3, P, K))
    p = dict(
        norm1=jnp.ones((d,)), norm2=jnp.ones((d,)),
        wq=_lin(ks[0], d, P), wk=_lin(ks[1], d, P), wv=_lin(ks[2], d, P),
        conv_w=conv_w, conv_b=jnp.zeros((3, P)),
        wb=_lin(ks[3], d, P, 0.5), bb=jnp.zeros((P,)),
        ww=_lin(ks[4], d, P, 0.5), bw=jnp.zeros((P,)),
        wd=_lin(ks[5], d, H, 0.1), bd=jnp.zeros((H,)),
        decay_a=decay_init(cfg),
        wog=_lin(ks[6], d, P), onorm=jnp.ones((P,)), wo=_lin(ks[7], P, d, og),
        w_gate=_lin(ks[9], d, cfg.mlp_mult * d), w_up=_lin(ks[10], d, cfg.mlp_mult * d),
        w_down=_lin(ks[11], cfg.mlp_mult * d, d, og),
    )
    Ha = cfg.ha
    if Ha:
        p.update(h_init=jnp.zeros((Ha, D, D)), route_tau=jnp.full((Ha,), cfg.tau_init),
                 route_lam=jnp.full((Ha,), cfg.lam_init), route_null=jnp.full((Ha,), cfg.null_init),
                 arch_gate=jnp.full((Ha,), cfg.gate_init))
    return {k: v.astype(jnp.float32) for k, v in p.items()}


def init_params(key, cfg: G3Config):
    ks = jax.random.split(key, cfg.n_layers + 1)
    return dict(embed=(jax.random.normal(ks[0], (cfg.vocab, cfg.d_model)) * 0.02).astype(jnp.float32),
                layers=[init_layer(ks[i + 1], cfg) for i in range(cfg.n_layers)],
                final_norm=jnp.ones((cfg.d_model,), jnp.float32))


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def layer_fwd(p, x, cfg, be, oracle=None, want_cache=False, want_stats=False):
    y, ex, st = mixer_fwd(p, x, cfg, be, oracle=oracle, want_cache=want_cache, want_stats=want_stats)
    x = x + y
    h2 = rmsnorm(x, p["norm2"])
    x = x + (jax.nn.silu(h2 @ p["w_gate"]) * (h2 @ p["w_up"])) @ p["w_down"]
    return x, ex, st


def forward(params, ids, cfg: G3Config, be, oracle=None, want_cache=False, want_stats=False):
    """ids (B,L) int -> (logits (B,L,V), aux).  Any L: it is right-padded to cfg.align(be.bt) (causal => the padded
    tail cannot influence real positions) and cropped.  The number of banks n = ceil(L / seg) is dynamic."""
    cfg.check_backend(be.bt)
    B, L0 = ids.shape
    al = cfg.align(be.bt)
    Lp = -(-L0 // al) * al
    if Lp != L0:
        if want_cache:
            raise ValueError("want_cache requires L to be a multiple of the alignment")
        ids = jnp.pad(ids, ((0, 0), (0, Lp - L0)), constant_values=cfg.pad_id)
        if oracle is not None:
            oracle = jnp.pad(oracle, ((0, 0), (0, Lp - L0), (0, 0)))
    x = params["embed"][ids]
    extras, stats = [], []
    use_remat = cfg.remat and not (want_cache or want_stats)
    for p in params["layers"]:
        if use_remat:
            x = jax.checkpoint(lambda p_, x_: layer_fwd(p_, x_, cfg, be, oracle)[0])(p, x)
        else:
            x, ex, st = layer_fwd(p, x, cfg, be, oracle, want_cache, want_stats)
            extras.append(ex)
            stats.append(st)
    logits = rmsnorm(x, params["final_norm"]) @ params["embed"].T
    return logits[:, :L0], dict(extras=extras, stats=stats)


def logits_fn(params, ids, cfg, be, oracle=None):
    return forward(params, ids, cfg, be, oracle)[0]


# ----------------------------------------------------------------------------- losses
def masked_ce(logits, ids, mask=None):
    """Next-token cross-entropy (+accuracy). logits (B,L,V) predict ids[:,1:]; mask (B,L) marks TARGET positions."""
    lg = logits[:, :-1].astype(jnp.float32)
    tgt = ids[:, 1:]
    lse = jax.nn.logsumexp(lg, axis=-1)
    nll = lse - jnp.take_along_axis(lg, tgt[..., None], axis=-1)[..., 0]
    hit = (jnp.argmax(lg, axis=-1) == tgt)
    if mask is None:
        return jnp.mean(nll), jnp.mean(hit.astype(jnp.float32))
    m = mask[:, 1:].astype(jnp.float32)
    den = jnp.maximum(m.sum(), 1.0)
    return (nll * m).sum() / den, (hit * m).sum() / den


def nll_by_position(logits, ids):
    """(B,L-1) token NLL, for loss-vs-position curves."""
    lg = logits[:, :-1].astype(jnp.float32)
    tgt = ids[:, 1:]
    return jax.nn.logsumexp(lg, axis=-1) - jnp.take_along_axis(lg, tgt[..., None], axis=-1)[..., 0]
