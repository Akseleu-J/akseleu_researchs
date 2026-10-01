"""Byte/BPE language model: embedding -> n_layers x (GDN-2 layer [+ banks]) -> RMSNorm -> tied head."""
import math
import jax, jax.numpy as jnp
from .layer import init_layer, layer_forward, rmsnorm


def init_model(key, cfg):
    ks = jax.random.split(key, cfg.n_layers + 1)
    return {"embed": jax.random.normal(ks[0], (cfg.vocab, cfg.d_model), jnp.float32) / math.sqrt(cfg.d_model),
            "final_norm": jnp.ones((cfg.d_model,), jnp.float32),
            "layers": [init_layer(ks[i + 1], cfg, cfg.has_banks(i)) for i in range(cfg.n_layers)]}


def count_params(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def forward(params, cfg, backend, tokens, force_gate=None):
    x = params["embed"][tokens]
    for i, p in enumerate(params["layers"]):
        banks = cfg.has_banks(i) and ("log_tau" in p)
        f = lambda p_, x_, banks=banks: layer_forward(p_, cfg, backend, x_, banks, force_gate)
        x = jax.checkpoint(f)(p, x) if cfg.remat else f(p, x)
    return rmsnorm(x, params["final_norm"]) @ params["embed"].T


def xent(logits, targets):
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return -jnp.take_along_axis(lp, targets[..., None], -1)[..., 0]


def lm_loss(params, cfg, backend, ids):
    return xent(forward(params, cfg, backend, ids)[:, :-1], ids[:, 1:]).mean()
