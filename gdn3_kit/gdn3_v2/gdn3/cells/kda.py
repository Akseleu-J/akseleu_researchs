import jax
import jax.numpy as jnp
from ..precision import linear

def gate(x,p,cfg):
    raw=linear(x,p["wd"],cfg.precision.dot)+p["bd"]
    if cfg.decay_mode=="kda_lowrank": raw=linear(jax.nn.silu(raw),p["wd_up"],cfg.precision.dot)
    if cfg.decay_mode in {"head","channel"}: raw=raw.reshape(x.shape[:-1]+(cfg.n_heads,1))
    else: raw=raw.reshape(x.shape[:-1]+(cfg.n_heads,cfg.d_key))
    if cfg.decay_mode=="atomic": return -jnp.minimum(jax.nn.softplus(raw),cfg.g_max)
    rate=jnp.exp(p["decay_a"])
    if cfg.decay_mode=="head": rate=jnp.mean(rate,-1,keepdims=True)
    return -jnp.minimum(rate*jax.nn.softplus(raw),cfg.g_max)
