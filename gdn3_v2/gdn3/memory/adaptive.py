import jax
import jax.numpy as jnp
from ..precision import linear

def weights(x,p,levels,logspans,cfg):
    H=len(cfg.heads)
    if cfg.lambda_mode=="off": return jnp.ones((H,levels.shape[0]),jnp.float32)
    if cfg.lambda_mode.startswith("paper"):
        n=cfg.lambda_paper_levels
        d=linear(x,p["lambda_proj"],cfg.precision.dot).reshape(H,n)
        h=linear(d,p["lambda_w1"],cfg.precision.dot)
        if cfg.lambda_mode=="paper_gelu": h=jax.nn.gelu(h)
        all_weights=jax.nn.softplus(linear(h,p["lambda_w2"],cfg.precision.dot)+p["lambda_bias"])
        return jnp.where(levels[None]>=0,jnp.take(all_weights,jnp.maximum(levels,0),axis=-1),1.0)
    h=linear(x,p["lambda_x"],cfg.precision.dot)[None,None,:]+p["lambda_head"][:,None,:]
    l=levels.astype(jnp.float32)
    features=jnp.stack((jnp.sin(l),jnp.cos(l),jnp.tanh(logspans/32.0),jnp.sin(logspans/8.0)),axis=-1)
    h=jax.nn.silu(h+linear(features,p["lambda_level"],cfg.precision.dot)[None])
    y=jax.nn.softplus(einsum_last(h,p["lambda_out"],cfg)+p["lambda_bias"])
    return jnp.where(levels[None]>=0,y,1.0)

def einsum_last(h,w,cfg):
    from ..precision import einsum
    return einsum("hnf,hf->hn",h,w,cfg.precision.dot)
