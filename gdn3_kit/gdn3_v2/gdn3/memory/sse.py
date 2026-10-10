import jax
import jax.numpy as jnp

def select(logits,k,shared=True):
    e=jax.nn.softmax(logits,axis=-1)
    count=e.shape[-1]
    mask=jnp.zeros_like(e,dtype=bool)
    if shared:
        mask=mask.at[...,0].set(True)
        candidates=e.at[...,0].set(-jnp.inf)
        take=k-1
    else: candidates=e;take=k
    if take:
        order=jnp.argsort(-candidates,axis=-1,stable=True)[...,:take]
        mask=mask|jnp.any(jnp.arange(count)==order[...,None],axis=-2)
    return e,mask

def expand(features,weights,mask,cfg):
    """Archive adapter: row-softmax addresses, write u scaled by selected partition gate."""
    out={}
    N=weights.shape[-1]
    for name,value in features.items():
        out[name]=jnp.broadcast_to(value[:,None],(value.shape[0],N)+value.shape[1:])
    if N>1:
        out["k"]=jax.nn.softmax(out["k_logits"],axis=-1)
        out["u_scale"]=weights*mask
        if cfg.sse_inactive=="freeze":out["g"]=out["g"]*mask[...,None]
    else:out["u_scale"]=jnp.ones_like(weights)
    return out

def balance(weights,mask):
    return weights.shape[-1]*jnp.sum(jnp.mean(mask.astype(jnp.float32),axis=0)*jnp.mean(weights,axis=0))
