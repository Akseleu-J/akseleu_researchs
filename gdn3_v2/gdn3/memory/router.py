import jax
import jax.numpy as jnp
from ..precision import einsum
from .. import counters as ct

def select(scores,valid,ids,k,null_logit):
    """Deterministic score-descending / stable counter-ID ascending selection."""
    scores=jnp.where(valid,scores,-jnp.inf)
    keys=tuple(jnp.broadcast_to(ids[...,i],scores.shape) for i in range(ids.shape[-1]))+(-scores,)
    order=jnp.lexsort(keys,axis=-1)[...,:min(k,scores.shape[-1])]
    picked=jnp.take_along_axis(scores,order,-1)
    logits=jnp.concatenate((picked,jnp.broadcast_to(null_logit,picked.shape[:-1]+(1,))),-1)
    weights=jax.nn.softmax(logits,-1)
    return order,weights[...,:-1],weights[...,-1]

def read(q,banks,p,cfg,position,level_weights,partition_weights=None):
    # q[H,D], banks[N,H,P,D,V]. Selection scores use a small descriptor, full state only for selected IDs.
    count=banks["valid"].shape[0]
    if cfg.bank_decay=="aligned":qbank=q[None]*jnp.exp(banks["decay"])
    else:qbank=jnp.broadcast_to(q[None],(count,)+q.shape)
    if partition_weights is None:partition_weights=jnp.ones(banks["norm"].shape[1:])/banks["norm"].shape[-1]
    sketch=jnp.sum(banks["sketch"]*partition_weights[None,...,None,None],axis=2)
    rd=einsum("nhd,nhdm->nhm",qbank,sketch,cfg.precision.dot)
    energy=jnp.sum(rd*rd,-1).T
    # Triangle norm bound is exact for one partition; declared conservative SSE score.
    norm=jnp.square(jnp.sum(jnp.sqrt(jnp.maximum(banks["norm"],0))*jnp.abs(partition_weights)[None],axis=2)).T
    qnorm=jnp.sum(qbank*qbank,-1).T
    score=jax.nn.softplus(p["route_tau"])[...,None]*jnp.log(jnp.maximum(energy*cfg.d_key/(norm*qnorm+1e-12),1e-12))
    age=ct.log1p(ct.sub(position,banks["end"]))
    score=score-jax.nn.softplus(p["route_age"])[...,None]*age[None]
    idx,weight,null=select(score,banks["valid"][None],banks["end"],cfg.top_k,p["route_null"][...,None])
    # [H,N,D,V] -> [H,k,D,V]. Shared indices are used for both dense and sparse semantics.
    from .bank import tensor
    selected={}
    for name in ("state","scale","left","right"):
        data=jnp.swapaxes(banks[name],0,1)  # H,N,P,...
        take=idx.reshape(idx.shape+(1,)*(data.ndim-2))
        selected[name]=jnp.take_along_axis(data,take,axis=1)
    matrices=tensor(selected)
    chosen=jnp.sum(matrices*partition_weights[:,None,:,None,None],axis=2)
    qh=qbank.transpose(1,0,2)
    chosen_q=jnp.take_along_axis(qh,idx[...,None],axis=1)
    val=einsum("hkd,hkdv->hkv",chosen_q,chosen,cfg.precision.dot)
    lam=jnp.take_along_axis(level_weights,idx,axis=1)
    out=jnp.sum(val*(weight*lam)[...,None],axis=1)*(cfg.d_key**-0.5)
    return out,dict(indices=idx,weights=weight,null_mass=null)
