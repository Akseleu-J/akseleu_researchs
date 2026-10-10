"""Window-buffered recurrent quantization with low-rank compensation.

The portable implementation materializes the state during reads/DLA scoring;
only low-bit boundary state and a bounded step buffer are retained. SVD is the
explicit rank-r compensator variant. This is a correctness-oriented backend,
not a claim of the paper's fused-kernel speed.
"""
import jax
import jax.numpy as jnp
from ..cells.base import Step,apply_steps,apply_step

def encode(state,bits=8,rank=1):
    shape=state.shape
    if rank:
        u,s,vh=jnp.linalg.svd(state,full_matrices=False)
        left=u[...,:rank]*s[...,None,:rank]
        right=vh[...,:rank,:]
        residual=state-jnp.matmul(left,right)
    else:
        left=jnp.zeros(shape[:-1]+(0,),state.dtype)
        right=jnp.zeros(shape[:-2]+(0,shape[-1]),state.dtype)
        residual=state
    smooth=jnp.sqrt(jnp.maximum(jnp.mean(jnp.abs(residual),axis=-1,keepdims=True),1e-12))
    whitened=residual/smooth
    limit=127 if bits==8 else 7
    global_scale=jnp.maximum(jnp.max(jnp.abs(whitened),axis=(-2,-1),keepdims=True)/limit,1e-12)
    scale=smooth*global_scale
    q=jnp.clip(jnp.rint(residual/scale),-limit,limit).astype(jnp.int8)
    if bits==4:
        if q.shape[-1]%2:q=jnp.pad(q,[(0,0)]*(q.ndim-1)+[(0,1)])
        codes=(q.astype(jnp.int16)+8).astype(jnp.uint8)
        q=codes[...,::2]|(codes[...,1::2]<<4)
    return dict(q=q,scale=scale,left=left,right=right)

def decode(p):
    q=p["q"]
    dv=p["right"].shape[-1]
    if q.dtype==jnp.uint8:
        low=(q&15).astype(jnp.int16)-8
        high=(q>>4).astype(jnp.int16)-8
        q=jnp.stack((low,high),-1).reshape(q.shape[:-1]+(2*q.shape[-1],))[...,:dv]
    return q.astype(jnp.float32)*p["scale"]+jnp.matmul(p["left"],p["right"])

def init(state,cfg):
    m=cfg.substeps; count=cfg.quant_window*m
    lead=state.shape[:-2];dk,dv=state.shape[-2:]
    return dict(boundary=encode(state,cfg.quant_bits,cfg.quant_rank),u=jnp.zeros((count,)+lead+(dk,)),r=jnp.zeros((count,)+lead+(dk,)),z=jnp.zeros((count,)+lead+(dv,)),g=jnp.zeros((count,)+lead+(dk,)),used=jnp.int32(0))

def materialize(cache,cfg):
    state=decode(cache["boundary"])
    for i in range(cache["u"].shape[0]):
        step=Step(cache["u"][i],cache["r"][i],cache["z"][i],cache["g"][i])
        state=jax.lax.cond(i<cache["used"]*cfg.substeps,lambda s:apply_step(s,step,cfg.precision.dot)[0],lambda s:s,state)
    return state

def append(cache,steps,cfg):
    state,score=apply_steps(materialize(cache,cfg),steps,cfg.precision.dot)
    out=dict(cache)
    for i,step in enumerate(steps):
        offset=cache["used"]*cfg.substeps+i
        for key,value in zip(("u","r","z","g"),step):out[key]=out[key].at[offset].set(value)
    out["used"]=cache["used"]+1
    out=jax.lax.cond(out["used"]>=cfg.quant_window,lambda _:init(state,cfg),lambda _:out,operand=None)
    return out,state,score

def flush(cache,cfg):return init(materialize(cache,cfg),cfg)
