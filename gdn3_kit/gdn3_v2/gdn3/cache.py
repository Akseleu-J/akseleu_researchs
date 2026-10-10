"""Allocation and exact multiword clock growth happen on the host, outside JIT."""
import math
import jax
import jax.numpy as jnp
import numpy as np
from . import counters as ct
from .memory import bank,ccq,hola
from .compression import leapquant
from .health import live_bytes

def init_layer_cache(p,cfg,levels,words):
    H,A,N,K,V,M=cfg.n_heads,cfg.ha,cfg.sse_partitions,cfg.d_key,cfg.d_value,cfg.delta_steps
    hc=jnp.zeros((H,K,V),jnp.float32)
    active=p["h_init"].astype(jnp.float32) if A and cfg.learned_init else jnp.zeros((A,N,K,V),jnp.float32)
    if cfg.quant=="leap" and cfg.quant_scope=="all":hc=leapquant.init(hc,cfg)
    if cfg.quant=="leap" and A:active=leapquant.init(active,cfg)
    bits=cfg.quant_bits if cfg.quant!="none" else 0
    rank=cfg.quant_rank if bits else 0
    b=lambda slots:bank.empty(slots,A,N,K,V,words,cfg.router_dim,bits,rank)
    c=dict(hc=hc,active=active,dc=jnp.ones((H,M,K)),da=jnp.ones((A,N,M,K)),pdn_A=jnp.ones((H,K)),conv_q=jnp.zeros((cfg.d_conv-1,H*K)),conv_k=jnp.zeros((cfg.d_conv-1,H*K)),conv_v=jnp.zeros((cfg.d_conv-1,H*V)),recent=b(cfg.recent_slots),levels=b(levels),count=jnp.int32(0),start=ct.zero((),words),density=jnp.float32(0),ccq=ccq.init(H if cfg.ccq!="off" else 0,K,words),hola=hola.init(A,K,V,cfg.hola_size,cfg.hola_block if cfg.hola_size else 0,words),warm_cursor=jnp.int32(0),warm_count=jnp.int32(0))
    for name,dim in [("u",K),("r",K),("z",V),("g",K)]:c["warm_"+name]=jnp.zeros((cfg.warm,cfg.substeps,A,N,dim))
    return c

def init_cache(params,cfg,batch=1,levels=None,words=None):
    levels=levels or cfg.initial_levels;words=words or cfg.initial_counter_words
    if cfg.max_levels and levels>cfg.max_levels:raise ValueError("requested levels exceed bounded configuration")
    single=[init_layer_cache(p,cfg.for_layer(i),levels,words) for i,p in enumerate(params["layers"])]
    layers=[jax.tree_util.tree_map(lambda a:jnp.broadcast_to(a[None],(batch,)+a.shape),c) for c in single]
    return dict(position=ct.zero((batch,),words),layers=layers)

def _widen_counters(c,words,path=()):
    if isinstance(c,dict):return {k:_widen_counters(v,words,path+(k,)) for k,v in c.items()}
    if isinstance(c,list):return [_widen_counters(v,words,path+(str(i),)) for i,v in enumerate(c)]
    if hasattr(c,"dtype") and c.dtype==jnp.uint32 and path[-1] in {"position","start","end","ids","count"}:
        return jnp.pad(c,[(0,0)]*(c.ndim-1)+[(0,words-c.shape[-1])])
    return c

def reserve(cache,cfg,additional_tokens):
    """Geometric capacity makes final shape independent of prefill chunking."""
    positions=np.asarray(jax.device_get(cache["position"]))
    maximum=max((sum(int(w)<<(32*i) for i,w in enumerate(row)) for row in positions),default=0)+int(additional_tokens)
    needed=max(1,maximum.bit_length()+1) # sentinel plus prefix bins
    old=cache["layers"][0]["levels"]["valid"].shape[1]
    levels=old
    while levels<needed:levels*=2
    if cfg.max_levels:
        # Bounded policy may cover fewer promotions than token upper bound. Reserve up to cap;
        # actual binary overflow is reported transactionally by the device.
        levels=min(levels,cfg.max_levels)
    words=cache["position"].shape[-1]
    while 32*words<maximum.bit_length():words*=2
    out=_widen_counters(cache,words) if words!=cache["position"].shape[-1] else cache
    if levels>old:
        out={**out,"layers":[{**c,"levels":jax.tree_util.tree_map(lambda a:jnp.pad(a,[(0,0),(0,levels-old)]+[(0,0)]*(a.ndim-2)),c["levels"])} for c in out["layers"]]}
    return out,dict(levels_before=old,levels_after=levels,counter_words=words,allocated_bytes=live_bytes(out),grew=levels>old or words!=cache["position"].shape[-1])
