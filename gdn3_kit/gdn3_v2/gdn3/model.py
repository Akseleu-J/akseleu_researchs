"""Pure-JAX language model, differentiable finite scans, host-resizable stream carry."""
from functools import lru_cache
import json
import math
import jax
import jax.numpy as jnp
import numpy as np
from .config import GDN3Config
from .parameters import init_params
from .cache import init_cache,init_layer_cache,reserve
from .stream import layer_token,empty_stats,STATS
from .precision import rmsnorm,layernorm,linear
from . import counters as ct
from .health import require_healthy

def _step(params,cache,tokens,valid,reset,cfg,training=False,retrieved=None):
    position=jnp.where(reset[:,None],jnp.zeros_like(cache["position"]),cache["position"])
    x=params["embed"][jnp.where(valid,tokens,0)]
    layers=[];stats={k:jnp.zeros(tokens.shape,jnp.float32) for k in STATS}
    for i,(p,c) in enumerate(zip(params["layers"],cache["layers"])):
        lc=cfg.for_layer(i)
        fresh=init_layer_cache(p,lc,c["levels"]["valid"].shape[1],position.shape[-1])
        c=jax.tree_util.tree_map(lambda a,b:jnp.where(reset.reshape(reset.shape+(1,)*(a.ndim-1)),jnp.broadcast_to(b[None],a.shape),a),c,fresh)
        def one(c0,x0,pos,ok,ext=None):
            fn=lambda args:layer_token(p,args[0],args[1],pos,lc,training,ext)
            if cfg.remat and training:fn=jax.checkpoint(fn)
            return jax.lax.cond(ok,fn,lambda args:(args[1],args[0],empty_stats()),(c0,x0))
        if retrieved is None:
            x,nc,st=jax.vmap(one)(c,x,position,valid)
        else:x,nc,st=jax.vmap(one)(c,x,position,valid,retrieved[i])
        layers.append(nc)
        for name in stats:
            stats[name]=jnp.maximum(stats[name],st[name]) if name in {"max_abs","overflow"} else stats[name]+st[name]
    if cfg.backbone=="atomic":logits=linear(layernorm(x),params["head"]["w"],cfg.precision.dot)+params["head"]["b"]
    else:logits=linear(rmsnorm(x,params["final_norm"]),params["embed"].T,cfg.precision.dot)
    logits=jnp.where(valid[:,None],logits,0.0)
    pos=jnp.where(valid[:,None],ct.add(position),position)
    return logits,dict(position=pos,layers=layers),stats

def _scan(params,cache,ids,valid,resets,cfg,training=False,retrieved=None):
    def body(c,inp):
        tok,ok,reset=inp[:3]
        ext=None if retrieved is None else list(inp[3:])
        logits,nc,stats=_step(params,c,tok,ok,reset,cfg,training,ext)
        return nc,(logits,stats)
    if retrieved is None: inputs=(ids.T,valid.T,resets.T)
    else:
        if len(retrieved)!=cfg.n_layers:raise ValueError("retrieved must contain one record tree per layer")
        ext_time=tuple(jax.tree_util.tree_map(lambda a:jnp.swapaxes(a,0,1),item) for item in retrieved)
        inputs=(ids.T,valid.T,resets.T)+ext_time
    cache,(logits,stats)=jax.lax.scan(body,cache,inputs)
    return logits.swapaxes(0,1),cache,stats

def _json(cfg):return json.dumps(cfg.to_dict(),sort_keys=True)
@lru_cache(maxsize=32)
def _step_fn(config_json):
    cfg=GDN3Config.from_dict(json.loads(config_json))
    return jax.jit(lambda p,c,t,v,r,ext:_step(p,c,t,v,r,cfg,False,ext))
@lru_cache(maxsize=32)
def _scan_fn(config_json,training=False):
    cfg=GDN3Config.from_dict(json.loads(config_json))
    return jax.jit(lambda p,c,t,v,r,e:_scan(p,c,t,v,r,cfg,training,e))

def _inputs(ids,cfg,valid):
    if not jnp.issubdtype(ids.dtype,jnp.integer):raise ValueError("token IDs must be integer")
    values=np.asarray(jax.device_get(ids));mask=np.asarray(jax.device_get(valid),bool)
    if np.any((values[mask]<0)|(values[mask]>=cfg.vocab)):raise ValueError("valid token ID outside vocabulary")

def decode_step(params,cache,tokens,cfg,valid=None,reset=None,retrieved=None,return_stats=False):
    tokens=jnp.asarray(tokens);valid=jnp.ones(tokens.shape,bool) if valid is None else jnp.asarray(valid,bool)
    reset=jnp.zeros(tokens.shape,bool) if reset is None else jnp.asarray(reset,bool)
    if tokens.ndim!=1 or valid.shape!=tokens.shape or reset.shape!=tokens.shape:raise ValueError("decode expects matching (B,) token/mask arrays")
    _inputs(tokens,cfg,valid)
    prepared,growth=reserve(cache,cfg,1)
    logits,new,stats=_step_fn(_json(cfg))(params,prepared,tokens,valid,reset,retrieved)
    if cfg.precision.strict_health:require_healthy(stats)
    return (logits,new,{**stats,"allocation":growth}) if return_stats else (logits,new)

def prefill(params,ids,cfg,cache=None,valid=None,resets=None,return_stats=False,retrieved=None):
    ids=jnp.asarray(ids)
    if ids.ndim!=2:raise ValueError("prefill IDs must be (B,T)")
    valid=jnp.ones(ids.shape,bool) if valid is None else jnp.asarray(valid,bool)
    resets=jnp.zeros(ids.shape,bool) if resets is None else jnp.asarray(resets,bool)
    if valid.shape!=ids.shape or resets.shape!=ids.shape:raise ValueError("mask shape mismatch")
    _inputs(ids,cfg,valid)
    cache=init_cache(params,cfg,ids.shape[0]) if cache is None else cache
    cache,growth=reserve(cache,cfg,ids.shape[1])
    logits,new,stats=_scan_fn(_json(cfg))(params,cache,ids,valid,resets,retrieved)
    if cfg.precision.strict_health:require_healthy(stats)
    return (logits,new,{**stats,"allocation":growth}) if return_stats else (logits,new)

def forward(params,ids,cfg,cache=None,valid=None,resets=None,training=True,retrieved=None):
    """Differentiable path. TBPTT callers reserve a supplied cache before tracing."""
    if training and cfg.quant!="none":raise ValueError("LeapQuant/archive_int8 are inference-only; train with quant=none")
    ids=jnp.asarray(ids);B,T=ids.shape
    valid=jnp.ones(ids.shape,bool) if valid is None else valid
    resets=jnp.zeros(ids.shape,bool) if resets is None else resets
    if cache is None:
        levels=cfg.initial_levels
        while levels<max(1,(T+1).bit_length()+1):levels*=2
        if cfg.max_levels:levels=min(levels,cfg.max_levels)
        cache=init_cache(params,cfg,B,levels=levels)
    return _scan(params,cache,ids,valid,resets,cfg,training,retrieved)

def masked_ce(logits,ids,mask=None):
    if ids.shape[1]<2:return jnp.float32(0),jnp.float32(0)
    pred=logits[:,:-1].astype(jnp.float32);targets=ids[:,1:]
    weights=jnp.ones(targets.shape) if mask is None else mask[:,1:].astype(jnp.float32)
    loss=jax.nn.logsumexp(pred,axis=-1)-jnp.take_along_axis(pred,targets[...,None],axis=-1)[...,0]
    den=jnp.maximum(weights.sum(),1)
    return (loss*weights).sum()/den,((pred.argmax(-1)==targets)*weights).sum()/den

def count_params(params):return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
