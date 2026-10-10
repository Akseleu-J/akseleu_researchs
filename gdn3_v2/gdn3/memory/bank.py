import jax
import jax.numpy as jnp
from .. import counters as ct
from ..compression.leapquant import encode,decode
from ..precision import einsum

def empty(slots,heads,partitions,dk,dv,words,sketch_dim=16,bits=0,rank=0):
    shape=(slots,heads,partitions,dk,dv)
    state=jnp.zeros(shape,jnp.float32)
    encoded=encode(state,bits,rank) if bits else dict(q=state,scale=jnp.ones(shape[:-1]+(1,)),left=jnp.zeros(shape[:-1]+(0,)),right=jnp.zeros(shape[:-2]+(0,dv)))
    return dict(state=encoded["q"],scale=encoded["scale"],left=encoded["left"],right=encoded["right"],sketch=jnp.zeros(shape[:-1]+(sketch_dim,)),norm=jnp.zeros(shape[:-2]),start=ct.zero((slots,),words),end=ct.zero((slots,),words),density=jnp.zeros((slots,),jnp.float32),valid=jnp.zeros((slots,),bool),decay=jnp.zeros((slots,heads,dk),jnp.float32))

def tensor(b):
    if jnp.issubdtype(b["state"].dtype,jnp.floating):return b["state"]
    return decode(dict(q=b["state"],scale=b["scale"],left=b["left"],right=b["right"]))

def item(state,start,end,density,projector,bits=0,rank=0):
    encoded=encode(state,bits,rank) if bits else dict(q=state,scale=jnp.ones(state.shape[:-1]+(1,)),left=jnp.zeros(state.shape[:-1]+(0,)),right=jnp.zeros(state.shape[:-2]+(0,state.shape[-1])))
    stored=decode(encoded) if bits else state
    return dict(state=encoded["q"],scale=encoded["scale"],left=encoded["left"],right=encoded["right"],sketch=einsum("...dv,vm->...dm",stored,projector),norm=jnp.sum(stored*stored,axis=(-2,-1)),start=start,end=end,density=density,valid=jnp.array(True),decay=jnp.zeros(state.shape[:1]+state.shape[-2:-1],jnp.float32))

def get(banks,index):return jax.tree_util.tree_map(lambda x:x[index],banks)
def put(banks,index,value):return jax.tree_util.tree_map(lambda x,y:x.at[index].set(y),banks,value)
def clear(banks,index):return put(banks,index,jax.tree_util.tree_map(jnp.zeros_like,get(banks,index)))
def shift(banks):return jax.tree_util.tree_map(lambda x:jnp.concatenate((x[1:],jnp.zeros_like(x[:1])),axis=0),banks)
def realized(b):return tensor(b)*jnp.exp(b["decay"])[...,None,:,None]

def merge(left,right,projector,aligned=False):
    ln=ct.log_value(ct.span_count(left["start"],left["end"]))
    rn=ct.log_value(ct.span_count(right["start"],right["end"]))
    alpha=jax.nn.sigmoid(ln-rn)
    a,b=(realized(left),realized(right)) if aligned else (tensor(left),tensor(right))
    bits=0 if jnp.issubdtype(left["state"].dtype,jnp.floating) else (4 if left["state"].dtype==jnp.uint8 else 8)
    return item(alpha*a+(1-alpha)*b,left["start"],right["end"],alpha*left["density"]+(1-alpha)*right["density"],projector,bits,left["left"].shape[-1])

def age_decay(banks,g):
    return {**banks,"decay":jnp.where(banks["valid"][...,None,None],jnp.maximum(banks["decay"]+g[None],-10000.0),banks["decay"])}

def merge_low_information(banks,projector,aligned=False):
    n=banks["valid"].sum()
    ln=ct.log_value(ct.span_count(banks["start"][:-1],banks["end"][:-1]))
    rn=ct.log_value(ct.span_count(banks["start"][1:],banks["end"][1:]))
    w=jax.nn.sigmoid(ln-rn)
    density=w*banks["density"][:-1]+(1-w)*banks["density"][1:]
    idx=jnp.argmin(jnp.where(banks["valid"][:-1]&banks["valid"][1:],density,jnp.inf))
    joined=merge(get(banks,idx),get(banks,idx+1),projector,aligned)
    banks=put(banks,idx,joined)
    def compact(x):
        ids=jnp.arange(x.shape[0]);src=jnp.minimum(ids+(ids>idx),x.shape[0]-1)
        y=x[src];mask=(ids<n-1).reshape((x.shape[0],)+(1,)*(x.ndim-1))
        return jnp.where(mask,y,jnp.zeros_like(y))
    return jax.tree_util.tree_map(compact,banks)
