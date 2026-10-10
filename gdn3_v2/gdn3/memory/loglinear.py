"""Exact affine-contribution lift to the paper sentinel + Fenwick partition.

Each old bucket is transported by every current P_t. Bucket addition is exact
because states contain value-origin contributions in the SAME current frame.
It is not addition of independent banks initialized with learned h_init.
"""
import jax
import jax.numpy as jnp
from . import bank
from .. import counters as ct
from ..cells.base import Step, apply_steps, transport_steps

def update(bins,position,steps,projector,precision="highest"):
    # position is zero-based current token. Old sentinel participates in prefix promotion.
    tz=ct.trailing_zeros(position,bins["valid"].shape[0]-1)
    active=~ct.equal(position,ct.zero((),position.shape[-1]))
    target=jnp.minimum(tz+1,bins["valid"].shape[0]-1)
    ids=jnp.arange(bins["valid"].shape[0])
    take=(ids<target)&bins["valid"]
    summed=jnp.sum(jnp.where(take.reshape((-1,)+(1,)*(bins["state"].ndim-1)),bins["state"],0.0),axis=0)
    # Lowest position among selected spans, using word-wise lexical comparison.
    start=position
    for i in range(bins["valid"].shape[0]):
        start=jnp.where(take[i]&ct.less(bins["start"][i],start),bins["start"][i],start)
    promoted=bank.item(summed,start,ct.sub(position,ct.from_int(1,position.shape[-1])),jnp.array(0.0),projector)
    def do_promote(b):
        for i in range(b["valid"].shape[0]):
            b=jax.lax.cond(ids[i]<target,lambda z:bank.clear(z,i),lambda z:z,b)
        return bank.put(b,target,promoted)
    bins=jax.lax.cond(active,do_promote,lambda b:b,bins)
    def moved(s):
        expanded=tuple(Step(x.u[None],x.r[None],x.z[None],x.g[None]) for x in steps)
        return transport_steps(s,expanded,precision)
    bins={**bins,"state":moved(bins["state"])}
    sentinel,_=apply_steps(jnp.zeros_like(bins["state"][0]),steps,precision)
    bins=bank.put(bins,0,bank.item(sentinel,position,position,jnp.array(0.0)))
    return bins
