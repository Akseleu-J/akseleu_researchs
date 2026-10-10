"""Recent information summaries plus one live binary bin per promotion level."""
import jax
import jax.numpy as jnp
from . import bank
from .. import counters as ct

def promote(levels,value,projector,aligned):
    pending=jnp.array(True)
    for i in range(levels["valid"].shape[0]):
        old=bank.get(levels,i)
        joined=bank.merge(old,value,projector,aligned)
        chosen=jax.tree_util.tree_map(lambda a,b:jnp.where(old["valid"],a,b),joined,value)
        next_levels=jax.lax.cond(old["valid"],lambda b:bank.clear(b,i),lambda b:bank.put(b,i,value),levels)
        levels=jax.tree_util.tree_map(lambda a,b:jnp.where(pending,a,b),next_levels,levels)
        value=jax.tree_util.tree_map(lambda a,b:jnp.where(pending,a,b),chosen,value)
        pending=pending&old["valid"]
    return levels,pending

def append(recent,levels,value,position,cfg,projector):
    """All control flow is JAX-compatible. Host reserves levels before entering scan."""
    overflow=jnp.array(False);merges=jnp.int32(0);promotions=jnp.int32(0)
    if cfg.memory=="fifo":
        recent=jax.lax.cond(jnp.all(recent["valid"]),bank.shift,lambda b:b,recent)
    else:
        # Independent age eviction ensures count-based merges do not prevent promotions forever.
        for _ in range(cfg.recent_slots):
            oldest=bank.get(recent,0)
            aged=oldest["valid"]&ct.small_at_least(ct.sub(position,oldest["end"]),cfg.recent_horizon)
            def move(args):
                r,h,o,p=args
                h,spilled=promote(h,bank.get(r,0),projector,cfg.bank_decay=="aligned")
                return bank.shift(r),h,o|spilled,p+1
            recent,levels,overflow,promotions=jax.lax.cond(aged,move,lambda a:a,(recent,levels,overflow,promotions))
        full=jnp.all(recent["valid"])
        recent=jax.lax.cond(full,lambda b:bank.merge_low_information(b,projector,cfg.bank_decay=="aligned"),lambda b:b,recent)
        merges=merges+full.astype(jnp.int32)
    slot=recent["valid"].sum().astype(jnp.int32)
    recent=bank.put(recent,jnp.minimum(slot,cfg.recent_slots-1),value)
    return recent,levels,dict(overflow=overflow,merges=merges,promotions=promotions)

def candidates(recent,levels):
    return jax.tree_util.tree_map(lambda a,b:jnp.concatenate((a,b),0),recent,levels)
