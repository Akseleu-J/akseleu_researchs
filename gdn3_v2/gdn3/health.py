"""Health is observed before any optional numerical rescue."""
import jax
import jax.numpy as jnp

def observe(*arrays):
    nonfinite=jnp.int32(0);maximum=jnp.float32(0)
    for a in arrays:
        nonfinite+=jnp.sum(~jnp.isfinite(a),dtype=jnp.int32)
        maximum=jnp.maximum(maximum,jnp.max(jnp.where(jnp.isfinite(a),jnp.abs(a),jnp.inf),initial=0))
    return dict(nonfinite=nonfinite,max_abs=maximum)

def require_healthy(stats):
    nonfinite=int(jax.device_get(stats["nonfinite"]).sum())
    if nonfinite:raise FloatingPointError(f"{nonfinite} non-finite values before sanitization")
    if "overflow" in stats and bool(jax.device_get(jnp.any(stats["overflow"]))):
        raise OverflowError("memory/counter capacity exhausted; reserve more levels before continuing")

def live_bytes(tree):
    return sum(int(x.size)*int(x.dtype.itemsize) for x in jax.tree_util.tree_leaves(tree) if hasattr(x,"dtype"))
