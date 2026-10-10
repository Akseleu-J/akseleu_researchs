"""Paper hypergradient and separately named asymmetric GDN-2 adaptation."""
import jax.numpy as jnp

def paper_gradient(k, beta, d, eps=1e-12):
    kk = k*k
    return -beta[...,None]*(1-beta*jnp.sum(d*kk,-1))[...,None]*kk/jnp.maximum(jnp.sum(kk,-1,keepdims=True),eps)

def generalized_gradient(k, r, d, eps=1e-12):
    return -(1-jnp.sum(r*d*k,-1))[...,None]*(r*k)/jnp.maximum(jnp.sum(r*r,-1,keepdims=True),eps)

def feasible(d, k, r, lo, hi, margin):
    """Box plus segment-to-d=1 repair. Feasible base is required by normalized address construction."""
    d = jnp.clip(d,lo,hi)
    a = r*k
    base, val = jnp.sum(a,-1), jnp.sum(a*d,-1)
    # Neutral updates are legal; enforce a closed [0,2-margin] rather than inventing contraction at zero.
    target = jnp.clip(val,0.0,2.0-margin)
    denom = val-base
    t = jnp.where(jnp.abs(denom)>1e-12,(target-base)/jnp.where(jnp.abs(denom)>1e-12,denom,1.0),1.0)
    repaired = 1.0 + jnp.clip(t,0.0,1.0)[...,None]*(d-1.0)
    return jnp.where(((val<0)|(val>2-margin))[...,None], repaired, d)
