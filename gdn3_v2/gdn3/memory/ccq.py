import jax
import jax.numpy as jnp
from .. import counters as ct
from ..precision import einsum

def init(heads,dk,words):
    return dict(count=ct.zero((),words),mean=jnp.zeros((heads,dk)),cov=jnp.zeros((heads,dk,dk)))

def update(c,k):
    count=ct.add(c["count"])
    a=jnp.exp(-ct.log_value(count))
    delta=k-c["mean"]
    mean=c["mean"]+a*delta
    # Normalized Welford covariance avoids an unbounded M2 accumulator.
    cov=(1-a)*c["cov"]+a*delta[..., :,None]*(k-mean)[...,None,:]
    return dict(count=count,mean=mean,cov=(cov+jnp.swapaxes(cov,-1,-2))*0.5)

def clean(q,c,strength,precision="highest"):
    return q-strength[...,None]*einsum("hde,he->hd",c["cov"],q,precision)
