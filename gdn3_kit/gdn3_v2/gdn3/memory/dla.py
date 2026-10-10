import jax
import jax.numpy as jnp

def information(before,candidate,eps=1e-6):
    axes=(-2,-1)
    a=before*jax.lax.rsqrt(jnp.mean(before*before,axis=axes,keepdims=True)+eps)
    b=candidate*jax.lax.rsqrt(jnp.mean(candidate*candidate,axis=axes,keepdims=True)+eps)
    score=jnp.linalg.norm(b-a,axis=axes)/(jnp.linalg.norm(a,axis=axes)+eps)
    return jnp.sqrt(jnp.mean(score*score))

def boundary(score,count,threshold,cfg,training=False):
    allowed=count>=cfg.segment_min
    forced=count>=cfg.segment_max
    hard=(allowed&(score>=threshold)) if cfg.dla else jnp.array(False)
    hard=hard|forced
    soft=jax.nn.sigmoid((score-threshold)/cfg.boundary_temperature)
    soft=jnp.where(allowed,soft,0.0)
    soft=jnp.where(forced,1.0,soft)
    if not training or cfg.boundary_grad=="fixed": weight=hard.astype(jnp.float32)
    elif cfg.boundary_grad=="soft": weight=soft
    else: weight=soft+jax.lax.stop_gradient(hard.astype(jnp.float32)-soft)
    # Topology uses a hard causal boundary; soft path interpolates only active state.
    return hard,weight,soft
