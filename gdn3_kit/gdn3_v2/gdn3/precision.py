"""One precision policy for projections, recurrence, descriptors and reads."""
import jax
import jax.numpy as jnp

def einsum(spec, a, b, mode="highest"):
    if mode == "bf16x3":
        ah, bh = a.astype(jnp.bfloat16).astype(jnp.float32), b.astype(jnp.bfloat16).astype(jnp.float32)
        al = (a - ah).astype(jnp.bfloat16).astype(jnp.float32)
        bl = (b - bh).astype(jnp.bfloat16).astype(jnp.float32)
        dot = lambda x, y: jnp.einsum(spec, x, y, precision=jax.lax.Precision.DEFAULT)
        return dot(ah, bh) + dot(ah, bl) + dot(al, bh)
    return jnp.einsum(spec, a, b, precision=jax.lax.Precision.HIGHEST if mode == "highest" else jax.lax.Precision.DEFAULT)

def linear(x, w, mode="highest"):
    return einsum("...d,de->...e", x, w, mode)

def rmsnorm(x, weight=1.0, eps=1e-6):
    f = x.astype(jnp.float32)
    return f * jax.lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + eps) * weight

def l2norm(x, group=0, eps=1e-6):
    if group:
        sh = x.shape
        y = x.reshape(sh[:-1] + (sh[-1] // group, group))
        return (y * jax.lax.rsqrt(jnp.sum(y*y, -1, keepdims=True) + eps)).reshape(sh)
    return x * jax.lax.rsqrt(jnp.sum(x*x, -1, keepdims=True) + eps)

def layernorm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) * jax.lax.rsqrt(x.var(-1, keepdims=True) + eps)
