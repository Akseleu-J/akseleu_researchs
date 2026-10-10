"""QED Eq.6, norm-aware projector variant for near-zero keys."""
import jax.numpy as jnp

def address(k, q, b, strength, eps=1e-12):
    x = b*q
    norm2 = jnp.sum(k*k, -1, keepdims=True)
    orth = x - k * (jnp.sum(k*x, -1, keepdims=True) / jnp.maximum(norm2, eps))
    return b*k + strength[...,None] * orth
