"""S[Dk,Dv] -> D*S + u*(z-r^T*D*S)^T. All leading axes broadcast."""
from typing import NamedTuple
import jax.numpy as jnp
from ..precision import einsum

class Step(NamedTuple):
    u: object
    r: object
    z: object
    g: object
    write: bool = True

def apply_step(state, step, precision="highest"):
    base = state * jnp.exp(step.g)[..., :, None]
    residual = step.z - einsum("...d,...de->...e", step.r, base, precision)
    delta = step.u[..., :, None] * residual[..., None, :]
    return base + delta, residual

def apply_steps(state, steps, precision="highest", primitive=None):
    score = jnp.zeros(state.shape[:-2], jnp.float32)
    for step in steps:
        if primitive is None:
            state, residual = apply_step(state, step, precision)
        else:
            state, residual = primitive(state, step, precision)
        score += jnp.linalg.norm(step.u, axis=-1) * jnp.linalg.norm(residual, axis=-1) * jnp.asarray(step.write,jnp.float32)
    return state, score

def transport_steps(state, steps, precision="highest"):
    for s in steps:
        state, _ = apply_step(state, Step(s.u, s.r, jnp.zeros(s.u.shape[:-1]+(state.shape[-1],),state.dtype), s.g, False), precision)
    return state

def affine(steps, d_key, d_value, precision="highest"):
    shape = steps[0].u.shape[:-1]
    P = jnp.broadcast_to(jnp.eye(d_key), shape + (d_key, d_key))
    U = jnp.zeros(shape + (d_key, d_value), jnp.float32)
    P = transport_steps(P, steps, precision)
    U, _ = apply_steps(U, steps, precision)
    return P, U
