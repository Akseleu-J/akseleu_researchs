"""Content-addressed bank router.

Score of bank j for query q:   tau * log( ||q S_j||^2 / E_q||q S_j||^2 )  -  lam * log(1 + age)
    E_q ||q S_j||^2 = ||S_j||_F^2 * ||q||^2 / D   (expected energy for a random query)
The ratio is ~1 for banks that do NOT store q's key and >> 1 for the bank that does, independent of how full
the bank is (see docs/DESIGN.md and sim/router_sim.py).  ||q S_j||^2 is estimated with a Johnson-Lindenstrauss
sketch  ||q (S_j R)||^2  (R: D x m, fixed Gaussian) so the index costs D*m instead of D*D per bank.
A learned "null" logit lets the model read nothing from the archive when no bank matches.
"""
from __future__ import annotations
import math

import jax
import jax.numpy as jnp
import numpy as np

NEG = -1e9
_PREC = {"default": jax.lax.Precision.DEFAULT, "high": jax.lax.Precision.HIGH,
         "highest": jax.lax.Precision.HIGHEST}


def prec(name: str):
    return _PREC[name]


def make_sketch(D: int, m: int, seed: int = 1234):
    """Fixed Gaussian JL matrix (D, m); None means 'exact' (no sketch)."""
    if m <= 0 or m >= D:
        return None
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(D, m)).astype(np.float32) / math.sqrt(m))


def sketch_bank(S, R):
    """S (...,D,D) -> T (...,D,m).  q @ T approximates q @ S in norm."""
    if R is None:
        return S
    return jnp.einsum("...kv,vm->...km", S, R, precision=jax.lax.Precision.HIGHEST)


def frob2(S):
    return jnp.sum(S * S, axis=(-2, -1))


def log_ratio(e, sf, qn2, D: int, eps: float = 1e-12):
    """e: ||q T||^2 ; sf: ||S||_F^2 ; qn2: ||q||^2 (all broadcastable)."""
    e0 = sf * qn2 / D
    return jnp.log((e + eps) / (e0 + eps))


def bank_gate(scores, valid, null_logit, top_k: int):
    """Sparse softmax over the top-k valid banks plus a 'null' option.

    scores (...,n); valid broadcastable to scores (bool); null_logit broadcastable to (...,1).
    Returns a (...,n) with sum_j a_j = 1 - p_null <= 1; zero for non-selected / invalid banks."""
    masked = jnp.where(valid, scores, NEG)
    kk = min(top_k, scores.shape[-1])
    kth = jax.lax.stop_gradient(jax.lax.top_k(masked, kk)[0][..., -1:])
    sel = (masked >= kth) & valid
    sc = jnp.where(sel, scores, NEG)
    nl = jnp.broadcast_to(null_logit, sc.shape[:-1] + (1,))
    allc = jnp.concatenate([sc, nl], axis=-1)
    return jax.nn.softmax(allc, axis=-1)[..., :-1] * sel
