"""Content-addressed router (energy-ratio score on a JL sketch of the bank + null slot) and the shared per-bank read."""
import math
import jax
import jax.numpy as jnp
import numpy as np

NEG = -1e9
_PREC = {"default": jax.lax.Precision.DEFAULT, "high": jax.lax.Precision.HIGH, "highest": jax.lax.Precision.HIGHEST}


def prec(name):
    return _PREC[name]


def make_sketch(D, m, seed=1234):
    """Fixed Gaussian JL matrix (D,m); None = exact (no sketch)."""
    if m <= 0 or m >= D:
        return None
    return jnp.asarray(np.random.default_rng(seed).normal(size=(D, m)).astype(np.float32) / math.sqrt(m))


def sketch_bank(S, R):
    if R is None:
        return S
    return jnp.einsum("...kv,vm->...km", S, R, precision=jax.lax.Precision.HIGHEST)


def frob2(S):
    return jnp.sum(S * S, axis=(-2, -1))


def log_ratio(e, sf, qn2, D, eps=1e-12):
    """log( ||q T||^2 / E_q||q S||^2 ),  E = ||S||_F^2 ||q||^2 / D  (~0 for banks without the key, >>0 for the one with it)."""
    return jnp.log((e + eps) / (sf * qn2 / D + eps))


def bank_gate(scores, valid, null_logit, top_k):
    """Sparse softmax over exactly top_k valid banks + a null option; returns (...,n) with sum = 1 - p_null."""
    n = scores.shape[-1]
    _, idx = jax.lax.top_k(jnp.where(valid, scores, NEG), min(top_k, n))
    sel = jnp.any(idx[..., None] == jnp.arange(n), axis=-2) & valid
    sc = jnp.where(sel, scores, NEG)
    nl = jnp.broadcast_to(null_logit, sc.shape[:-1] + (1,))
    return jax.nn.softmax(jnp.concatenate([sc, nl], -1), axis=-1)[..., :-1] * sel


def read_core(qe, mm, mmT, rown, sf, lam, scale, cfg):
    """Read of one bank S (layout-agnostic).  qe (...,D) decayed query; mm(x)=x@S ; mmT(x)=x@S^T ;
    rown (...,D) row norms of S ; sf (...,1) ||S||_F^2 ; lam (...,1) CCQ strength.
      B4 SSE : keep only the sse_k rows with the largest |q_i|*||S_i||.
      C1 CCQ (STATE-DERIVED VARIANT, not the paper's): q' = q - lam * q Sigma_S with Sigma_S = S S^T/||S||_F^2 (trace 1 => I - lam*Sigma_S PSD),
               computed as the algebraically identical r - lam*(r S^T) S/||S||^2.  The paper (arXiv 2606.01294) uses the running KEY
               covariance C_t = decay*C_{t-1} + k k^T instead; that variant is NOT implemented."""
    if cfg.sse_k:
        c = jnp.abs(qe) * rown
        kth = jax.lax.top_k(c, min(cfg.sse_k, c.shape[-1]))[0][..., -1:]
        qe = qe * (c >= kth)
    r = mm(qe)
    if cfg.ccq:
        # r - lam*(rS^T)S/||S||^2 == (q - lam*q*Sigma_S) S with Sigma_S = S S^T/||S||^2 (post-read and pre-read forms coincide for the SAME Sigma)
        r = r - lam * mm(mmT(r)) / (sf + 1e-9)
    return scale * r
