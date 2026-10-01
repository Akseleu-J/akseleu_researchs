"""Archive read: content-based routing over banks by read-norm confidence + null slot (online softmax).

For token t (segment i) and bank j < i:   r_j = scale * (q_t * exp(G_t - Gend_j)) @ S_j
   logit_j = tau * rms(r_j)                       (a bank that really holds the key returns a coherent, large vector)
   a = softmax([theta, logit_1..logit_{i}])       (theta = null slot, value 0: "nothing is confident")
   o_arch = sum_j a_j r_j
Memory O(L*D) (scan over banks, checkpointed). Compute 2*L*D^2*n per head. Causality: only j < seg(t).
"""
import jax, jax.numpy as jnp

NEG = -1e30
_PREC = {"default": jax.lax.Precision.DEFAULT, "high": jax.lax.Precision.HIGH, "highest": jax.lax.Precision.HIGHEST}


def read_banks(q, S, Gtok, Gend, seg_idx, tau, theta, scale, use_decay=True, precision="high"):
    """q (B,H,L,D); S (B,H,n,D,D) [key,value]; Gtok (B,H,L,D) global cumsum of g; Gend (B,H,n,D) cumsum at bank end;
    seg_idx (L,) int: token t may read banks j < seg_idx[t]; tau, theta (H,). Returns (B,H,L,D)."""
    prec = _PREC[precision]
    B, H, L, D = q.shape; n = S.shape[2]
    tau_ = tau[None, :, None]
    m0 = jnp.broadcast_to(theta[None, :, None], (B, H, L)).astype(jnp.float32)
    z0 = jnp.ones((B, H, L), jnp.float32); a0 = jnp.zeros((B, H, L, D), jnp.float32)

    def body(carry, xs):
        m, z, acc = carry
        S_j, Ge_j, j = xs
        qe = q * jnp.exp(jnp.minimum(Gtok - Ge_j[:, :, None, :], 0.0)) if use_decay else q
        r = jnp.einsum("bhld,bhde->bhle", qe, S_j, precision=prec) * scale
        s = tau_ * jnp.sqrt(jnp.mean(r * r, -1) + 1e-6)
        ok = (j < seg_idx)[None, None, :]
        s = jnp.where(ok, s, NEG)
        mn = jnp.maximum(m, s)
        a = jnp.exp(m - mn); e = jnp.exp(s - mn) * ok
        return (mn, z * a + e, acc * a[..., None] + e[..., None] * r), None

    xs = (jnp.moveaxis(S, 2, 0), jnp.moveaxis(Gend, 2, 0), jnp.arange(n))
    (_, z, acc), _ = jax.lax.scan(jax.checkpoint(body), (m0, z0, a0), xs)
    return acc / z[..., None]
