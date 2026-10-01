"""EXPERIMENTAL exact landmark cache (HOLA-style). Per segment keep the P most 'surprising' tokens
(surprise = ||v_new|| * mean(beta), v_new = delta-rule residual) as exact (k, w*v) pairs; read them with a sharp softmax
over all PAST landmarks (+ null slot). Needs backend(..., return_vnew=True). Causal: segment j feeds only segments > j."""
import jax, jax.numpy as jnp

NEG = -1e9


def pick_landmarks(k, wv, vnew, beta, n, C, P):
    """k, wv, vnew, beta: (B,H,n*C,D) -> lk, lv (B,H,n,P,D)."""
    B, H, L, D = k.shape
    surp = jnp.linalg.norm(jax.lax.stop_gradient(vnew), axis=-1) * jax.lax.stop_gradient(beta).mean(-1)   # (B,H,L)
    _, idx = jax.lax.top_k(surp.reshape(B, H, n, C), P)                                                    # (B,H,n,P)
    take = lambda t: jnp.take_along_axis(t.reshape(B, H, n, C, D), idx[..., None], axis=3)
    return take(k), take(wv)


def landmark_read(q, lk, lv, sharp, lnull, C, qblk=512):
    """q (B,H,L,D) -> (B,H,L,D). sharp, lnull: (H,)."""
    B, H, L, D = q.shape; n, P = lk.shape[2], lk.shape[3]
    K = lk.reshape(B, H, n * P, D); V = lv.reshape(B, H, n * P, D)
    segm = jnp.arange(n * P) // P
    qblk = min(qblk, L); pad = (-L) % qblk
    qp = jnp.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0))) if pad else q
    nb = qp.shape[2] // qblk

    def one(args):
        qb, pos0 = args
        seg_q = (pos0 + jnp.arange(qblk)) // C
        valid = segm[None, :] < seg_q[:, None]
        s = jnp.einsum("bhqd,bhmd->bhqm", qb, K) * sharp[None, :, None, None]
        s = jnp.where(valid[None, None], s, NEG)
        sn = jnp.broadcast_to(lnull[None, :, None, None], s.shape[:3] + (1,))
        a = jax.nn.softmax(jnp.concatenate([s, sn], -1), -1)[..., :-1]
        return jnp.einsum("bhqm,bhmd->bhqd", a, V)

    qb = qp.reshape(B, H, nb, qblk, D).transpose(2, 0, 1, 3, 4)
    out = jax.lax.map(jax.checkpoint(one), (qb, jnp.arange(nb) * qblk))
    return out.transpose(1, 2, 0, 3, 4).reshape(B, H, nb * qblk, D)[:, :, :L]
