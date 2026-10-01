"""Multi-Query Associative Recall (MQAR) with a capacity knob P and a distance knob (filler length).
Layout of one example (length exactly L):
   [k1 v1 k2 v2 ... kP vP] [filler x F] [SEP k_i v_i] x n_q          (F = L - 2P - 3 n_q)
Loss/accuracy only on the answer tokens v_i. Keys are single tokens (vocab nk), values nv classes, chance = 1/nv."""
import numpy as np
import jax, jax.numpy as jnp


def vocab_of(nk=2048, nv=64, nf=32):
    return nk + nv + nf + 1


def mqar_example(rng, L, P, n_q, nk=2048, nv=64, nf=32):
    assert P <= nk, "need nk >= P (unique keys)"
    F = L - 2 * P - 3 * n_q
    assert F >= 0, f"L={L} too short for P={P}, n_q={n_q}"
    VAL0, FIL0, SEP = nk, nk + nv, nk + nv + nf
    keys = rng.choice(nk, size=P, replace=False); vals = VAL0 + rng.integers(0, nv, P)
    body = np.stack([keys, vals], 1).reshape(-1)
    fill = FIL0 + rng.integers(0, nf, F)
    qi = rng.integers(0, P, n_q)
    qs = np.stack([np.full(n_q, SEP), keys[qi], vals[qi]], 1).reshape(-1)
    seq = np.concatenate([body, fill, qs]); m = np.zeros(L, bool); m[2 * P + F + 2::3] = True
    return seq.astype(np.int32), m


def make_batch(rng, bsz, L, P, n_q=32, nk=2048, nv=64, nf=32):
    """P may be an int or a sequence of ints (one chosen per example)."""
    Ps = [P] * bsz if isinstance(P, int) else [int(rng.choice(P)) for _ in range(bsz)]
    ex = [mqar_example(rng, L, p, n_q, nk, nv, nf) for p in Ps]
    return np.stack([e[0] for e in ex]), np.stack([e[1] for e in ex])


def mqar_loss_acc(logits, ids, mask):
    tgt = ids[:, 1:]; m = mask[:, 1:].astype(jnp.float32)
    lp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), -1)
    nll = -jnp.take_along_axis(lp, tgt[..., None], -1)[..., 0]
    acc = ((jnp.argmax(logits[:, :-1], -1) == tgt) * m).sum() / m.sum()
    return (nll * m).sum() / m.sum(), acc
