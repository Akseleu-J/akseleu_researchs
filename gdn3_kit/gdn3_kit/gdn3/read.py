"""Parallel (training / prefill) archive read: router + top-k weighted bank reads.

Token t of segment s reads banks j < s only (closed banks) -> causal at token level.
Python loops over banks use static slices, so each bank only touches tokens of later segments
(triangular cost ~ L*n/2*D*D) and every iteration is rematerialised in the backward pass."""
from __future__ import annotations
import math

import jax
import jax.numpy as jnp

from .routing import prec, make_sketch, sketch_bank, frob2, log_ratio, bank_gate


def bank_energy(q, T, C: int, pr):
    """q (B,Ha,L,D); T (B,Ha,n,D,m) -> energies (B,Ha,L,n); zero where the bank is not yet closed."""
    B, Ha, L, D = q.shape
    n = T.shape[2]

    def f(qs, Tj):
        r = jnp.einsum("bhld,bhdm->bhlm", qs, Tj, precision=pr)
        return jnp.sum(r * r, axis=-1)

    cols = []
    for j in range(n):
        lo = (j + 1) * C
        if lo >= L:
            cols.append(jnp.zeros((B, Ha, L), jnp.float32))
            continue
        e = jax.checkpoint(f)(q[:, :, lo:], T[:, :, j])
        cols.append(jnp.pad(e, ((0, 0), (0, 0), (lo, 0))))
    return jnp.stack(cols, axis=-1)


def routed_read(q, S, a, C: int, scale: float, pr):
    """q (B,Ha,L,D); S (B,Ha,n,D,D); a (B,Ha,L,n) -> scale * sum_j a_j (q S_j)   (B,Ha,L,D)."""
    B, Ha, L, D = q.shape
    n = S.shape[2]

    def f(qs, Sj, aj):
        r = jnp.einsum("bhld,bhde->bhle", qs, Sj, precision=pr)
        return aj[..., None] * r

    out = jnp.zeros_like(q)
    for j in range(n):
        lo = (j + 1) * C
        if lo >= L:
            break
        r = jax.checkpoint(f)(q[:, :, lo:], S[:, :, j], a[:, :, lo:, j])
        out = out + jnp.pad(r, ((0, 0), (0, 0), (lo, 0), (0, 0)))
    return out * scale


def archive_read(q, S, p, cfg, C: int, oracle=None, want_stats: bool = False):
    """q (B,Ha,L,D) (archive heads); S (B,Ha,n,D,D) bank states; p: layer params.
    Returns (o_arch (B,Ha,L,D), stats dict)."""
    B, Ha, L, D = q.shape
    n = S.shape[2]
    pr = prec(cfg.precision)
    R = make_sketch(D, cfg.sketch_m)
    T = sketch_bank(S, R)
    e = bank_energy(q, T, C, pr)                                   # (B,Ha,L,n)
    sf = frob2(S)                                                  # (B,Ha,n)
    qn2 = jnp.sum(q * q, axis=-1)                                  # (B,Ha,L)
    logr = log_ratio(e, sf[:, :, None, :], qn2[..., None], D)
    seg = jnp.arange(L) // C
    j = jnp.arange(n)
    age = jnp.maximum(seg[:, None] - j[None, :] - 1, 0).astype(jnp.float32)      # (L,n)
    valid = j[None, :] < seg[:, None]                                            # (L,n)
    tau = p["route_tau"][None, :, None, None]
    lam = p["route_lam"][None, :, None, None]
    null = tau * p["route_null"][None, :, None, None]
    scores = tau * logr - lam * jnp.log1p(age)
    a = bank_gate(scores, valid, null, cfg.top_k)
    if oracle is not None:                                         # oracle (B,L,n): diagnostics only
        a = jnp.broadcast_to(oracle[:, None, :, :n], a.shape) * valid
    out = routed_read(q, S, a, C, 1.0 / math.sqrt(D), pr)
    st = {}
    if want_stats:
        mass = jnp.sum(a, axis=-1)
        st = dict(read_mass=jnp.mean(mass),
                  read_age=jnp.sum(a * age) / (jnp.sum(a) + 1e-6),
                  top_logratio=jnp.mean(jnp.max(jnp.where(valid, logr, -1e3), axis=-1)))
    return out, st
