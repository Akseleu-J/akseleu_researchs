"""Numpy-only checks of the math behind GDN-3 (no JAX needed). python -m tests.test_numpy_ref"""
import numpy as np
from bench.mechanism_np import delta_scan, read_banks_np, merge_pair, unit


def test_merge_exact_without_erase():
    """beta=0 (pure decayed linear attention): merged bank == bank of the union span (decay-only merge is exact)."""
    rng = np.random.default_rng(0); D, C = 16, 12
    k = unit(rng.normal(size=(2 * C, D))); v = rng.normal(size=(2 * C, D)); w = np.ones((2 * C, D))
    b = np.zeros((2 * C, D)); g = -0.05 * rng.random((2 * C, D)); z = np.zeros((D, D)); q0 = np.zeros((2 * C, D))
    _, Sa, _ = delta_scan(q0[:C], k[:C], v[:C], w[:C], b[:C], g[:C], z)
    _, Sb, _ = delta_scan(q0[C:], k[C:], v[C:], w[C:], b[C:], g[C:], z)
    _, Su, _ = delta_scan(q0, k, v, w, b, g, z)
    Sm, _ = merge_pair(Sa, g[:C].sum(0), Sb, g[C:].sum(0))
    err = np.abs(Sm - Su).max(); assert err < 1e-10, err
    return err


def test_merge_error_with_erase():
    """beta=1: decay-only merge is approximate (ignores erase of old content by new keys). Report relative error."""
    rng = np.random.default_rng(1); D, C = 32, 8
    k = unit(rng.normal(size=(2 * C, D))); v = rng.normal(size=(2 * C, D)); w = np.ones((2 * C, D))
    b = np.ones((2 * C, D)); g = np.zeros((2 * C, D)); z = np.zeros((D, D)); q0 = np.zeros((2 * C, D))
    _, Sa, _ = delta_scan(q0[:C], k[:C], v[:C], w[:C], b[:C], g[:C], z)
    _, Sb, _ = delta_scan(q0[C:], k[C:], v[C:], w[C:], b[C:], g[C:], z)
    _, Su, _ = delta_scan(q0, k, v, w, b, g, z)
    Sm, _ = merge_pair(Sa, g[:C].sum(0), Sb, g[C:].sum(0))
    return float(np.linalg.norm(Sm - Su) / np.linalg.norm(Su))


def test_bank_decomposition_exact():
    """Full state = A_b h_prev + B  ; with beta=0, A is pure decay, so main state == sum of decayed banks (exact)."""
    rng = np.random.default_rng(2); D, C, n = 16, 10, 4; L = C * n
    k = unit(rng.normal(size=(L, D))); v = rng.normal(size=(L, D)); w = np.ones((L, D)); b = np.zeros((L, D))
    g = -0.03 * rng.random((L, D)); z = np.zeros((D, D)); q0 = np.zeros((L, D))
    _, hf, _ = delta_scan(q0, k, v, w, b, g, z)
    G = np.cumsum(g, 0); tot = np.zeros((D, D))
    for j in range(n):
        sl = slice(j * C, (j + 1) * C); _, Sj, _ = delta_scan(q0[sl], k[sl], v[sl], w[sl], b[sl], g[sl], z)
        tot += np.exp(G[-1] - G[(j + 1) * C - 1])[:, None] * Sj
    err = np.abs(tot - hf).max(); assert err < 1e-10, err
    return err


def test_online_softmax_equals_dense():
    rng = np.random.default_rng(3); L, D, n = 20, 8, 5
    q = rng.normal(size=(L, D)); S = rng.normal(size=(n, D, D)); G = np.zeros((L, D)); Ge = np.zeros((n, D))
    seg = rng.integers(0, n + 1, L); tau, th = 3.0, 0.7
    got = read_banks_np(q, S, G, Ge, seg, tau, th)
    ref = np.zeros((L, D))
    for t in range(L):
        js = [j for j in range(n) if j < seg[t]]
        if not js: continue
        r = np.stack([q[t] @ S[j] for j in js]); s = tau * np.sqrt((r * r).mean(-1) + 1e-6)
        lg = np.concatenate([[th], s]); p = np.exp(lg - lg.max()); p /= p.sum()
        ref[t] = (p[1:, None] * r).sum(0)
    err = np.abs(got - ref).max(); assert err < 1e-10, err
    assert np.abs(got[seg == 0]).max() == 0.0     # no valid bank -> exactly zero
    return err


def test_causal_validity_mask():
    """tokens of segment i may only read banks j < i."""
    L, C = 32, 8; seg = np.arange(L) // C; n = L // C
    valid = np.arange(n)[None, :] < seg[:, None]
    assert valid[:C].sum() == 0 and valid[C:2 * C, 0].all() and not valid[C:2 * C, 1].any()


if __name__ == "__main__":
    print("merge exact (beta=0)     max|err| =", test_merge_exact_without_erase())
    print("merge approx (beta=1)    rel err  =", round(test_merge_error_with_erase(), 3))
    print("bank decomposition exact max|err| =", test_bank_decomposition_exact())
    print("online softmax == dense  max|err| =", test_online_softmax_equals_dense())
    test_causal_validity_mask(); print("validity mask ok")
