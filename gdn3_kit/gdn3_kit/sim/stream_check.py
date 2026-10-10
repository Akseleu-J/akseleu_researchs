"""numpy check: streaming A/B warm-up bookkeeping (cache.mixer_step) == fold_overlap semantics (mixer_fwd local mode)."""
import numpy as np
rng = np.random.default_rng(0)
D, C, W, n = 6, 16, 4, 5
L = n * C
q, k, v = (rng.normal(size=(L, D)) for _ in range(3)); k /= np.linalg.norm(k, axis=1, keepdims=True)
w = rng.uniform(size=(L, D)); b = rng.uniform(size=(L, D)); g = -rng.uniform(0, .1, size=(L, D))
h_init = 0.1 * rng.normal(size=(D, D))

def step(h, t):
    h = h * np.exp(g[t])[:, None]
    h = h + np.outer(k[t], w[t] * v[t] - (b[t] * k[t]) @ h)
    return h, (q[t] / np.sqrt(D)) @ h

# --- parallel semantics: segment j runs tokens [jC-W, (j+1)C) from h_init (zero-padded start)
o_par = np.zeros((L, D)); banks = []
for j in range(n):
    h = h_init.copy()
    for t in range(max(j * C - W, 0), (j + 1) * C):
        h, o = step(h, t)
        if t >= j * C: o_par[t] = o
    banks.append(h)

# --- streaming semantics
A, Bw = h_init.copy(), h_init.copy(); o_st = np.zeros((L, D)); arch = []
for t in range(L):
    u = t % C
    A, o = step(A, t)
    Bin = h_init if u == C - W else Bw
    B2, _ = step(Bin, t)
    B2 = B2 if u >= C - W else Bw
    o_st[t] = o
    if u == C - 1:
        arch.append(A.copy()); A = B2.copy(); Bw = h_init.copy()
    else:
        Bw = B2
print("max|o diff|:", np.abs(o_par - o_st).max(), " max|bank diff|:", max(np.abs(a - b_).max() for a, b_ in zip(arch, banks)))
