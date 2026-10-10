"""Module library: the 16 GDN-3 components as pure JAX functions (autodiff-safe, f32, Precision.HIGHEST).

Conventions: h (B,H,D,D) indexed [key,value]; per-token vectors (B,H,D); per-head scalars (B,H).
Readout o = scale * (h^T q).
"""
import math
import jax
import jax.numpy as jnp

HI = jax.lax.Precision.HIGHEST


def _ein(spec, *a):
    return jnp.einsum(spec, *a, precision=HI)


def _vec_h(x, h):            # x @ h   (...,D),(...,D,E) -> (...,E)
    return _ein("...d,...de->...e", x, h)


def _h_vec(h, y):            # h @ y   (...,D,E),(...,E) -> (...,D)
    return _ein("...de,...e->...d", h, y)


def _outer(a, b):
    return a[..., :, None] * b[..., None, :]


def _decay(h, g):
    return h * jnp.exp(jnp.minimum(g, 0.0))[..., :, None]


def _readout(h, q, scale):
    return scale * _vec_h(q, h)


def _bh(x):                  # (B,H) -> (B,H,1,1)
    return jnp.asarray(x)[..., None, None]


# ------------------------------------------------------------------ base primitive
def gdn_core(h, q, k, v, w, b, g):
    """One GDN-2 token (q unused). Returns (h_new, v_new)."""
    h = _decay(h, g)
    v_new = w * v - _vec_h(b * k, h)
    return h + _outer(k, v_new), v_new


def gdn_step(h, q, k, v, w, b, g):
    return gdn_core(h, q, k, v, w, b, g)[0]


# ------------------------------------------------------------------ A2 KDA
def kda_gate(z, decay_a, g_max):
    """g = -min(exp(decay_a) * softplus(z), g_max)  (broadcasting)."""
    return -jnp.minimum(jnp.exp(decay_a) * jax.nn.softplus(z), g_max)


def kda_decay(x, decay_a, w_d, b_d, g_max):
    """x (B,H,L,D), decay_a (H,D), w_d (D,D), b_d (D,) -> g (B,H,L,D) in [-g_max, 0]."""
    return kda_gate(_ein("bhld,de->bhle", x, w_d) + b_d, decay_a[None, :, None, :], g_max)


# ------------------------------------------------------------------ A3 QED
def qed_step(h, q, k, v, w, b, g, gamma_q, scale):
    """Decay -> erase along q_perp (q orthogonal to k) -> delta write. Returns (h, o)."""
    h = _decay(h, g)
    q_perp = q - (jnp.sum(q * k, -1, keepdims=True) / (jnp.sum(k * k, -1, keepdims=True) + 1e-6)) * k
    h = h - _bh(gamma_q) * _outer(q_perp, _vec_h(q_perp, h))
    v_new = w * v - _vec_h(b * k, h)
    h = h + _outer(k, v_new)
    return h, _readout(h, q, scale)


# ------------------------------------------------------------------ A11 EDA
def eda_step(h, q, k, v, w, b, g, e, gamma_e, scale):
    """Decay -> erase along address e -> delta write with key k. Returns (h, o)."""
    h = _decay(h, g)
    ee = jnp.sum(e * e, -1, keepdims=True) + 1e-6
    h = h - _bh(gamma_e) * _outer(e, _vec_h(e / ee, h))
    v_new = w * v - _vec_h(b * k, h)
    h = h + _outer(k, v_new)
    return h, _readout(h, q, scale)


# ------------------------------------------------------------------ A4 DeltaProduct
def deltaproduct_step(h, q, k, v, w, b, g, ks_extra, bs_extra, vs_extra, ws_extra, scale=1.0):
    """Main GDN-2 step then n_extra undecayed delta steps; extras are (B,H,n_extra,D). Returns (h, o)."""
    h = gdn_step(h, q, k, v, w, b, g)
    n = ks_extra.shape[-2]
    for i in range(n):
        v_new = ws_extra[..., i, :] * vs_extra[..., i, :] - _vec_h(bs_extra[..., i, :] * ks_extra[..., i, :], h)
        h = h + _outer(ks_extra[..., i, :], v_new)
    return h, _readout(h, q, scale)


# ------------------------------------------------------------------ A5 OSDN
def osdn_step(h, q, k, v, w, b, g, d, eta, scale=1.0):
    """Diagonal-preconditioned write key k_pc = k*d; d <- max(d + eta*grad_d, 1e-4), grad_d = -k*(h_new v_new).
    NOTE: grad_d is a HEURISTIC online-feedback signal, not an autodiff hypergradient (audit problem 8)."""
    h = _decay(h, g)
    k_pc = k * d
    v_new = w * v - _vec_h(b * k_pc, h)
    h = h + _outer(k_pc, v_new)
    d_new = jnp.maximum(d + eta * (-k * _h_vec(h, v_new)), 1e-4)
    return h, _readout(h, q, scale), d_new


# ------------------------------------------------------------------ B1 DLA
def dla_boundary_score(dx, beta=None, mode="residual"):
    """(...,D) -> (...). residual: ||dx||_2 ; beta: mean(beta)."""
    if mode == "residual":
        return jnp.linalg.norm(dx, axis=-1)
    if mode == "beta":
        return jnp.mean(beta, axis=-1)
    raise ValueError(mode)


def dla_scan(cand, min_len, max_len):
    """cand (B,H,L) bool -> boundary mask (B,H,L) bool. Greedy causal: boundary at t iff
    (cand[t] and t-last >= min_len) or t-last >= max_len."""
    def body(last, x):
        t, c = x
        n = t - last
        bnd = jnp.logical_or(jnp.logical_and(c, n >= min_len), n >= max_len)
        return jnp.where(bnd, t, last), bnd
    last0 = jnp.zeros(cand.shape[:-1], jnp.int32)
    _, bnds = jax.lax.scan(body, last0, (jnp.arange(cand.shape[-1], dtype=jnp.int32), jnp.moveaxis(cand, -1, 0)))
    return jnp.moveaxis(bnds, 0, -1)


def dla_segment(score, theta, min_len, max_len):
    """Causal greedy segmentation -> int32 segment id (B,H,L) (= #boundaries in [0..t]).
    Integer output has no gradient; use dla_boundaries_st for the straight-through version."""
    return jnp.cumsum(dla_scan(score > theta, min_len, max_len).astype(jnp.int32), axis=-1).astype(jnp.int32)


def dla_boundaries_st(score, theta, min_len, max_len, tau=1.0):
    """Float boundaries (B,H,L): hard + (soft - sg(soft)), soft = sigmoid((score-theta)/tau). Value == hard."""
    soft = jax.nn.sigmoid((score - theta) / tau)
    hard = dla_scan(jax.lax.stop_gradient(soft) > 0.5, min_len, max_len).astype(soft.dtype)
    return hard + (soft - jax.lax.stop_gradient(soft))


# ------------------------------------------------------------------ B2 Fenwick / adaptive lambda
def fenwick_build(S_list):
    """list of n (B,H,D,D) -> prefix (B,H,n+1,D,D), prefix[:,:,0] = 0."""
    S = jnp.stack([jnp.asarray(s) for s in S_list], axis=2)
    return jnp.concatenate([jnp.zeros_like(S[:, :, :1]), jnp.cumsum(S, axis=2)], axis=2)


def fenwick_read(prefix, start, end):   # range sum via prefix sums (a prefix-sum table, not a Fenwick tree)
    return prefix[:, :, end] - prefix[:, :, start]


def fenwick_num_levels(n):
    n = int(n)
    return 0 if n <= 1 else int(math.ceil(math.log2(n)))


def adaptive_lambda(x, w1, b1, w2, b2):
    """softplus(gelu(x@w1+b1)@w2+b2) > 0  (tanh-gelu)."""
    return jax.nn.softplus(_ein("...i,ij->...j", jax.nn.gelu(_ein("...i,ij->...j", x, w1) + b1), w2) + b2)


def adaptive_lambda_mlp(x, params, n_levels=None):
    """DEPRECATED thin wrapper around adaptive_lambda (n_levels ignored; kept for spec-test compatibility)."""
    return adaptive_lambda(x, params["w1"], params["b1"], params["w2"], params["b2"])


# ------------------------------------------------------------------ B4 SSE
def sse_topk_rows(h, query, top_k):
    """scores_i = sum_j h_ij query_j ; keep rows with score >= top_k-th largest, zero the rest."""
    s = _ein("...ij,...j->...i", h, query)
    kth = jax.lax.top_k(s, int(top_k))[0][..., -1:]
    return h * (s >= kth).astype(h.dtype)[..., None]


def sse_partitions(h, N):
    """(...,D,E) -> (...,N,D,E//N) column partitions."""
    N = int(N)
    D, E = h.shape[-2:]
    if E % N:
        raise ValueError("E must be divisible by N")
    return jnp.moveaxis(h.reshape(h.shape[:-1] + (N, E // N)), -2, -3)


# ------------------------------------------------------------------ B5 h_init
def init_h_init(key, H, D, scale=0.05):
    return scale * jax.random.normal(key, (H, D, D), jnp.float32)


def broadcast_h_init(h_init, B, n=None):
    lead = (B,) if n is None else (B, n)
    return jnp.broadcast_to(h_init, lead + h_init.shape)


# ------------------------------------------------------------------ B6 bank decay
def bank_read_with_decay(q, S_j, gc_t, gc_end_j, scale):
    """q,gc_t (B,H,L,D); S_j (B,H,D,D); gc_end_j (B,H,D): scale * (q*exp(gc_t-gc_end_j)) @ S_j (spec-exact)."""
    return scale * jnp.matmul(q * jnp.exp(gc_t - gc_end_j[..., None, :]), S_j, precision=HI)


# ------------------------------------------------------------------ B8 HOLA
def hola_surprise(beta, v_new):
    """||v_new||_2 * mean(beta) over the last axis (per token)."""
    return jnp.linalg.norm(v_new, axis=-1) * jnp.mean(beta, axis=-1)


def hola_insert(cache, k, v, surprise, threshold, max_size):
    """HOST helper: python control flow on a concrete value, NOT jit-able (eager only).
    cache=(K (n,D), V (n,D)); insert iff surprise > threshold; FIFO-drop beyond max_size. jit path: ring_insert."""
    K, V = cache
    if float(surprise) > threshold:
        K = jnp.concatenate([K, k[None]], 0)
        V = jnp.concatenate([V, v[None]], 0)
        if K.shape[0] > max_size:
            K, V = K[-max_size:], V[-max_size:]
    return K, V


def hola_read(q, cache, gamma, scale):
    """softmax(gamma*scale*q K^T) V ; empty cache -> zeros_like(q)."""
    K, V = cache
    if K.shape[0] == 0:
        return jnp.zeros_like(q)
    p = jax.nn.softmax(gamma * scale * jnp.matmul(q, K.T, precision=HI), axis=-1)
    return jnp.matmul(p, V, precision=HI)


def ring_insert(K, V, cnt, k, v, do):
    """jit FIFO ring. K,V (...,M,D); cnt (...) int32 (#inserted); k,v (...,D); do (...) bool."""
    M = K.shape[-2]
    oh = (jnp.arange(M) == (cnt % M)[..., None]) & do[..., None]
    return (jnp.where(oh[..., None], k[..., None, :], K), jnp.where(oh[..., None], v[..., None, :], V),
            cnt + do.astype(jnp.int32))


def ring_read(q, K, V, cnt, gamma, scale):
    """q (...,D); K,V (...,M,D); gamma broadcastable to (...). Masked sharp softmax; empty -> 0."""
    M = K.shape[-2]
    valid = jnp.arange(M) < jnp.minimum(cnt, M)[..., None]
    lg = jnp.where(valid, jnp.asarray(gamma)[..., None] * scale * _ein("...d,...md->...m", q, K), -1e30)
    return _ein("...m,...md->...d", jax.nn.softmax(lg, axis=-1) * valid, V)


# ------------------------------------------------------------------ C1 CCQ
def ccq_contract(q, Sigma, lam):
    """q (B,H,L,D), Sigma (B,H,D,D), lam (B,H,L): q - lam * (q @ Sigma)."""
    return q - lam[..., None] * jnp.matmul(q, Sigma, precision=HI)


def ccq_lambda(q, w_lam, b_lam):
    """sigmoid(q.w + b) in (0,1)."""
    return jax.nn.sigmoid(_ein("...d,d->...", q, w_lam) + b_lam)


# ------------------------------------------------------------------ D1 LeapQuant
def leap_quantize(S):
    """Per (D,D) block int8: scale = max|S|/127 (1 for all-zero blocks). Returns (int8, scale (...,1,1))."""
    S = jnp.asarray(S, jnp.float32)
    amax = jnp.max(jnp.abs(S), axis=(-2, -1), keepdims=True)
    scale = jnp.where(amax > 0, amax / 127.0, 1.0)
    return jnp.clip(jnp.round(S / scale), -127, 127).astype(jnp.int8), scale


def leap_compensator(S, top_k):
    """Top-k |S| outliers per block: (vals (...,k), flat_idx (...,k))."""
    flat = S.reshape(S.shape[:-2] + (-1,))
    _, idx = jax.lax.top_k(jnp.abs(flat), min(int(top_k), flat.shape[-1]))
    return jnp.take_along_axis(flat, idx, axis=-1), idx.astype(jnp.int32)


def leap_dequantize(S_int8, scale, compensator=None):
    """int8*scale; with compensator=(vals,idx) the outliers are restored exactly (scatter-free one-hot)."""
    S_hat = S_int8.astype(jnp.float32) * scale
    if compensator is None:
        return S_hat
    vals, idx = compensator
    flat = S_hat.reshape(S_hat.shape[:-2] + (-1,))
    oh = idx[..., None] == jnp.arange(flat.shape[-1])
    restored = jnp.sum(jnp.where(oh, vals[..., None], 0.0), axis=-2)
    return jnp.where(jnp.any(oh, axis=-2), restored, flat).reshape(S_hat.shape)


# ------------------------------------------------------------------ D2 OjaKV
def oja_update(W, x, eta):
    """y = x@W ; W <- W + eta*(outer(x,y) - W*y^2). W (D,r), x (D,)."""
    y = _ein("...d,...dr->...r", x, W)
    return W + eta * (_outer(x, y) - W * (y ** 2)[..., None, :])


def ojakv_compress(S, rank):
    U, s, Vt = jnp.linalg.svd(S, full_matrices=False)
    r = int(rank)
    return U[..., :, :r] * s[..., None, :r], Vt[..., :r, :]


def ojakv_decompress(U, V):
    return jnp.matmul(U, V, precision=HI)


# ------------------------------------------------------------------ D3 StateX
def expand_state(h_small, expand_dim):
    e = int(expand_dim)
    if e == 0:
        return h_small
    return jnp.pad(h_small, [(0, 0)] * (h_small.ndim - 2) + [(0, e), (0, e)])


def expand_projection(W, H, expand_dim, mode="zero"):
    """(d_in, H*D) -> (d_in, H*(D+e)): e zero channels appended to every head."""
    e = int(expand_dim)
    if e == 0:
        return W
    if mode != "zero":
        raise ValueError(mode)
    D = W.shape[-1] // H
    Wr = jnp.pad(W.reshape(W.shape[:-1] + (H, D)), [(0, 0)] * (W.ndim - 1) + [(0, 0), (0, e)])
    return Wr.reshape(W.shape[:-1] + (H * (D + e),))


# ------------------------------------------------------------------ honest aliases (audit P1.4): these are prefix sums
leveled_prefix_build = fenwick_build
prefix_read = fenwick_read
