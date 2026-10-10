"""Two engines with IDENTICAL semantics (tested against each other):

  mixer_parallel : kernel backend (serial | pallas). Whole sequence at once; fixed segmentation; cells gdn2/qed/eda/dp via
                   sub-step interleaving; banks are independent segment states folded into the batch axis (one kernel call).
  mixer_scan     : token-level lax.scan carrying (state, bank stream, dynamic archive, exact KV ring). Supports every feature
                   (dla boundaries, osdn, hola, budgeted archive policies, int8 banks) and IS the streaming/decode step.

Per token t of segment s:   o_t = o_main(t) + tanh(gate) * sum_j a(t,j) * scale * (q_t * exp(G_t - Gend_j)) S_j     (banks j < s)
"""
import math
import jax
import jax.numpy as jnp

from . import ops, archive as ar
from .layer import substeps, token_update, interleave
from .routing import prec, make_sketch, sketch_bank, frob2, log_ratio, bank_gate, read_core


def _gate(p, cfg, force_gate):
    Ha = cfg.ha
    g = jnp.tanh(p["arch_gate"]) if force_gate is None else jnp.full((Ha,), force_gate, jnp.float32)
    return g


# =============================================================================================== parallel engine
def _fold(t, n, Cm):
    B, H, _, D = t.shape
    return t.reshape(B, H, n, Cm, D).transpose(0, 2, 1, 3, 4).reshape(B * n, H, Cm, D)


def _seg_split(t, n, Cm):
    B, H, _, D = t.shape
    return t.reshape(B, H, n, Cm, D).transpose(2, 0, 1, 3, 4)


def archive_read_parallel(p, cfg, q, S, Gtok, Gend, lamvec, ccql, oracle=None):
    """q (B,Ha,L,D); S (B,Ha,n,D,D); Gtok (B,Ha,L,D); Gend (B,Ha,n,D); lamvec (B,Ha,L,nl)|None; ccql (B,Ha,L)|None."""
    B, Ha, L, D = q.shape
    n, C = S.shape[2], cfg.seg
    pr, R = prec(cfg.precision), make_sketch(D, cfg.sketch_m)
    T = sketch_bank(S, R)
    cols = []
    for j in range(n):
        lo = (j + 1) * C
        if lo >= L:
            cols.append(jnp.zeros((B, Ha, L)))
            continue
        e = jax.checkpoint(lambda qs, Tj: jnp.sum(jnp.matmul(qs, Tj, precision=pr) ** 2, -1))(q[:, :, lo:], T[:, :, j])
        cols.append(jnp.pad(e, ((0, 0), (0, 0), (lo, 0))))
    e = jnp.stack(cols, -1)                                                       # (B,Ha,L,n)
    sf = frob2(S)
    logr = log_ratio(e, sf[:, :, None, :], jnp.sum(q * q, -1)[..., None], D)
    seg, j = jnp.arange(L) // C, jnp.arange(n)
    age = jnp.maximum(seg[:, None] - j[None, :] - 1, 0).astype(jnp.float32)      # (L,n)
    valid = j[None, :] < seg[:, None]
    if cfg.adaptive_lam:
        bucket = jnp.clip(jnp.floor(jnp.log2(1.0 + age)).astype(jnp.int32), 0, cfg.n_lv - 1)
        lam_eff = jnp.take_along_axis(lamvec, jnp.broadcast_to(bucket, (B, Ha, L, n)), axis=-1)
    else:
        lam_eff = p["route_lam"][None, :, None, None]
    tau = p["route_tau"][None, :, None, None]
    a = bank_gate(tau * logr - lam_eff * jnp.log1p(age), valid, tau * p["route_null"][None, :, None, None], cfg.top_k)
    if oracle is not None:                                   # diagnostic: externally supplied gate weights (B,L,n), causal via `valid`
        a = jnp.where(valid, oracle[:, None, :, :].astype(a.dtype), 0.0)
    scale, out = 1.0 / math.sqrt(D), jnp.zeros_like(q)
    for jj in range(n):
        lo = (jj + 1) * C
        if lo >= L:
            break
        Sj = S[:, :, jj]
        qe = q[:, :, lo:]
        if cfg.bank_decay:
            qe = qe * jnp.exp(jnp.minimum(Gtok[:, :, lo:] - Gend[:, :, jj][:, :, None, :], 0.0))

        def one(qe_, Sj_, sfj, aj, cl):
            mm = lambda x: jnp.matmul(x, Sj_, precision=pr)
            mmT = lambda x: jnp.matmul(x, jnp.swapaxes(Sj_, -1, -2), precision=pr)
            r = read_core(qe_, mm, mmT, jnp.sqrt(jnp.sum(Sj_ * Sj_, -1))[:, :, None, :], sfj[:, :, None, None],
                          cl, scale, cfg)
            return aj[..., None] * r
        cl = ccql[:, :, lo:, None] if cfg.ccq else 0.0
        out = out + jnp.pad(jax.checkpoint(one)(qe, Sj, sf[:, :, jj], a[:, :, lo:, jj], cl), ((0, 0), (0, 0), (lo, 0), (0, 0)))
    return out


def mixer_parallel(p, cfg, be, f, force_gate=None, want_cache=False, N=0, oracle=None):
    """f: features (B,L,H,D) [+extras]. Returns (o (B,L,H,D), carry|None). L must be a multiple of cfg.align(be.bt)."""
    B, L, H, D = f["q"].shape
    Ha, m, C = cfg.ha, cfg.m, cfg.seg
    if not cfg.parallel_ok():
        raise ValueError("parallel engine does not support cell=osdn / seg_mode=dla / hola: use engine='scan'")
    if L % cfg.align(be.bt):
        raise ValueError(f"L={L} must be a multiple of {cfg.align(be.bt)}")
    hm = lambda t: jnp.moveaxis(t, 1, 2)
    fh = {k_: hm(v_) for k_, v_ in f.items() if k_ not in ("lamvec", "ccql")}
    st = interleave(cfg, fh)                                                      # 6 x (B,H,L*m,D)
    n = L // C if Ha else 0
    S = None
    if cfg.mode == "snap":
        Cm = C * m
        xs = tuple(_seg_split(t, n, Cm) for t in st)

        def body(h, x):
            o_, h2 = be.call(*x, h)
            return h2, (o_, h2)
        hf, (os_, hs) = jax.lax.scan(body, jnp.zeros((B, H, D, D), jnp.float32), xs)
        o_all = os_.transpose(1, 2, 0, 3, 4).reshape(B, H, L * m, D)
        S = hs.transpose(1, 2, 0, 3, 4)[:, :Ha]
    else:
        o_all, hf = be.call(*st)
        if Ha:
            Cm = C * m
            fa = lambda t: _fold(t[:, :Ha], n, Cm)
            h0 = jnp.broadcast_to(p["h_init"][None], (B * n, Ha, D, D))
            _, S_f = be.call(jnp.zeros_like(fa(st[1])), *(fa(t) for t in st[1:]), h0)
            S = S_f.reshape(B, n, Ha, D, D).transpose(0, 2, 1, 3, 4)
    o = o_all[:, :, m - 1::m]
    carry = None
    Gtok = jnp.cumsum(fh["g"], axis=2)
    if Ha:
        Gend = Gtok[:, :Ha, C - 1::C]
        o_arch = archive_read_parallel(p, cfg, fh["q"][:, :Ha], S, Gtok[:, :Ha], Gend,
                                       hm(f["lamvec"]) if cfg.adaptive_lam else None,
                                       jnp.moveaxis(f["ccql"], 1, 2) if cfg.ccq else None, oracle)
        o = o + jnp.pad(_gate(p, cfg, force_gate)[None, :, None, None] * o_arch, ((0, 0), (0, H - Ha), (0, 0), (0, 0)))
    if want_cache:
        carry = dict(h=hf, prev_k=fh["k"][:, :, -1])
        if Ha:
            R = make_sketch(D, cfg.sketch_m)
            carry.update(A=jnp.broadcast_to(p["h_init"][None], (B, Ha, D, D)), gs=jnp.zeros((B, Ha, D)),
                         n=jnp.zeros((B, Ha), jnp.int32), nseg=jnp.full((B, Ha), n, jnp.int32),
                         arch=ar.export_archive(cfg, R, S, Gend, N))
    return jnp.moveaxis(o, 1, 2), carry


# =============================================================================================== scan / streaming engine
def init_carry(p, cfg, B, N):
    H, D, Ha, f32 = cfg.n_heads, cfg.d_head, cfg.ha, jnp.float32
    c = dict(h=jnp.zeros((B, H, D, D), f32), prev_k=jnp.zeros((B, H, D), f32))
    if cfg.cell == "osdn":
        c["d"] = jnp.ones((B, H, D), f32)
    if Ha:
        c.update(A=jnp.broadcast_to(p["h_init"][None], (B, Ha, D, D)), gs=jnp.zeros((B, Ha, D), f32),
                 n=jnp.zeros((B, Ha), jnp.int32), nseg=jnp.zeros((B, Ha), jnp.int32), arch=ar.init_arch(cfg, B, Ha, N))
    if cfg.hola:
        c.update(hK=jnp.zeros((B, H, cfg.hola, D), f32), hV=jnp.zeros((B, H, cfg.hola, D), f32),
                 hcnt=jnp.zeros((B, H), jnp.int32))
    return c


def make_body(p, cfg, force_gate=None):
    D, Ha, H = cfg.d_head, cfg.ha, cfg.n_heads
    scale = 1.0 / math.sqrt(D)
    R = make_sketch(D, cfg.sketch_m)
    hybrid = cfg.mode == "hybrid"
    EXTRA = ("gam", "e", "ke", "ve", "be")

    def body(c, x):
        c = dict(c)
        q, k, v, w, b, g = (x[n_] for n_ in "qkvwbg")
        ex = {n_: x[n_] for n_ in EXTRA if n_ in x}
        if cfg.cell == "osdn":
            ex["d"] = c["d"]
        subs, mi = substeps(cfg, q, k, v, w, b, g, ex)
        o_arch = None
        if Ha:
            gA = g[:, :Ha]
            n_ = c["n"]
            if cfg.seg_mode == "dla":
                sig = ops.dla_boundary_score(k[:, :Ha] - c["prev_k"][:, :Ha], b[:, :Ha], cfg.dla_mode)
                soft = jax.nn.sigmoid((sig - p["dla_theta"]) / cfg.dla_tau)
                cand = jax.lax.stop_gradient(soft) > 0.5
            else:
                cand = jnp.zeros(n_.shape, bool)
            hard = (cand & (n_ >= cfg.min_len)) | (n_ >= cfg.max_len)                 # boundary BEFORE this token
            hf_ = hard.astype(jnp.float32)
            bnd = hf_ + (soft - jax.lax.stop_gradient(soft)) if cfg.seg_mode == "dla" else hf_   # straight-through (value == hard)
            bank_src = c["A"] if hybrid else c["h"][:, :Ha]

            def do_close(arch):
                new = ar.append_batched(cfg, R, arch, bank_src, c["gs"], c["nseg"])
                return jax.tree_util.tree_map(
                    lambda a_new, a_old: jnp.where(hard.reshape(hard.shape + (1,) * (a_old.ndim - 2)), a_new, a_old), new, arch)
            arch = jax.lax.cond(jnp.any(hard), do_close, lambda a: a, c["arch"])
            if hybrid:
                bb = bnd[..., None, None]
                c["A"] = c["A"] * (1.0 - bb) + p["h_init"][None] * bb
            gs = c["gs"] * (1.0 - hf_[..., None])
            n_ = jnp.where(hard, 0, n_)
            nseg = c["nseg"] + hard.astype(jnp.int32)
            arch = dict(arch, acc=arch["acc"] + gA[:, :, None, :])                    # decay since each bank's end, incl. token t
            B_ = q.shape[0]
            lamvec = x["lamvec"] if cfg.adaptive_lam else jnp.zeros((B_, Ha, cfg.n_lv))
            ccql = x["ccql"] if cfg.ccq else jnp.zeros((B_, Ha))
            o_arch = ar.read_batched(cfg, R, q[:, :Ha], arch, lamvec, ccql, p["route_tau"], p["route_lam"],
                                     p["route_null"], nseg, scale)
            c.update(arch=arch, gs=gs + gA, n=n_ + 1, nseg=nseg)
            if hybrid:
                c["A"], _ = token_update(c["A"], subs)
        h, vns = token_update(c["h"], subs)
        o = scale * jnp.einsum("bhd,bhde->bhe", q, h, precision=jax.lax.Precision.HIGHEST)
        if cfg.cell == "osdn":
            eta = jnp.exp(p["log_eta"])[None, :, None]
            c["d"] = jnp.maximum(c["d"] + eta * (-k * ops._h_vec(h, vns[mi])), 1e-4)
        if Ha:
            o = o + jnp.pad(_gate(p, cfg, force_gate)[None, :, None] * o_arch, ((0, 0), (0, H - Ha), (0, 0)))
        if cfg.hola:
            sur = ops.hola_surprise(b, vns[mi])
            rd = ops.ring_read(q, c["hK"], c["hV"], c["hcnt"], jnp.exp(p["hola_lgamma"])[None, :], scale)
            o = o + jnp.tanh(p["hola_gate"])[None, :, None] * rd
            c["hK"], c["hV"], c["hcnt"] = ops.ring_insert(c["hK"], c["hV"], c["hcnt"], k, w * v, sur > cfg.hola_thr)
        c["h"], c["prev_k"] = h, k
        return c, o
    return body


def mixer_scan(p, cfg, f, carry, force_gate=None):
    """f: features (B,L,H,D); carry from init_carry/previous call. Returns (o (B,L,H,D), carry)."""
    xs = jax.tree_util.tree_map(lambda t: jnp.moveaxis(t, 1, 0), f)
    carry, o = jax.lax.scan(make_body(p, cfg, force_gate), carry, xs)
    return jnp.moveaxis(o, 0, 1), carry
