"""GDN-3 test suite (real JAX, serial kernel backend, CPU-friendly tiny shapes).  Run:  python tests/run.py [--only substr] [-x]
Pallas-vs-serial tests live in test_pallas.py (TPU only)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp

from gdn3 import ops, archive as ar, tasks
from gdn3.config import G3Config
from gdn3.backend import make_backend, serial_backend
from gdn3.layer import substeps, token_update, layer_pre, layer_post
from gdn3.model import (init_params, forward, prefill, decode_step, init_caches, masked_ce, lm_loss, count_params)
from gdn3.evaluate import eval_mqar, p_star, eval_mqar_stream
from gdn3.routing import make_sketch

TOL = 2e-4


def mk(**over):
    kw = dict(vocab=48, d_model=32, n_heads=2, d_head=16, n_layers=2, mlp_mult=2, seg=16, top_k=2, sketch_dim=8)
    kw.update(over)
    cfg = G3Config(**kw)
    return cfg, make_backend("serial", cfg.d_head)


def mk_params(cfg, seed=0, decay_boost=2.0):
    p = init_params(jax.random.PRNGKey(seed), cfg)
    for i, lp in enumerate(p["layers"]):
        lp["decay_a"] = lp["decay_a"] + decay_boost
        if cfg.ha:
            k1, k2 = jax.random.split(jax.random.PRNGKey(100 + seed * 10 + i))
            lp["h_init"] = 0.05 * jax.random.normal(k1, lp["h_init"].shape)
            lp["arch_gate"] = jnp.full_like(lp["arch_gate"], 0.7)
            lp["route_tau"] = lp["route_tau"] * (1.0 + 0.2 * jax.random.normal(k2, lp["route_tau"].shape))
    return p


def ids_of(cfg, B, L, seed=1):
    return jax.random.randint(jax.random.PRNGKey(seed), (B, L), 0, cfg.vocab)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))


def rand_in(key, B, H, L, D):
    ks = jax.random.split(key, 6)
    l2 = lambda t: t * jax.lax.rsqrt(jnp.sum(t * t, -1, keepdims=True) + 1e-6)
    return (l2(jax.random.normal(ks[0], (B, H, L, D))), l2(jax.random.normal(ks[1], (B, H, L, D))),
            jax.random.normal(ks[2], (B, H, L, D)) * 0.5, jax.nn.sigmoid(jax.random.normal(ks[3], (B, H, L, D))),
            jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, L, D))), -jax.nn.softplus(jax.random.normal(ks[5], (B, H, L, D))) * 0.2)


def np_serial(q, k, v, w, b, g, h0, scale):
    q, k, v, w, b, g, h = (np.asarray(x, np.float64) for x in (q, k, v, w, b, g, h0))
    o = np.zeros_like(q)
    for t in range(q.shape[2]):
        h = h * np.exp(np.minimum(g[:, :, t], 0.0))[..., :, None]
        vn = w[:, :, t] * v[:, :, t] - np.einsum("bhd,bhde->bhe", b[:, :, t] * k[:, :, t], h)
        h = h + k[:, :, t][..., :, None] * vn[..., None, :]
        o[:, :, t] = scale * np.einsum("bhde,bhd->bhe", h, q[:, :, t])
    return o, h


# ============================================================================ kernel / math
def test_serial_matches_numpy_f64():
    B, H, L, D = 1, 2, 48, 16
    q, k, v, w, b, g = rand_in(jax.random.PRNGKey(0), B, H, L, D)
    h0 = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (B, H, D, D))
    o, hf = serial_backend(D).call(q, k, v, w, b, g, h0)
    o_ref, h_ref = np_serial(q, k, v, w, b, g, h0, 1 / math.sqrt(D))
    assert rel(o, o_ref) < 1e-5 and rel(hf, h_ref) < 1e-5, (rel(o, o_ref), rel(hf, h_ref))


def test_chunk_invariance_and_split_grads():
    B, H, L, D, cut = 1, 2, 48, 16, 20
    be = serial_backend(D)
    a = rand_in(jax.random.PRNGKey(2), B, H, L, D)
    ct = jax.random.normal(jax.random.PRNGKey(3), (B, H, L, D))
    h0 = jnp.zeros((B, H, D, D))
    o, _ = be.call(*a)
    o1, h1 = be.call(*(t[:, :, :cut] for t in a))
    o2, _ = be.call(*(t[:, :, cut:] for t in a), h1)
    assert rel(jnp.concatenate([o1, o2], 2), o) < 1e-5

    full = lambda *x: jnp.sum(be.call(*x[:6])[0] * ct)

    def split(*x):
        o1, h1 = be.call(*(t[:, :, :cut] for t in x[:6]), x[6])
        o2, _ = be.call(*(t[:, :, cut:] for t in x[:6]), h1)
        return jnp.sum(o1 * ct[:, :, :cut]) + jnp.sum(o2 * ct[:, :, cut:])
    gf = jax.grad(lambda *x: full(*x), argnums=range(6))(*a, h0)
    gs = jax.grad(split, argnums=range(6))(*a, h0)
    for n, x, y in zip("qkvwbg", gf, gs):
        assert rel(y, x) < 1e-4, n


def test_substeps_equal_ops_cells():
    """Sub-step formulation (the one used by the scan engine AND the kernel interleaving) == ops.* spec functions."""
    B, H, D = 2, 2, 8
    r = np.random.default_rng(0)
    f = lambda *s, sc=1.0: jnp.asarray(sc * r.standard_normal(s), jnp.float32)
    unit = lambda t: t / jnp.linalg.norm(t, axis=-1, keepdims=True)
    h = f(B, H, D, D, sc=0.3); q = f(B, H, D, sc=0.5); k = unit(f(B, H, D)); v = f(B, H, D)
    w = jnp.asarray(r.uniform(0.5, 1, (B, H, D)), jnp.float32); b = jnp.asarray(r.uniform(0.1, 0.9, (B, H, D)), jnp.float32)
    g = -jnp.asarray(r.uniform(0, 0.5, (B, H, D)), jnp.float32)
    sc = 0.7
    gam = jnp.asarray(r.uniform(0.1, 0.9, (B, H)), jnp.float32)
    for cell in ("qed", "eda", "dp"):
        cfg = G3Config(d_head=D, n_heads=H, cell=cell, dp_extra=2)
        ex = {}
        if cell in ("qed", "eda"):
            ex["gam"] = gam
        if cell == "eda":
            ex["e"] = unit(f(B, H, D))
        if cell == "dp":
            ex.update(ke=unit(f(B, H, 2, D)), ve=f(B, H, 2, D), be=jnp.asarray(r.uniform(0.1, 0.9, (B, H, 2, D)), jnp.float32))
        subs, _ = substeps(cfg, q, k, v, w, b, g, ex)
        h_sub, _ = token_update(h, subs)
        if cell == "qed":
            h_ref, _ = ops.qed_step(h, q, k, v, w, b, g, gam, sc)
        elif cell == "eda":
            h_ref, _ = ops.eda_step(h, q, k, v, w, b, g, ex["e"], gam, sc)
        else:
            h_ref, _ = ops.deltaproduct_step(h, q, k, v, w, b, g, ex["ke"], ex["be"], ex["ve"], jnp.broadcast_to(w[:, :, None], (B, H, 2, D)), sc)
        assert rel(h_sub, h_ref) < 1e-5, (cell, rel(h_sub, h_ref))


def test_ops_hola_ring_equals_host_helper():
    """jit ring buffer == host FIFO helper (incl. wrap-around at cnt % M), insert only above threshold."""
    D, M = 4, 3
    r = np.random.default_rng(0)
    ks, vs = r.standard_normal((9, D)).astype(np.float32), r.standard_normal((9, D)).astype(np.float32)
    sur = np.array([.9, .1, .9, .9, .9, .1, .9, .9, .9])
    cache = (jnp.zeros((0, D)), jnp.zeros((0, D)))
    K, V, cnt = jnp.zeros((M, D)), jnp.zeros((M, D)), jnp.int32(0)
    ins = jax.jit(ops.ring_insert)
    for i in range(9):
        cache = ops.hola_insert(cache, jnp.asarray(ks[i]), jnp.asarray(vs[i]), float(sur[i]), 0.5, M)
        K, V, cnt = ins(K, V, cnt, jnp.asarray(ks[i]), jnp.asarray(vs[i]), jnp.asarray(sur[i] > 0.5))
    q = jnp.asarray(r.standard_normal((2, D)).astype(np.float32))
    host = ops.hola_read(q, cache, 3.0, 0.5)
    ring = jax.vmap(lambda qq: ops.ring_read(qq, K, V, cnt, 3.0, 0.5))(q)
    assert rel(ring, host) < 1e-5, rel(ring, host)
    assert int(cnt) == int((sur > 0.5).sum())


# ============================================================================ archive
def _rand_bank(key, D):
    return jax.random.normal(key, (D, D)) * 0.3


def test_merge_exact_without_erase():
    """beta=0: merged bank == bank of the union span (decay-only merge is exact)."""
    cfg, _ = mk(sketch_dim=0, quant="none")
    D, C = 8, 6
    r = np.random.default_rng(1)
    k = jnp.asarray(r.standard_normal((2 * C, D)), jnp.float32); v = jnp.asarray(r.standard_normal((2 * C, D)), jnp.float32)
    g = -jnp.asarray(r.uniform(0, 0.3, (2 * C, D)), jnp.float32)
    one, z = jnp.ones((D,)), jnp.zeros((D,))
    run = lambda sl: jax.lax.scan(lambda h, x: (ops.gdn_core(h, None, x[0], x[1], one, z, x[2])[0], None),
                                  jnp.zeros((D, D)), (k[sl], v[sl], g[sl]))[0]
    Sa, Sb, Su = run(slice(0, C)), run(slice(C, 2 * C)), run(slice(0, 2 * C))
    a = ar.entry_new(cfg, None, Sa, g[:C].sum(0), 0)
    b = ar.entry_new(cfg, None, Sb, g[C:].sum(0), 1)
    m = ar.merge_entries(cfg, None, a, b)
    assert rel(m["S"], Su) < 1e-5 and rel(m["span"], g.sum(0)) < 1e-6


def test_archive_policies_invariants():
    """After any number of appends: slots chronological & compacted, count bounded; merge/fenwick keep the decayed
    aggregate  sum_j exp(acc_j) S_j  EXACTLY (decay-correct superposition); fifo only drops."""
    D, n = 6, 23
    for policy, N in (("fifo", 4), ("merge", 4), ("fenwick", 8)):
        cfg, _ = mk(sketch_dim=0, bank_policy=policy, per_level=2, max_banks=N, d_head=D)
        arch = ar.init_arch(cfg, 1, 1, N)
        r = np.random.default_rng(3)
        agg = np.zeros((D, D))
        for j in range(n):
            S = jnp.asarray(r.standard_normal((1, 1, D, D)) * 0.3, jnp.float32)
            span = -jnp.asarray(r.uniform(0, 0.4, (1, 1, D)), jnp.float32)
            arch = dict(arch, acc=arch["acc"] + span[:, :, None, :])
            agg = np.exp(np.asarray(span[0, 0]))[:, None] * agg + np.asarray(S[0, 0])
            arch = ar.append_batched(cfg, None, arch, S, span, jnp.full((1, 1), j, jnp.int32))
            sid = np.asarray(arch["sid"][0, 0])
            cnt = int((sid >= 0).sum())
            assert cnt <= N and (sid[:cnt] >= 0).all() and (sid[cnt:] == -1).all(), (policy, j, sid)
            assert (np.diff(sid[:cnt]) > 0).all(), (policy, sid)
        got = (np.exp(np.asarray(arch["acc"][0, 0]))[:, :, None] * np.asarray(arch["S"][0, 0])).sum(0)
        if policy != "fifo":
            assert np.abs(got - agg).max() < 1e-4, (policy, np.abs(got - agg).max())
        if policy == "fenwick":
            lv = np.asarray(arch["lvl"][0, 0])[:cnt]
            assert (np.diff(lv) <= 0).all(), lv                      # older banks sit on higher levels
            assert cnt <= cfg.per_level * (int(math.log2(n)) + 2), cnt


def test_export_archive_equals_stream_appends():
    cfg, _ = mk(bank_policy="merge", max_banks=3)
    B, Ha, n, D = 1, 2, 7, cfg.d_head
    S = jax.random.normal(jax.random.PRNGKey(0), (B, Ha, n, D, D)) * 0.2
    Gend = -jnp.cumsum(jax.random.uniform(jax.random.PRNGKey(1), (B, Ha, n, D)) * 0.3, axis=2)
    R = make_sketch(D, cfg.sketch_m)
    a = ar.export_archive(cfg, R, S, Gend, 3)
    cnt = int((np.asarray(a["sid"]) >= 0).sum(-1).max())
    assert cnt == 3 and np.isfinite(np.asarray(a["S"])).all()
    full = ar.export_archive(cfg.with_(max_banks=0), R, S, Gend, 8)
    # aggregate preserved by the merge policy
    agg = lambda x: (np.exp(np.asarray(x["acc"]))[..., None] * np.asarray(x["S"])).sum(2)
    assert np.abs(agg(a) - agg(full)).max() < 1e-4


# ============================================================================ engines
CFGS = {
    "hybrid": dict(),
    "snap": dict(mode="snap"),
    "qed+adaptive+ccq": dict(cell="qed", adaptive_lam=True, ccq=True),
    "eda+sse+kda+nodecay": dict(cell="eda", sse_k=4, decay_mode="kda", bank_decay=False),
    "dp2+head+topk1+exact": dict(cell="dp", dp_extra=2, decay_mode="head", top_k=1, sketch_dim=0),
    "adaptive_auto_levels": dict(adaptive_lam=True, n_levels=0, max_banks=8),
}


def _check_parallel_eq_scan(name):
    cfg, be = mk(**CFGS[name])
    p = mk_params(cfg)
    ids = ids_of(cfg, 2, 64)
    a = forward(p, ids, cfg, be, "parallel")[0]
    b = forward(p, ids, cfg, be, "scan")[0]
    assert np.isfinite(np.asarray(a)).all()
    assert rel(b, a) < TOL, f"{name}: scan vs parallel {rel(b, a):.2e}"
    # the archive must actually matter, otherwise equality above is vacuous
    z = forward(p, ids, cfg, be, "parallel", force_gate=0.0)[0]
    assert np.abs(np.asarray(a) - np.asarray(z))[:, 16:].max() > 1e-4 and np.abs(np.asarray(a) - np.asarray(z))[:, :16].max() < 1e-5


def test_parallel_eq_scan_hybrid(): _check_parallel_eq_scan("hybrid")
def test_parallel_eq_scan_snap(): _check_parallel_eq_scan("snap")
def test_parallel_eq_scan_qed_adaptive_ccq(): _check_parallel_eq_scan("qed+adaptive+ccq")
def test_parallel_eq_scan_eda_sse_kda(): _check_parallel_eq_scan("eda+sse+kda+nodecay")
def test_parallel_eq_scan_dp_head_exact(): _check_parallel_eq_scan("dp2+head+topk1+exact")


def test_causality_both_engines():
    cfg, be = mk(cell="qed", adaptive_lam=True, ccq=True)
    p = mk_params(cfg)
    ids = ids_of(cfg, 2, 64)
    for engine in ("parallel", "scan"):
        base = forward(p, ids, cfg, be, engine)[0]
        for cut in (1, 15, 16, 19, 32):
            ids2 = ids.at[:, cut:].set((ids[:, cut:] + 7) % cfg.vocab)
            d = float(jnp.abs(forward(p, ids2, cfg, be, engine)[0] - base)[:, :cut].max())
            assert d < 1e-5, f"{engine} cut={cut} leak {d:.2e}"


def test_gate0_equals_gdn2():
    for engine in ("parallel", "scan"):
        cfg, be = mk()
        p = mk_params(cfg)
        ids = ids_of(cfg, 2, 64)
        a = forward(p, ids, cfg, be, engine, force_gate=0.0)[0]
        b = forward(p, ids, cfg.with_(mode="gdn2"), be, engine)[0]
        assert rel(a, b) < 1e-5, (engine, rel(a, b))


def test_end_to_end_gdn2_equivalence():
    """All flags off (cell gdn2, mode gdn2, head decay) == an independent float64 numpy GDN-2 mixer under the same layers."""
    cfg, be = mk(mode="gdn2", decay_mode="head")
    p = mk_params(cfg)
    ids = ids_of(cfg, 2, 48)
    got = forward(p, ids, cfg, be, "auto")[0]
    x = p["embed"][ids]
    for lp in p["layers"]:
        xn, f, _ = layer_pre(lp, cfg, x)
        hm = lambda t: np.asarray(jnp.moveaxis(t, 1, 2))
        o, _ = np_serial(*(hm(f[n]) for n in "qkvwbg"), np.zeros((2, cfg.n_heads, cfg.d_head, cfg.d_head)), 1 / math.sqrt(cfg.d_head))
        x = layer_post(lp, cfg, x, xn, jnp.moveaxis(jnp.asarray(o, jnp.float32), 1, 2))
    from gdn3.layer import rmsnorm
    ref = rmsnorm(x, p["final_norm"]) @ p["embed"].T
    assert rel(got, ref) < 1e-4, rel(got, ref)
    # gdn2 with every extra switched on but gated off is the same function
    cfg2, _ = mk(decay_mode="head", ccq=True, adaptive_lam=True, sse_k=4, hola=0)
    p2 = mk_params(cfg2)
    for lp, l1 in zip(p2["layers"], p["layers"]):
        for k_ in l1:
            lp[k_] = l1[k_]
    assert rel(forward(p2, ids, cfg2, be, "scan", force_gate=0.0)[0], ref) < 1e-4


def test_streaming_equals_batch():
    cfg, be = mk()
    p = mk_params(cfg)
    ids = ids_of(cfg, 2, 80)
    ref = forward(p, ids, cfg, be, "parallel")[0]
    n = 80 // cfg.seg + 1
    # (a) pure token-by-token decode from an empty cache
    caches = init_caches(p, cfg, 2, n)
    outs = []
    step = jax.jit(lambda c, t: decode_step(p, c, t, cfg))
    for t in range(80):
        lg, caches = step(caches, ids[:, t])
        outs.append(lg)
    assert rel(jnp.stack(outs, 1), ref) < TOL, rel(jnp.stack(outs, 1), ref)
    # (b) prefill (parallel prefix + scan tail) with different block patterns, then decode
    for first, rest in ((64, [16]), (48, [7, 9]), (70, [10]), (80, [])):
        lg, caches = prefill(p, ids[:, :first], cfg, be, n)
        parts, i = [lg], first
        for m in rest:
            lg, caches = forward(p, ids[:, i:i + m], cfg, be, "scan", caches, True, n)
            parts.append(lg); i += m
        e = rel(jnp.concatenate(parts, 1), ref[:, :i])
        assert e < TOL, (first, rest, e)


def test_streaming_all_cells():
    for name in ("snap", "qed+adaptive+ccq", "eda+sse+kda+nodecay", "dp2+head+topk1+exact"):
        cfg, be = mk(**CFGS[name])
        p = mk_params(cfg)
        ids = ids_of(cfg, 1, 64)
        ref = forward(p, ids, cfg, be, "parallel")[0]
        lg, c = prefill(p, ids[:, :48], cfg, be, 5)
        lg2, _ = forward(p, ids[:, 48:], cfg, be, "scan", c, True, 5)
        e = rel(jnp.concatenate([lg, lg2], 1), ref)
        assert e < TOL, (name, e)


def test_budget_policies_bounded():
    cfg0, be = mk()
    p = mk_params(cfg0)
    ids = ids_of(cfg0, 1, 96)
    full = forward(p, ids, cfg0, be, "scan")[0]
    for policy in ("fifo", "merge", "fenwick"):
        cfg = cfg0.with_(bank_policy=policy, max_banks=3)
        lg, caches = forward(p, ids, cfg, be, "scan", None, True, 3)
        assert np.isfinite(np.asarray(lg)).all()
        assert int((np.asarray(caches[0]["mix"]["arch"]["sid"]) >= 0).sum(-1).max()) <= 3
        assert rel(lg, full) > 1e-4                      # budget must change the answer (6 banks > 3 slots)
        assert rel(lg[:, :48], full[:, :48]) < 1e-5 if policy == "fifo" else True   # first 3 banks fit: identical until bank 4


def test_quant_int8_close_to_float():
    cfg, be = mk(max_banks=6)
    p = mk_params(cfg)
    ids = ids_of(cfg, 1, 96)
    a = forward(p, ids, cfg, be, "scan")[0]
    b = forward(p, ids, cfg.with_(quant="int8", quant_k=8), be, "scan")[0]
    e = rel(b, a)
    assert 0 < e < 5e-2, e


def test_dla_adaptive_segmentation():
    cfg, be = mk(seg_mode="dla", dla_min=4, dla_max=12, dla_theta=0.3, max_banks=20)
    assert not cfg.parallel_ok()
    p = mk_params(cfg)
    ids = ids_of(cfg, 2, 64)
    lg, caches = forward(p, ids, cfg, be, "auto", None, True, 20)
    assert np.isfinite(np.asarray(lg)).all()
    nseg = np.asarray(caches[0]["mix"]["nseg"])
    assert (nseg >= 64 // 13).all() and (nseg <= 64 // 4).all(), nseg
    # theta=inf & min=max=seg  ==  fixed segmentation
    fx, _ = mk()
    dl, _ = mk(seg_mode="dla", dla_min=16, dla_max=16, dla_theta=1e9)
    pf = mk_params(fx)
    pd = mk_params(dl)
    for a_, b_ in zip(pf["layers"], pd["layers"]):
        for k_ in a_:
            b_[k_] = a_[k_]
    pd["embed"], pd["final_norm"] = pf["embed"], pf["final_norm"]
    assert rel(forward(pd, ids, dl, be)[0], forward(pf, ids, fx, be, "scan")[0]) < 1e-6
    # straight-through gradient reaches dla_theta
    g = jax.grad(lambda pp: lm_loss(pp, cfg, be, ids))(p)
    th = np.asarray(g["layers"][0]["dla_theta"])
    assert np.isfinite(th).all()


def test_dla_ops_straight_through():
    sc = jnp.asarray(np.random.default_rng(0).uniform(0.3, 0.7, (2, 2, 32)), jnp.float32)
    hard = ops.dla_scan(sc > 0.5, 2, 8)
    st = ops.dla_boundaries_st(sc, 0.5, 2, 8, 1.0)
    assert float(jnp.abs(st - hard.astype(jnp.float32)).max()) < 1e-6          # value == hard
    gr = jax.grad(lambda th: jnp.sum(ops.dla_boundaries_st(sc, th, 2, 8, 1.0) ** 2))(0.5)
    assert float(jnp.abs(gr)) > 1e-6
    sg = ops.dla_segment(sc, 0.5, 2, 8)
    assert sg.dtype == jnp.int32 and int(sg[..., -1].min()) >= 32 // 8 - 1


def test_hola_exact_cache_scan_engine():
    cfg, be = mk(hola=4, hola_thr=0.0)
    p = mk_params(cfg)
    ids = ids_of(cfg, 2, 48)
    a = forward(p, ids, cfg, be, "auto")[0]
    assert np.isfinite(np.asarray(a)).all()
    for lp in p["layers"]:
        lp["hola_gate"] = jnp.zeros_like(lp["hola_gate"])
    off = forward(p, ids, cfg, be)[0]
    ref = forward(p, ids, cfg.with_(hola=0), be, "scan")[0]
    assert rel(off, ref) < 1e-6                                     # gate 0 -> no effect
    ids2 = ids.at[:, 20:].set((ids[:, 20:] + 5) % cfg.vocab)
    for lp in p["layers"]:
        lp["hola_gate"] = jnp.full_like(lp["hola_gate"], 0.5)
    on = forward(p, ids, cfg, be)[0]
    assert rel(on, off) > 1e-5
    d = float(jnp.abs(forward(p, ids2, cfg, be)[0] - on)[:, :20].max())
    assert d < 1e-5, d                                               # causal
    g = jax.grad(lambda pp: lm_loss(pp, cfg, be, ids))(p)
    assert float(jnp.abs(g["layers"][0]["hola_gate"]).sum()) > 0


def test_osdn_scan_engine():
    cfg, be = mk(cell="osdn", mode="gdn2")
    p = mk_params(cfg)
    ids = ids_of(cfg, 1, 40)
    base = forward(p, ids, cfg.with_(cell="gdn2"), be, "scan")[0]
    for lp in p["layers"]:
        lp["log_eta"] = jnp.full_like(lp["log_eta"], -40.0)         # eta -> 0, d stays 1  => GDN-2
    assert rel(forward(p, ids, cfg, be)[0], base) < 1e-6
    for lp in p["layers"]:
        lp["log_eta"] = jnp.full_like(lp["log_eta"], math.log(0.3))
    assert rel(forward(p, ids, cfg, be)[0], base) > 1e-4
    try:
        forward(p, ids, cfg, be, "parallel"); raise AssertionError("osdn must be rejected by the parallel engine")
    except ValueError:
        pass


# ============================================================================ gradients / training
def test_gradients_flow_and_engines_agree():
    cfg, be = mk(adaptive_lam=True, ccq=True)
    p = mk_params(cfg)
    ids = ids_of(cfg, 2, 64)
    gp = jax.grad(lambda pp: lm_loss(pp, cfg, be, ids, "parallel"))(p)
    gs = jax.grad(lambda pp: lm_loss(pp, cfg, be, ids, "scan"))(p)
    l0p, l0s = gp["layers"][0], gs["layers"][0]
    for k_ in ("h_init", "arch_gate", "route_tau", "route_null", "decay_a", "wq", "wk", "lam_w2", "ccq_b"):
        assert np.isfinite(np.asarray(l0p[k_])).all() and float(jnp.abs(l0p[k_]).sum()) > 0, f"no grad to {k_}"
    for (path, a), (_, b) in zip(jax.tree_util.tree_leaves_with_path(gp), jax.tree_util.tree_leaves_with_path(gs)):
        na, nb = float(jnp.linalg.norm(a)), float(jnp.linalg.norm(b))
        if max(na, nb) < 1e-9:
            continue
        e = float(jnp.linalg.norm(a - b)) / (max(na, nb) + 1e-12)
        assert e < 2e-3, (jax.tree_util.keystr(path), e)


def test_mqar_data_disjoint_vocab():
    rng = np.random.default_rng(0)
    P, nq, L = 6, 3, 64
    ids, mask, qi = tasks.make_mqar(rng, 4, P, nq, L)
    assert ids.shape == (4, L) and int(mask.sum()) == 4 * nq
    for b in range(4):
        facts = {(ids[b, 3 * i], ids[b, 3 * i + 1]): ids[b, 3 * i + 2] for i in range(P)}
        base = L - 4 * nq
        assert (ids[b, :3 * P:3] < tasks.NK).all() and (ids[b, 2:3 * P:3] >= tasks.VAL0).all() and (ids[b, 2:3 * P:3] < tasks.FIL0).all()
        assert ((ids[b, 3 * P:base] >= tasks.FIL0) & (ids[b, 3 * P:base] < tasks.SEP)).all()
        for i in range(nq):
            assert ids[b, base + 4 * i] == tasks.SEP and mask[b, base + 4 * i + 3]
            assert facts[(ids[b, base + 4 * i + 1], ids[b, base + 4 * i + 2])] == ids[b, base + 4 * i + 3]


def test_train_and_eval_smoke():
    from gdn3.train import train
    cfg, be = mk(vocab=tasks.VOCAB, seg=16)
    fn = tasks.mqar_batch_fn([2, 3], 2, 64, 4)
    params, hist = train(cfg, be, fn, steps=40, lr=3e-3, warmup=5, log_every=1, verbose=False)
    ls = [h["loss"] for h in hist]
    assert all(np.isfinite(ls)) and np.mean(ls[-5:]) < np.mean(ls[:5]), (ls[:5], ls[-5:])
    curve = eval_mqar(params, cfg, be, [2, 3], 64, n_q=2, B=4, n_batches=1)
    assert set(curve) == {2, 3} and p_star({2: dict(acc=1.0), 4: dict(acc=0.95), 8: dict(acc=0.5)}) == 4
    assert p_star({2: dict(acc=.1)}) == 0
    acc = eval_mqar_stream(params, cfg.with_(bank_policy="fifo"), be, 3, 64, 2, n_q=2, B=2, n_batches=1)
    assert 0.0 <= acc <= 1.0
    assert count_params(params) > 0


# ============================================================================ audit P0 additions
def test_route_lam_gradient_nonadaptive():
    """route_lam is the recency prior ONLY when adaptive_lam=False; with adaptive_lam=True it is unused (grad == 0 by
    construction, lam comes from lam_w*). The earlier 'no grad to route_lam' failure was a test mis-configuration."""
    cfg, be = mk()                                     # adaptive_lam=False
    p, ids = mk_params(cfg), ids_of(cfg, 2, 64)
    g = jax.grad(lambda pp: lm_loss(pp, cfg, be, ids, "parallel"))(p)
    assert float(jnp.abs(g["layers"][0]["route_lam"]).sum()) > 0, "route_lam gets no gradient"
    cfg2, _ = mk(adaptive_lam=True)
    p2 = mk_params(cfg2)
    g2 = jax.grad(lambda pp: lm_loss(pp, cfg2, be, ids, "parallel"))(p2)
    assert float(jnp.abs(g2["layers"][0]["route_lam"]).sum()) == 0.0
    assert float(jnp.abs(g2["layers"][0]["lam_w2"]).sum()) > 0


def test_bank_decay_causal_and_bounded():
    """bank_decay read weight exp(acc) must be <= 1 (acc accumulates decay since the bank END, so only tokens AFTER the
    bank read it), and the archive must contribute nothing before the first bank closes. Tested with the M1 setting
    decay_boost=0 (no artificial forgetting)."""
    for bd in (True, False):
        cfg, be = mk(bank_decay=bd)
        p = mk_params(cfg, decay_boost=0.0)
        ids = ids_of(cfg, 2, 64)
        for engine in ("parallel", "scan"):
            a = forward(p, ids, cfg, be, engine)[0]
            z = forward(p, ids, cfg, be, engine, force_gate=0.0)[0]
            assert float(jnp.abs(a - z)[:, :cfg.seg].max()) < 1e-5, (bd, engine, "archive leaks into first segment")
            for cut in (17, 31, 33):
                ids2 = ids.at[:, cut:].set((ids[:, cut:] + 3) % cfg.vocab)
                d = float(jnp.abs(forward(p, ids2, cfg, be, engine)[0] - a)[:, :cut].max())
                assert d < 1e-5, (bd, engine, cut, d)
        _, caches = forward(p, ids, cfg, be, "scan", None, True, 6)
        acc = np.asarray(caches[0]["mix"]["arch"]["acc"])
        assert acc.max() <= 1e-6, acc.max()                 # exp(acc) <= 1


def test_leveled_policy_alias():
    """'leveled' (honest name of the LSM-style compaction) == legacy 'fenwick'."""
    cfg0, be = mk(max_banks=3)
    p, ids = mk_params(cfg0), ids_of(cfg0, 1, 96)
    a = forward(p, ids, cfg0.with_(bank_policy="fenwick"), be, "scan", None, True, 3)[0]
    b = forward(p, ids, cfg0.with_(bank_policy="leveled"), be, "scan", None, True, 3)[0]
    assert rel(a, b) < 1e-7


def test_parallel_eq_scan_auto_levels(): _check_parallel_eq_scan("adaptive_auto_levels")


def test_ccq_equals_query_contraction():
    """read_core(ccq) == float64 numpy read of (I - lam*Sigma_S) q with the STATE-derived Sigma_S = S S^T/||S||^2 (pre-read and
    post-read forms coincide). This is NOT the paper's CCQ (Sigma = running key covariance). lam=0 -> plain read;
    n_levels=0 resolves to floor(log2(slots))+1."""
    from gdn3.routing import read_core
    r = np.random.default_rng(0); D = 8
    S = r.standard_normal((D, D)); q = r.standard_normal((3, D)); lam = 0.4
    sf = (S * S).sum()
    ref = (q - lam * (q @ S @ S.T) / sf) @ S * 0.5
    Sj, SjT = jnp.asarray(S, jnp.float32), jnp.asarray(S.T, jnp.float32)
    mm, mmT = (lambda x: x @ Sj), (lambda x: x @ SjT)
    rown = jnp.sqrt(jnp.asarray((S * S).sum(-1), jnp.float32))
    cfg = G3Config(d_head=D, ccq=True)
    got = read_core(jnp.asarray(q, jnp.float32), mm, mmT, rown, jnp.float32(sf), jnp.float32(lam), 0.5, cfg)
    assert rel(got, ref) < 1e-4, rel(got, ref)
    zero = read_core(jnp.asarray(q, jnp.float32), mm, mmT, rown, jnp.float32(sf), jnp.float32(0.0), 0.5, cfg)
    assert rel(zero, 0.5 * q @ S) < 1e-4
    assert G3Config(n_levels=0, max_banks=8).n_lv == 4 and G3Config(n_levels=6).n_lv == 6


def test_oracle_weights_numpy():
    """tasks.oracle_weights puts the gate on the bank(s) of the asked fact, only on that query's 4 positions."""
    rng = np.random.default_rng(0)
    P, nq, L, seg = 6, 3, 64, 16
    ids, mask, qi = tasks.make_mqar(rng, 4, P, nq, L)
    w = tasks.oracle_weights(qi, P, nq, L, seg)
    assert w.shape == (4, L, L // seg) and np.allclose(w.sum(-1)[w.sum(-1) > 0], 1.0)
    base = L - 4 * nq
    assert w[:, :base].sum() == 0
    for b in range(4):
        for i in range(nq):
            f = int(qi[b, i])
            for t_ in range(4):
                row = w[b, base + 4 * i + t_]
                assert row[(3 * f) // seg] > 0 and row[(3 * f + 2) // seg] > 0


def test_oracle_routing_parallel():
    """oracle=0 -> archive silent (== force_gate=0); oracle one-hot changes the output only at weighted positions >= bank end;
    gradients stay finite; scan engine rejects an oracle."""
    cfg, be = mk()
    p = mk_params(cfg, decay_boost=0.0)
    ids = ids_of(cfg, 2, 64)
    n = 64 // cfg.seg
    z = jnp.zeros((2, 64, n))
    a = forward(p, ids, cfg, be, "parallel", oracle=z)[0]
    b = forward(p, ids, cfg, be, "parallel", force_gate=0.0)[0]
    assert rel(a, b) < 1e-5, rel(a, b)
    w = jnp.zeros((2, 64, n)).at[:, 40:44, 0].set(1.0)
    c = forward(p, ids, cfg, be, "parallel", oracle=w)[0]
    d = np.abs(np.asarray(c) - np.asarray(b))
    assert d[:, 40:44].max() > 1e-5 and d[:, :40].max() < 1e-5
    g = jax.grad(lambda pp: lm_loss_oracle(pp, cfg, be, ids, w))(p)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(g))
    try:
        forward(p, ids, cfg, be, "scan", oracle=w); raise AssertionError("scan engine must reject oracle")
    except ValueError:
        pass


def lm_loss_oracle(params, cfg, be, ids, w):
    return masked_ce(forward(params, ids, cfg, be, "parallel", oracle=w)[0], ids)[0]
