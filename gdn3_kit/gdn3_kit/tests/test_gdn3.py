"""GDN-3 test-suite.  Every test takes `kind` in {"serial", "pallas"}:
   serial : pure-JAX reference backend (CPU is fine, seconds)       -> logic of GDN-3
   pallas : atomic_gdn2 kernels (TPU; on CPU it runs in interpret mode and is slow) -> kernel + integration
Run:  python tests/run.py --backend serial        (anywhere)
      python tests/run.py --backend both          (Kaggle TPU)
      pytest -q tests                             (GDN3_BACKEND=serial|pallas|serial,pallas)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from gdn3.config import G3Config
from gdn3.backend import make_backend, serial_backend, gdn_step
from gdn3.model import init_params, forward, masked_ce
from gdn3.cache import init_cache, decode_step, prefill, step_read, fill_archive
from gdn3 import segments as sg
from gdn3 import routing as rt
from gdn3 import tasks

CPU = jax.default_backend() == "cpu"
TOL = 2e-4 if CPU else 6e-2          # relative, parallel vs streaming (TPU default matmul precision is bf16)


try:
    import pytest
    Skip = pytest.skip.Exception
except Exception:  # plain runner
    class Skip(Exception):
        pass


def geom(kind, warm=False):
    if kind == "serial":
        return dict(D=16, seg=16, warm=4 if warm else 0, L=64)
    if warm:
        return dict(D=64, seg=256, warm=128, L=768)
    return dict(D=64, seg=128, warm=0, L=512)


def mk(kind, mode="hybrid", warm=False, **over):
    g = geom(kind, warm)
    kw = dict(vocab=48, d_model=32, n_heads=2, d_head=g["D"], n_layers=2, mlp_mult=2, mode=mode, seg=g["seg"],
              warm=g["warm"] if mode == "local" else 0, top_k=2, sketch_dim=0, remat=False)
    kw.update(over)
    cfg = G3Config(**kw)
    be = make_backend(kind, cfg.d_head)
    return cfg, be, g


def mk_params(cfg, seed=0):
    p = init_params(jax.random.PRNGKey(seed), cfg)
    if cfg.ha:
        for i, lp in enumerate(p["layers"]):
            k1, k2 = jax.random.split(jax.random.PRNGKey(100 + seed * 10 + i))
            lp["h_init"] = 0.05 * jax.random.normal(k1, lp["h_init"].shape)
            lp["arch_gate"] = jnp.full_like(lp["arch_gate"], 0.7)
            lp["route_tau"] = lp["route_tau"] * (1.0 + 0.2 * jax.random.normal(k2, lp["route_tau"].shape))
    return p


def mk_ids(cfg, B, L, seed=1):
    return jax.random.randint(jax.random.PRNGKey(seed), (B, L), 0, cfg.vocab)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))


def rand_inputs(key, B, H, L, D):
    ks = jax.random.split(key, 6)
    l2 = lambda t: t * jax.lax.rsqrt(jnp.sum(t * t, -1, keepdims=True) + 1e-6)
    q = l2(jax.random.normal(ks[0], (B, H, L, D)))
    k = l2(jax.random.normal(ks[1], (B, H, L, D)))
    v = jax.random.normal(ks[2], (B, H, L, D)) * 0.5
    w = jax.nn.sigmoid(jax.random.normal(ks[3], (B, H, L, D)))
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, L, D)))
    g = -jax.nn.softplus(jax.random.normal(ks[5], (B, H, L, D))) * 0.05
    return q, k, v, w, b, g


def drop_arch(params):
    keep = lambda lp: {k: v for k, v in lp.items() if k not in ("h_init", "route_tau", "route_lam", "route_null", "arch_gate")}
    return dict(params, layers=[keep(lp) for lp in params["layers"]])


# ============================================================================ segmentation / planning
def test_segments_roundtrip(kind):
    B, H, D, n, C = 2, 2, 8, 3, 16
    x = jax.random.normal(jax.random.PRNGKey(0), (B, H, n * C, D))
    for W in (0, 4):
        f = sg.fold_overlap(x, n, C, W)
        assert f.shape == (B * n, H, W + C, D)
        back = sg.unfold_out(f, B, n, W)
        assert float(jnp.abs(back - x).max()) == 0.0, f"W={W}"
        for j in range(n):                                  # segment j starts W tokens early
            lo = j * C - W
            ref = x[:, :, max(lo, 0):(j + 1) * C]
            got = f.reshape(B, n, H, W + C, D)[:, j, :, (W + C) - ref.shape[2]:]
            assert float(jnp.abs(ref - got).max()) == 0.0
        if W:
            assert float(jnp.abs(f.reshape(B, n, H, W + C, D)[:, 0, :, :W]).max()) == 0.0     # zero left pad


def test_plan_segments(kind):
    p = sg.plan_segments(100_000, 512)
    assert p["C"] == 512 and p["n"] == 196 and p["L_padded"] == 196 * 512
    p = sg.plan_segments(100_000, 512, n_max=64)
    assert p["n"] <= 64 and p["C"] % 128 == 0 and p["L_padded"] >= 100_000
    p = sg.plan_segments(10, 100, align=128)
    assert p["C"] == 128 and p["n"] == 1


# ============================================================================ kernel (pallas) vs reference
def test_backend_matches_serial(kind):
    if kind != "pallas":
        raise Skip("kernel-vs-reference test")
    B, H, L, D = 1, 2, 256, 64
    q, k, v, w, b, g = rand_inputs(jax.random.PRNGKey(0), B, H, L, D)
    ref, kern = serial_backend(D).call, make_backend("pallas", D).call
    o1, h1 = ref(q, k, v, w, b, g)
    o2, h2 = kern(q, k, v, w, b, g)
    assert rel(o2, o1) < 3e-3, rel(o2, o1)
    assert rel(h2, h1) < 3e-3, rel(h2, h1)
    ct = jax.random.normal(jax.random.PRNGKey(5), (B, H, L, D))
    h0 = 0.1 * jax.random.normal(jax.random.PRNGKey(6), (B, H, D, D))

    def loss(call):
        return lambda *a: jnp.sum(call(*a)[0] * ct) + jnp.sum(call(*a)[1] ** 2)
    g1 = jax.grad(loss(ref), argnums=range(7))(q, k, v, w, b, g, h0)
    g2 = jax.grad(loss(kern), argnums=range(7))(q, k, v, w, b, g, h0)
    for name, a, c in zip("qkvwbgh", g2, g1):
        e = float(jnp.linalg.norm(a - c) / (jnp.linalg.norm(c) + 1e-12))
        assert e < 3e-2, f"d{name} rel.err {e:.2e}"


def test_chunk_invariance(kind):
    g = geom(kind)
    be = make_backend(kind, g["D"])
    L = g["L"]
    cut = L // 2
    q, k, v, w, b, gg = rand_inputs(jax.random.PRNGKey(0), 1, 2, L, g["D"])
    o, _ = be.call(q, k, v, w, b, gg)
    a = lambda t: t[:, :, :cut]
    c = lambda t: t[:, :, cut:]
    o1, h1 = be.call(*map(a, (q, k, v, w, b, gg)))
    o2, _ = be.call(*map(c, (q, k, v, w, b, gg)), h1)
    assert rel(jnp.concatenate([o1, o2], 2), o) < 1e-3


def test_split_grads(kind):
    g = geom(kind)
    be = make_backend(kind, g["D"])
    L, D = g["L"], g["D"]
    cut = L // 2
    q, k, v, w, b, gg = rand_inputs(jax.random.PRNGKey(1), 1, 2, L, D)
    h0 = jnp.zeros((1, 2, D, D), jnp.float32)
    ct = jax.random.normal(jax.random.PRNGKey(2), (1, 2, L, D))

    def full(*a):
        return jnp.sum(be.call(*a)[0] * ct)

    def split(q, k, v, w, b, gg, h0):
        s = lambda t, lo, hi: t[:, :, lo:hi]
        o1, h1 = be.call(*(s(t, 0, cut) for t in (q, k, v, w, b, gg)), h0)
        o2, _ = be.call(*(s(t, cut, L) for t in (q, k, v, w, b, gg)), h1)
        return jnp.sum(o1 * ct[:, :, :cut]) + jnp.sum(o2 * ct[:, :, cut:])

    gf = jax.grad(full, argnums=range(7))(q, k, v, w, b, gg, h0)
    gs = jax.grad(split, argnums=range(7))(q, k, v, w, b, gg, h0)
    for n_, a, c in zip("qkvwbgh", gf, gs):
        e = float(jnp.linalg.norm(a - c) / (jnp.linalg.norm(a) + 1e-12))
        assert e < 2e-3, f"d{n_}: {e:.2e}"


def test_fold_equals_slices(kind):
    """fold_overlap with warm-up == running each segment slice separately from h_init."""
    g = geom(kind, warm=True)
    D, C, W = g["D"], g["seg"], g["warm"]
    n = 3
    be = make_backend(kind, D)
    q, k, v, w, b, gg = rand_inputs(jax.random.PRNGKey(3), 1, 2, n * C, D)
    h_init = 0.05 * jax.random.normal(jax.random.PRNGKey(4), (2, D, D))
    h0 = jnp.broadcast_to(h_init[None], (n, 2, D, D))
    of, Sf = be.call(*(sg.fold_overlap(t, n, C, W) for t in (q, k, v, w, b, gg)), h0)
    of = sg.unfold_out(of, 1, n, W)
    for j in range(n):
        lo = max(0, j * C - W)
        sl = lambda t: t[:, :, lo:(j + 1) * C]
        pad = W - (j * C - lo)                                  # zero-pad only matters for j=0
        oj, _ = be.call(*(jnp.pad(sl(t), ((0, 0), (0, 0), (pad, 0), (0, 0))) for t in (q, k, v, w, b, gg)), h0[:1])
        assert rel(of[:, :, j * C:(j + 1) * C], oj[:, :, -C:]) < 1e-3, j


# ============================================================================ model-level correctness
def test_causality(kind):
    for mode, warm in (("gdn2", False), ("hybrid", False), ("snap", False), ("local", False), ("local", True)):
        cfg, be, g = mk(kind, mode, warm)
        params = mk_params(cfg)
        L, C = g["L"], cfg.seg
        ids = mk_ids(cfg, 2, L)
        base = forward(params, ids, cfg, be)[0]
        for cut in (1, C - 1, C, C + 3, 2 * C):
            if cut >= L:
                continue
            ids2 = ids.at[:, cut:].set((ids[:, cut:] + 7) % cfg.vocab)
            d = float(jnp.abs(forward(params, ids2, cfg, be)[0] - base)[:, :cut].max())
            assert d < (1e-5 if CPU else 2e-2), f"{mode} warm={warm} cut={cut}: leak {d:.2e}"


def test_hybrid_gate0_equals_gdn2(kind):
    cfg, be, g = mk(kind, "hybrid")
    params = mk_params(cfg)
    for lp in params["layers"]:
        lp["arch_gate"] = jnp.zeros_like(lp["arch_gate"])
    ids = mk_ids(cfg, 2, g["L"])
    a = forward(params, ids, cfg, be)[0]
    cfg2 = cfg.with_(mode="gdn2")
    b = forward(drop_arch(params), ids, cfg2, make_backend(kind, cfg.d_head))[0]
    assert rel(a, b) < 1e-5, rel(a, b)


def test_local_single_segment_equals_gdn2(kind):
    for warm in (False, True):
        cfg, be, g = mk(kind, "local", warm)
        params = mk_params(cfg)
        for lp in params["layers"]:
            lp["h_init"] = jnp.zeros_like(lp["h_init"])
        ids = mk_ids(cfg, 2, cfg.seg)                                  # n = 1 -> no archive reads
        a = forward(params, ids, cfg, be)[0]
        b = forward(drop_arch(params), ids, cfg.with_(mode="gdn2"), make_backend(kind, cfg.d_head))[0]
        assert rel(a, b) < 1e-4, (warm, rel(a, b))


def stream_all(params, ids, cfg, N):
    cache = init_cache(params, cfg, ids.shape[0], N)

    def body(c, tok):
        lg, c2 = decode_step(params, c, tok, cfg)
        return c2, lg
    _, lg = jax.lax.scan(body, cache, ids.T)
    return lg.transpose(1, 0, 2)


def test_decode_equals_parallel(kind):
    cases = [("gdn2", False, {}), ("hybrid", False, {}), ("hybrid", False, dict(sketch_dim=8 if kind == "serial" else 32)),
             ("snap", False, {}), ("local", False, {}), ("local", True, {}), ("local", True, dict(arch_heads=1)),
             ("hybrid", False, dict(arch_heads=1))]
    for mode, warm, over in cases:
        cfg, be, g = mk(kind, mode, warm, **over)
        params = mk_params(cfg)
        L, C = g["L"], cfg.seg
        ids = mk_ids(cfg, 2, L)
        par = forward(params, ids, cfg, be)[0]
        st = stream_all(params, ids, cfg, N=L // C + 1)
        assert rel(st, par) < TOL, f"{mode} warm={warm} {over}: stream vs parallel {rel(st, par):.2e}"
        # prefill: parallel aligned prefix + recurrent tail
        Lx = L - cfg.align(be.bt) // 2 if cfg.align(be.bt) > 1 else L
        Lx = max(Lx, C + 1)
        lg, cache = prefill(params, ids[:, :Lx], cfg, be, max_banks=L // C + 1)
        assert rel(lg, par[:, :Lx]) < TOL, f"{mode} warm={warm} prefill {rel(lg, par[:, :Lx]):.2e}"


def test_dynamic_length(kind):
    cfg, be, g = mk(kind, "hybrid")
    params = mk_params(cfg)
    C, L = cfg.seg, g["L"]
    ids = mk_ids(cfg, 2, L + 7)
    full = forward(params, ids, cfg, be)[0]                           # padded to alignment internally
    assert full.shape == (2, L + 7, cfg.vocab)
    for Lx in (C + 3, 2 * C, L - 5):
        part = forward(params, ids[:, :Lx], cfg, be)[0]
        assert part.shape[1] == Lx
        assert rel(part, full[:, :Lx]) < (1e-5 if CPU else 2e-2), Lx


def test_archive_policies(kind):
    cfg, be, g = mk(kind, "hybrid")
    params = mk_params(cfg)
    C, L = cfg.seg, g["L"]
    n = L // C
    ids = mk_ids(cfg, 1, L + C // 2)
    ref = forward(params, ids, cfg, be)[0]
    lg, cache = prefill(params, ids, cfg, be, max_banks=n)            # enough slots -> identical to parallel
    assert rel(lg, ref) < TOL
    for policy in ("fifo", "merge"):
        c2 = cfg.with_(bank_policy=policy)
        for N in (1, 2, max(n - 1, 1)):
            lg2, cache2 = prefill(params, ids, c2, be, max_banks=N)
            assert lg2.shape == ref.shape and bool(jnp.all(jnp.isfinite(lg2))), (policy, N)
            assert int(cache2["layers"][0]["cnt"]) <= N
    # merge keeps total mass: sum of banks equal to the sum of the uncapped archive
    S_full = prefill(params, ids[:, :L], cfg.with_(bank_policy="merge"), be, max_banks=n)[1]["layers"][0]["S"]
    S_cap = prefill(params, ids[:, :L], cfg.with_(bank_policy="merge"), be, max_banks=2)[1]["layers"][0]["S"]
    assert rel(S_cap.sum(2), S_full.sum(2)) < 1e-4


def test_fill_archive(kind):
    S = jnp.arange(2 * 1 * 5, dtype=jnp.float32).reshape(2, 1, 5, 1, 1)
    Sf, seg, cnt = fill_archive(S, 3, "fifo")
    assert seg.tolist() == [2, 3, 4] and int(cnt) == 3
    Sm, seg, cnt = fill_archive(S, 3, "merge")
    assert seg.tolist() == [2, 3, 4] and float(Sm.sum()) == float(S.sum())
    Sp, seg, cnt = fill_archive(S, 8, "fifo")
    assert seg.tolist() == [0, 1, 2, 3, 4, -1, -1, -1] and int(cnt) == 5


def test_gradients_flow(kind):
    for mode, warm in (("hybrid", False), ("local", True), ("snap", False)):
        cfg, be, g = mk(kind, mode, warm)
        params = mk_params(cfg)
        ids = mk_ids(cfg, 2, g["L"])
        mask = jnp.ones(ids.shape, bool)

        def loss(p):
            return masked_ce(forward(p, ids, cfg, be)[0], ids, mask)[0]
        gr = jax.grad(loss)(params)
        flat = jax.tree_util.tree_leaves(gr)
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in flat), mode
        lp = gr["layers"][0]
        assert float(jnp.abs(lp["arch_gate"]).sum()) > 0, f"{mode}: no grad to arch_gate"
        if mode != "snap":
            assert float(jnp.abs(lp["h_init"]).sum()) > 0, f"{mode}: no grad to h_init"
        assert float(jnp.abs(lp["route_tau"]).sum()) > 0, f"{mode}: router receives no gradient"


def test_sparse_equals_dense_read(kind):
    cfg, be, g = mk(kind, "hybrid", top_k=2)
    params = mk_params(cfg)
    C, L = cfg.seg, g["L"]
    ids = mk_ids(cfg, 2, L)
    _, cache = prefill(params, ids, cfg, be, max_banks=L // C + 2)
    c = cache["layers"][0]
    p = params["layers"][0]
    q = jax.random.normal(jax.random.PRNGKey(9), (2, cfg.ha, cfg.d_head))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    a = step_read(q, c, p, cfg, cache["pos"], sparse=True)
    b = step_read(q, c, p, cfg, cache["pos"], sparse=False)
    assert rel(a, b) < 1e-5


# ============================================================================ router quality (pure math)
def test_router_finds_true_bank(kind):
    rng = np.random.default_rng(0)
    D, n_banks, ppb, m, trials = 64, 8, 8, 32, 120
    R = rt.make_sketch(D, m)
    hits = 0
    for _ in range(trials):
        S, keys = [], []
        for _j in range(n_banks):
            k = rng.normal(size=(ppb, D)).astype(np.float32)
            k /= np.linalg.norm(k, axis=1, keepdims=True)
            v = rng.normal(size=(ppb, D)).astype(np.float32) / math.sqrt(D)
            h = jnp.zeros((D, D))
            for i in range(ppb):
                h, _ = gdn_step(h, jnp.zeros(D), jnp.asarray(k[i]), jnp.asarray(v[i]), jnp.ones(D), jnp.ones(D), jnp.zeros(D), 1.0)
            S.append(h)
            keys.append(k)
        S = jnp.stack(S)
        jt, it = rng.integers(n_banks), rng.integers(ppb)
        q = jnp.asarray(keys[jt][it])
        T = rt.sketch_bank(S, R)
        e = jnp.sum(jnp.einsum("d,jdm->jm", q, T) ** 2, -1)
        lr = rt.log_ratio(e, rt.frob2(S), jnp.sum(q * q), D)
        hits += int(jnp.argmax(lr)) == jt
    assert hits / trials >= 0.9, hits / trials


def test_bank_gate_properties(kind):
    sc = jnp.array([[2.0, 1.0, 0.5, -1.0]])
    valid = jnp.array([True, True, True, False])
    a = rt.bank_gate(sc, valid, jnp.array([[0.0]]), top_k=2)
    assert float(a[0, 2]) == 0.0 and float(a[0, 3]) == 0.0           # not top-2 / invalid
    assert 0.0 < float(a.sum()) < 1.0                                   # null takes mass
    a0 = rt.bank_gate(sc, jnp.zeros(4, bool), jnp.array([[0.0]]), top_k=2)
    assert float(jnp.abs(a0).sum()) == 0.0                              # nothing valid -> read nothing


# ============================================================================ data / training smoke
def test_mqar_data(kind):
    rng = np.random.default_rng(0)
    P, nq, L = 6, 3, 64
    ids, mask, qi = tasks.make_mqar(rng, 4, P, nq, L)
    assert ids.shape == (4, L) and int(mask.sum()) == 4 * nq
    for b in range(4):
        facts = {(ids[b, 3 * i], ids[b, 3 * i + 1]): ids[b, 3 * i + 2] for i in range(P)}
        base = L - 4 * nq
        for i in range(nq):
            assert ids[b, base + 4 * i] == tasks.SEP
            assert facts[(ids[b, base + 4 * i + 1], ids[b, base + 4 * i + 2])] == ids[b, base + 4 * i + 3]
            assert mask[b, base + 4 * i + 3]


def test_train_smoke(kind):
    if kind != "serial":
        raise Skip("smoke training uses the serial backend")
    from gdn3.train import train
    cfg, be, g = mk(kind, "hybrid", seg=16, vocab=256)
    fn = tasks.mqar_batch_fn([3, 4], 2, 64, 2)
    params, hist = train(cfg, be, fn, steps=3, lr=1e-3, warmup=1, log_every=1, verbose=False)
    assert all(np.isfinite(h["loss"]) for h in hist)
