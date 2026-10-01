"""GDN-3 test-suite (JAX). Usage:
    python -m tests.run_all                 # ref backend (any device; CPU is fine)
    python -m tests.run_all --atomic        # additionally tests the atomic_gdn2 Pallas backend (TPU or interpret)
Each test prints PASS/FAIL; exit code != 0 on failure."""
import sys, argparse, traceback
import numpy as np, jax, jax.numpy as jnp
from gdn3 import GDN3Config, gdn2_like, init_model, forward, count_params, StreamingLM, make_backend
from gdn3.bank import merge_pair

SMALL = dict(vocab=64, d_model=64, n_heads=2, d_head=16, n_layers=3, mlp_mult=2, seg=16, bank_every=1, gate_init=0.0)


def mk(**kw):
    cfg = GDN3Config(**{**SMALL, **kw}); be = make_backend("ref", cfg.d_head)
    return cfg, be, init_model(jax.random.PRNGKey(0), cfg)


def set_gates(params, val=0.8):
    for p in params["layers"]:
        if "gate_raw" in p: p["gate_raw"] = jnp.full_like(p["gate_raw"], val)
    return params


def ids_of(B, L, V=64, seed=0):
    return jnp.asarray(np.random.default_rng(seed).integers(0, V, (B, L)), jnp.int32)


def t_gate0_equals_gdn2():
    """banks with gate==0 must equal the GDN-2 baseline (same params) bit-for-bit (up to float)."""
    cfg, be, params = mk(); ids = ids_of(2, 64)
    a = forward(params, cfg, be, ids, force_gate=0.0)
    b = forward(params, gdn2_like(cfg), be, ids)
    err = float(jnp.abs(a - b).max()); assert err < 1e-5, err


def t_causal(cut):
    """changing tokens >= cut must not change logits < cut (nonzero gates so the archive path is active)."""
    cfg, be, params = mk(); params = set_gates(params); ids = ids_of(2, 96)
    ids2 = ids.at[:, cut:].set((ids[:, cut:] + 7) % 64)
    d = float(jnp.abs(forward(params, cfg, be, ids) - forward(params, cfg, be, ids2))[:, :cut].max())
    assert d < 1e-4, f"CAUSALITY VIOLATED max|d|={d:.3e}"


def t_archive_active():
    """with nonzero gate the archive must change outputs of tokens beyond the first segment (and not before)."""
    cfg, be, params = mk(); ids = ids_of(1, 64)
    a = forward(set_gates(params, 0.0), cfg, be, ids); b = forward(set_gates(params, 0.9), cfg, be, ids)
    C = cfg.seg
    assert float(jnp.abs(a - b)[:, :C].max()) < 1e-5 and float(jnp.abs(a - b)[:, C:].max()) > 1e-6


def t_dynamic_n():
    """parameter tree is independent of sequence length; forward works for any number of segments incl. partial last."""
    cfg, be, params = mk(); n0 = count_params(params)
    for L in (8, 16, 40, 64, 100):
        out = forward(params, cfg, be, ids_of(1, L)); assert out.shape == (1, L, cfg.vocab) and bool(jnp.isfinite(out).all())
    assert count_params(init_model(jax.random.PRNGKey(1), cfg)) == n0


def t_streaming_equals_batch():
    cfg, be, params = mk(); params = set_gates(params, 0.7); ids = ids_of(1, 80)
    ref = forward(params, cfg, be, ids)
    for blocks in ([80], [16] * 5, [7, 9, 24, 1, 39], [1] * 20 + [60]):
        s = StreamingLM(params, cfg, be, B=1); outs = []; i = 0
        for m in blocks:
            outs.append(s.feed(ids[:, i:i + m])); i += m
        err = float(jnp.abs(jnp.concatenate(outs, 1) - ref).max()); assert err < 1e-4, (blocks, err)
    assert s.n_banks() == [5] * cfg.n_layers      # 80 tokens / seg 16 -> 5 banks per bank layer


def t_budget_bounds_memory():
    cfg, be, params = mk(); params = set_gates(params); ids = ids_of(1, 160)
    s = StreamingLM(params, cfg, be, B=1, budget=3, policy="oldest"); out = s.feed(ids)
    assert max(s.n_banks()) <= 3 and bool(jnp.isfinite(out).all())
    s = StreamingLM(params, cfg, be, B=1, policy="fenwick", per_level=2); s.feed(ids)
    assert max(s.n_banks()) <= 2 * 5, s.n_banks()


def t_merge_exact_no_erase():
    from gdn3.backend import make_ref_backend
    rng = np.random.default_rng(0); B, H, D, C = 1, 1, 8, 10; be = make_ref_backend(1.0)
    k = jnp.asarray(rng.normal(size=(B, H, 2 * C, D)), jnp.float32); k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jnp.asarray(rng.normal(size=(B, H, 2 * C, D)), jnp.float32); w = jnp.ones_like(v); b = jnp.zeros_like(v)
    g = -0.05 * jnp.asarray(rng.random((B, H, 2 * C, D)), jnp.float32); z = jnp.zeros((B, H, D, D)); q0 = jnp.zeros_like(k)
    _, Sa = be(q0[:, :, :C], k[:, :, :C], v[:, :, :C], w[:, :, :C], b[:, :, :C], g[:, :, :C], z)
    _, Sb = be(q0[:, :, C:], k[:, :, C:], v[:, :, C:], w[:, :, C:], b[:, :, C:], g[:, :, C:], z)
    _, Su = be(q0, k, v, w, b, g, z)
    Sm, _ = merge_pair(Sa, g[:, :, :C].sum(2), Sb, g[:, :, C:].sum(2))
    err = float(jnp.abs(Sm - Su).max()); assert err < 1e-5, err


def t_grads_finite_and_reach_router():
    cfg, be, params = mk(); params = set_gates(params, 0.5); ids = ids_of(2, 64)
    from gdn3 import lm_loss
    loss, gr = jax.value_and_grad(lambda p: lm_loss(p, cfg, be, ids))(params)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(gr))
    l0 = gr["layers"][0]
    assert float(jnp.abs(l0["log_tau"]).sum()) > 0 and float(jnp.abs(l0["theta"]).sum()) > 0 and float(jnp.abs(l0["gate_raw"]).sum()) > 0


def t_landmarks_causal():
    cfg, be, params = mk(n_landmarks=4, seg=16); params = set_gates(params, 0.8); ids = ids_of(1, 96)
    ids2 = ids.at[:, 50:].set((ids[:, 50:] + 5) % 64)
    d = float(jnp.abs(forward(params, cfg, be, ids) - forward(params, cfg, be, ids2))[:, :50].max())
    assert d < 1e-4, d


def t_atomic_vs_ref():
    """atomic Pallas backend == token-serial reference (forward and gradients) incl. h0 chaining."""
    D = 64; ref = make_backend("ref", D); atom = make_backend("atomic", D, g_max=5.0)
    r = np.random.default_rng(0); B, H, L = 1, 2, 256
    l2 = lambda t: t / np.linalg.norm(t, axis=-1, keepdims=True)
    q = jnp.asarray(l2(r.normal(size=(B, H, L, D))), jnp.float32); k = jnp.asarray(l2(r.normal(size=(B, H, L, D))), jnp.float32)
    v = jnp.asarray(r.normal(size=(B, H, L, D)) * 0.5, jnp.float32)
    w = jnp.asarray(1 / (1 + np.exp(-r.normal(size=(B, H, L, D)))), jnp.float32); b = jnp.asarray(1 / (1 + np.exp(-r.normal(size=(B, H, L, D)))), jnp.float32)
    g = jnp.asarray(-0.05 * np.abs(r.normal(size=(B, H, L, D))), jnp.float32)
    h0 = jnp.asarray(r.normal(size=(B, H, D, D)) * 0.01, jnp.float32)
    o1, h1 = ref(q, k, v, w, b, g, h0); o2, h2 = atom(q, k, v, w, b, g, h0)
    rel = lambda a, c: float(jnp.linalg.norm(a - c) / (jnp.linalg.norm(a) + 1e-12))
    assert rel(o1, o2) < 2e-3 and rel(h1, h2) < 2e-3, (rel(o1, o2), rel(h1, h2))
    ct = jnp.asarray(r.normal(size=o1.shape), jnp.float32)
    f = lambda be_: (lambda *a: jnp.sum(be_(*a)[0] * ct) + jnp.sum(be_(*a)[1]))
    g1 = jax.grad(f(ref), argnums=range(7))(q, k, v, w, b, g, h0); g2 = jax.grad(f(atom), argnums=range(7))(q, k, v, w, b, g, h0)
    for n, a, c in zip("qkvwbgh", g1, g2):
        e = rel(a, c); print(f"      d{n} rel.err={e:.2e}"); assert e < 5e-2, (n, e)
    # chunk invariance of the atomic kernel (split == full)
    oa, ha = atom(q[:, :, :128], k[:, :, :128], v[:, :, :128], w[:, :, :128], b[:, :, :128], g[:, :, :128], h0)
    ob, _ = atom(q[:, :, 128:], k[:, :, 128:], v[:, :, 128:], w[:, :, 128:], b[:, :, 128:], g[:, :, 128:], ha)
    assert float(jnp.abs(jnp.concatenate([oa, ob], 2) - o2).max()) < 1e-3


def t_atomic_model_matches_ref():
    cfg = GDN3Config(**{**SMALL, "d_head": 64, "d_model": 128, "seg": 128, "n_layers": 2, "bank_every": 1}); params = set_gates(init_model(jax.random.PRNGKey(0), cfg), 0.5)
    ids = ids_of(1, 384); a = forward(params, cfg, make_backend("ref", 64), ids)
    b = forward(params, cfg, make_backend("atomic", 64, g_max=cfg.g_max), ids)
    err = float(jnp.abs(a - b).max()); assert err < 5e-2, err


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--atomic", action="store_true"); a = ap.parse_args()
    tests = [("gate0 == GDN-2", t_gate0_equals_gdn2), ("causal @ mid-segment (cut=50)", lambda: t_causal(50)),
             ("causal @ segment boundary (cut=48)", lambda: t_causal(48)), ("archive active only beyond 1st segment", t_archive_active),
             ("dynamic N / partial last segment", t_dynamic_n), ("streaming == batch (4 block patterns)", t_streaming_equals_batch),
             ("budget/fenwick bound #banks", t_budget_bounds_memory), ("merge exact when beta=0", t_merge_exact_no_erase),
             ("grads finite, reach router", t_grads_finite_and_reach_router), ("landmarks causal", t_landmarks_causal)]
    if a.atomic:
        tests += [("atomic == ref (fwd, grads, h0, split)", t_atomic_vs_ref), ("model: atomic == ref", t_atomic_model_matches_ref)]
    bad = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception:
            bad += 1; print(f"FAIL  {name}"); traceback.print_exc()
    print(f"\n{len(tests)-bad}/{len(tests)} passed"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
