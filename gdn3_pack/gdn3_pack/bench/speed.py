"""Honest jit benchmark: train step (fwd+bwd) time, warm-up excluded, block_until_ready, median of N.
   python -m bench.speed --backend atomic --L 4096 8192 --seg 128 256 512"""
import argparse, time, statistics
import numpy as np, jax, jax.numpy as jnp
from gdn3 import GDN3Config, gdn2_like, init_model, lm_loss, make_backend, count_params


def bench(cfg, be, B, L, n=5):
    params = init_model(jax.random.PRNGKey(0), cfg); ids = jnp.asarray(np.random.default_rng(0).integers(0, cfg.vocab, (B, L)), jnp.int32)
    f = jax.jit(jax.value_and_grad(lambda p, x: lm_loss(p, cfg, be, x)))
    jax.block_until_ready(f(params, ids))                                  # compile + warm-up
    ts = []
    for _ in range(n):
        t = time.perf_counter(); jax.block_until_ready(f(params, ids)); ts.append(time.perf_counter() - t)
    return statistics.median(ts), count_params(params)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="atomic"); ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--L", type=int, nargs="+", default=[4096]); ap.add_argument("--seg", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--bank_every", type=int, nargs="+", default=[2, 1]); ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=4); ap.add_argument("--d_head", type=int, default=128); ap.add_argument("--n_layers", type=int, default=4)
    a = ap.parse_args()
    base = GDN3Config(d_model=a.d_model, n_heads=a.n_heads, d_head=a.d_head, n_layers=a.n_layers)
    be = make_backend(a.backend, a.d_head, **({"g_max": base.g_max} if a.backend == "atomic" else {}))
    print(f"{'config':<28}{'L':>7}{'ms/step':>10}{'us/tok':>9}{'x GDN-2':>9}{'params(M)':>10}")
    for L in a.L:
        t0, p0 = bench(gdn2_like(base), be, a.B, L); print(f"{'gdn2':<28}{L:>7}{t0*1e3:>10.1f}{t0/(a.B*L)*1e6:>9.2f}{1.0:>9.2f}{p0/1e6:>10.2f}")
        for be_ in a.bank_every:
            for C in a.seg:
                if L % C: continue
                cfg = base.with_(seg=C, bank_every=be_); t, p = bench(cfg, be, a.B, L)
                print(f"{f'gdn3 C={C} bank_every={be_}':<28}{L:>7}{t*1e3:>10.1f}{t/(a.B*L)*1e6:>9.2f}{t/t0:>9.2f}{p/1e6:>10.2f}")
