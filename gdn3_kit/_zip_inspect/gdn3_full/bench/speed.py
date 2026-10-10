"""Honest timing: compile time, steady-state forward us/token (parallel & scan engines), and decode us/token vs #banks N.
Everything is block_until_ready()'d; compile is measured on the first call, steady state on the median of --reps."""
import argparse
import os
import statistics
import time

from _common import dump, pick_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/speed")
    ap.add_argument("--backend", default="serial")
    ap.add_argument("--L", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--banks", type=int, nargs="+", default=[2, 8, 32, 128])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.L, a.banks, a.reps = [64], [2, 4], 2
    import jax
    import jax.numpy as jnp
    from gdn3.config import G3Config
    from gdn3.model import init_params, forward, init_caches, decode_step
    small = a.quick
    cfg = G3Config(vocab=256, d_model=32 if small else 256, n_heads=2 if small else 4, d_head=16 if small else 64,
                   n_layers=2 if small else 4, mode="hybrid", seg=16 if small else 128)
    be = pick_backend(a.backend, cfg.d_head)
    params = init_params(jax.random.PRNGKey(0), cfg)
    out = {"args": vars(a), "device": str(jax.devices()[0]), "forward": {}, "decode": {}}

    def timed(fn, *args):
        t = time.perf_counter(); jax.block_until_ready(fn(*args)); first = time.perf_counter() - t
        ts = []
        for _ in range(a.reps):
            t = time.perf_counter(); jax.block_until_ready(fn(*args)); ts.append(time.perf_counter() - t)
        return first, statistics.median(ts)

    for eng in ("parallel", "scan"):
        for L in a.L:
            ids = jnp.zeros((1, L), jnp.int32)
            f = jax.jit(lambda p, x, e=eng: forward(p, x, cfg, be, e)[0])
            first, med = timed(f, params, ids)
            out["forward"][f"{eng}/L={L}"] = dict(compile_plus_first_s=first, steady_s=med, us_per_token=med / L * 1e6)
            print(eng, L, f"compile+first={first:.1f}s steady={med * 1e3:.1f}ms {med / L * 1e6:.1f} us/tok", flush=True)
            dump(os.path.join(a.out, "speed.json"), out)
    tok = jnp.zeros((1,), jnp.int32)
    for N in a.banks:
        caches = init_caches(params, cfg, 1, N)
        step = jax.jit(lambda c, t: decode_step(params, c, t, cfg))
        first, med = timed(lambda c, t: step(c, t)[0], caches, tok)
        out["decode"][N] = dict(compile_plus_first_s=first, steady_s=med, us_per_token=med * 1e6)
        print(f"decode N={N:4d} {med * 1e6:.1f} us/tok", flush=True)
        dump(os.path.join(a.out, "speed.json"), out)


if __name__ == "__main__":
    main()
