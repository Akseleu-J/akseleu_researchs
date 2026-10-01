"""Honest jit benchmark (compile time reported separately, warm-up, block_until_ready, median of N):
   * training step  (forward + backward)  us/token for each config and length L
   * decode step latency vs number of archive banks (streaming path)
Replaces the eager-trace timing of the first report (which gave ms/token instead of us/token)."""
import argparse
import os
import statistics
import time

import numpy as np

from _common import base_args, parse_cfgs, make_be, dump, md_table


def timeit(fn, *args, iters=7):
    t0 = time.perf_counter()
    out = fn(*args)
    import jax
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return compile_s, statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    base_args(ap)
    ap.add_argument("--L", default="2048,4096,8192,16384")
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--decode-banks", default="1,8,32,128")
    ap.add_argument("--skip-decode", action="store_true")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from gdn3.env import device_summary
    from gdn3.model import init_params, forward, masked_ce, count_params
    from gdn3.cache import init_cache, decode_step
    print(device_summary())

    rows, rec = [], []
    for name, cfg in parse_cfgs(a):
        be = make_be(a, cfg)
        params = init_params(jax.random.PRNGKey(0), cfg)
        for L in [int(x) for x in a.L.split(",")]:
            if L % cfg.align(be.bt):
                continue
            ids = jax.random.randint(jax.random.PRNGKey(1), (a.bsz, L), 0, 256)
            mask = jnp.ones(ids.shape, bool)

            def loss(p, ids, mask):
                return masked_ce(forward(p, ids, cfg, be)[0], ids, mask)[0]

            fwd_bwd = jax.jit(jax.value_and_grad(loss))
            try:
                c, t = timeit(fwd_bwd, params, ids, mask, iters=a.iters)
            except Exception as e:  # OOM etc.
                print(f"  {name} L={L}: FAILED {type(e).__name__}: {str(e)[:120]}")
                continue
            us = t / (a.bsz * L) * 1e6
            rows.append([name, L, f"{count_params(params) / 1e6:.2f}M", f"{c:.1f}", f"{t * 1e3:.1f}", f"{us:.2f}"])
            rec.append(dict(name=name, L=L, compile_s=c, step_ms=t * 1e3, us_per_token=us))
            print("  ", rows[-1], flush=True)
    print("\n" + md_table(["config", "L", "params", "compile s", "fwd+bwd ms", "us/token"], rows))

    drows = []
    if not a.skip_decode:
        for name, cfg in parse_cfgs(a):
            if cfg.mode == "gdn2":
                continue
            params = init_params(jax.random.PRNGKey(0), cfg)
            for N in [int(x) for x in a.decode_banks.split(",")]:
                cache = init_cache(params, cfg, 1, N)
                step = jax.jit(lambda p, c, t: decode_step(p, c, t, cfg))
                tok = jnp.zeros((1,), jnp.int32)
                # fill the archive by faking count (all slots valid) to measure the full-archive cost
                for c_ in cache["layers"]:
                    c_["seg"] = jnp.arange(N, dtype=jnp.int32)
                    c_["S"] = 0.01 * jax.random.normal(jax.random.PRNGKey(2), c_["S"].shape)
                cache["pos"] = jnp.int32(N * cfg.seg + 1)
                comp, t = timeit(lambda p, c, t_: step(p, c, t_)[0], params, cache, tok, iters=20)
                drows.append([name, N, f"{t * 1e3:.3f}"])
                print("  decode", drows[-1], flush=True)
        print("\n" + md_table(["config", "banks N", "decode ms/token"], drows))
    dump(os.path.join(a.out, "speed.json"), dict(train=rec, decode=drows))


if __name__ == "__main__":
    main()
