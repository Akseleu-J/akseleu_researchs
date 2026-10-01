"""Byte-level LM on enwik8 (or any byte file) + loss-by-position curves, including lengths BEYOND the training L.

If the multi-matrix memory works, loss(position >> seg) of GDN-3 must lie below GDN-2 at equal quality on early
positions ("almost no loss"): the early-position bins check that nothing was lost, the late bins check the gain.

  python bench/lm_position.py --cfg gdn2:mode=gdn2 --cfg hybrid:mode=hybrid,seg=512 --steps 6000 --L 8192 --eval-L 32768
  python bench/lm_position.py --data synthetic --quick          # offline plumbing check
"""
import argparse
import os

import numpy as np

from _common import base_args, parse_cfgs, make_be, dump, md_table


def main():
    ap = argparse.ArgumentParser()
    base_args(ap)
    ap.add_argument("--data", default="enwik8")
    ap.add_argument("--L", type=int, default=8192)
    ap.add_argument("--eval-L", type=int, default=32768)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.steps, a.L, a.eval_L, a.bsz = 100, 2048, 4096, 2

    from gdn3 import tasks
    from gdn3.train import train, make_eval_loss
    from gdn3.evaluate import loss_by_position
    from gdn3.env import device_summary
    import jax.numpy as jnp
    print(device_summary())
    data, val = tasks.get_bytes(a.data, "train"), tasks.get_bytes(a.data, "val")
    edges = [0] + [e for e in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768) if e < a.eval_L] + [a.eval_L - 1]
    results = {}
    for name, cfg in parse_cfgs(a):
        be = make_be(a, cfg)
        for seed in range(a.seeds):
            print(f"\n=== {name} seed={seed} ===", flush=True)
            bf = tasks.lm_batch_fn(data, a.L, a.bsz)
            ev = make_eval_loss(cfg, be)

            def eval_fn(params, step):
                rng = np.random.default_rng(0)
                ls = []
                for _ in range(4):
                    b = tasks.lm_batch_fn(val, a.L, a.bsz)(rng, 0)
                    l, _ = ev(params, jnp.asarray(b["ids"]), jnp.asarray(b["mask"]))
                    ls.append(float(l))
                return dict(val_bpb=float(np.mean(ls) / np.log(2)))

            params, hist = train(cfg, be, bf, steps=a.steps, lr=a.lr, warmup=a.warmup, seed=seed,
                                 log_every=max(a.steps // 20, 1), eval_every=max(a.steps // 5, 1), eval_fn=eval_fn,
                                 log_path=os.path.join(a.out, f"lm_{name}_s{seed}.jsonl"))
            curve = loss_by_position(params, cfg, be, val, a.eval_L, edges, B=1, n_batches=6)
            results.setdefault(name, []).append(dict(edges=edges, nll=curve, final=hist[-1]))
            print("  loss by position:", [round(x, 3) for x in curve], flush=True)
            dump(os.path.join(a.out, "lm_position.json"), results)

    header = ["config"] + [f"[{edges[i]},{edges[i + 1]})" for i in range(len(edges) - 1)]
    rows = [[n] + [f"{np.mean([r['nll'][i] for r in rs]) / np.log(2):.3f}" for i in range(len(edges) - 1)]
            for n, rs in results.items()]
    print("\nbits/byte by position\n" + md_table(header, rows))
    with open(os.path.join(a.out, "lm_position.md"), "w") as f:
        f.write(md_table(header, rows) + "\n")


if __name__ == "__main__":
    main()
