"""Dynamic archive under a memory budget (streaming inference).

Trains (or loads) a GDN-3 model on MQAR, then answers the same queries through prefill()+decode steps with
max_banks = N for N in a sweep and policy in {fifo, merge}.  Reports accuracy vs N (and vs state memory), i.e. how
gracefully quality degrades when the number of matrices is capped.   N = n_full is the uncapped reference.

  python bench/archive_policy.py --mode hybrid --seg 128 --L 4096 --P 128 --steps 3000
  python bench/archive_policy.py --ckpt runs/mqar_hybrid_s128_seed0.ckpt --mode hybrid --seg 128 --L 4096 --P 128
"""
import argparse
import os

from _common import base_args, parse_cfgs, make_be, dump, md_table


def main():
    ap = argparse.ArgumentParser()
    base_args(ap)
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--P", type=int, default=128)
    ap.add_argument("--n-q", type=int, default=16)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--p-train", default="16,32,64,128,256")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not a.cfg:
        a.cfg = ["hybrid:mode=hybrid,seg=128"]
    if a.quick:
        a.steps, a.L, a.P = 150, 2048, 32
        a.p_train = "8,16,32"

    import jax
    import jax.numpy as jnp
    from gdn3 import tasks
    from gdn3.train import train, load_ckpt
    from gdn3.evaluate import eval_mqar_stream
    from gdn3.env import device_summary
    print(device_summary())
    name, cfg = parse_cfgs(a)[0]
    assert cfg.mode != "gdn2", "archive policy needs an archive (mode != gdn2)"
    be = make_be(a, cfg)
    if a.ckpt:
        params = jax.tree_util.tree_map(jnp.asarray, load_ckpt(a.ckpt)["params"])
    else:
        fn = tasks.mqar_batch_fn([int(x) for x in a.p_train.split(",")], a.n_q, a.L, a.bsz)
        params, _ = train(cfg, be, fn, steps=a.steps, lr=a.lr, warmup=min(200, a.steps // 5),
                          log_every=max(a.steps // 10, 1))
    n_full = -(-a.L // cfg.seg)
    sweep = sorted({n_full, max(n_full // 2, 1), max(n_full // 4, 1), max(n_full // 8, 1), 2, 1})
    rows, rec = [], []
    D, H, layers = cfg.d_head, cfg.ha, cfg.n_layers
    for policy in ("fifo", "merge"):
        c2 = cfg.with_(bank_policy=policy)
        for N in sweep:
            acc = eval_mqar_stream(params, c2, be, a.P, a.L, N, n_q=a.n_q, B=4, n_batches=2)
            mb = layers * H * D * D * N * 4 / 2**20
            rows.append([policy, N, f"{mb:.1f} MB", f"{acc:.3f}"])
            rec.append(dict(policy=policy, N=N, mem_mb=mb, acc=acc))
            print("  ", rows[-1], flush=True)
    print("\n" + md_table(["policy", "max_banks N", "archive memory", "acc"], rows))
    dump(os.path.join(a.out, "archive_policy.json"), rec)


if __name__ == "__main__":
    main()
