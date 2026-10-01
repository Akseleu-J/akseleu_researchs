"""Capacity curve  acc(P)  and  P* (= max P with acc >= 0.9)  for GDN-2 vs GDN-3 variants, several seeds.

Example (Kaggle TPU):
  python bench/mqar_capacity.py --seeds 3 --steps 4000 --L 4096 \
     --cfg gdn2:mode=gdn2 --cfg hybrid128:mode=hybrid,seg=128 --cfg local128:mode=local,seg=128 \
     --cfg snap128:mode=snap,seg=128
  iso-state control (only sensible for few banks): L=4096, seg=1024 -> n=4 banks -> compare with
     --cfg gdn2_wide:mode=gdn2,d_head=288,n_heads=2   (288^2 ~ 5*128^2); see docs/DESIGN.md section 9
Success criterion (docs/DESIGN.md): P*(GDN-3) >= 2 * P*(GDN-2) at equal training, with seed std << the gap.
"""
import argparse
import os
import numpy as np

from _common import base_args, parse_cfgs, make_be, dump, md_table


def main():
    ap = argparse.ArgumentParser()
    base_args(ap)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--n-q", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--p-train", default="16,32,64,128,256,384,512")
    ap.add_argument("--p-eval", default="32,64,128,256,384,512,768,1024")
    ap.add_argument("--oracle", action="store_true", help="also evaluate with oracle routing (router diagnosis)")
    ap.add_argument("--quick", action="store_true", help="tiny run for plumbing checks")
    a = ap.parse_args()
    if a.quick:
        a.steps, a.seeds, a.L = 200, 1, 2048
        a.p_train, a.p_eval = "8,16,32", "16,32,64"

    from gdn3 import tasks
    from gdn3.train import train
    from gdn3.evaluate import eval_mqar, p_star
    from gdn3.env import device_summary
    print(device_summary())
    P_train = [int(x) for x in a.p_train.split(",")]
    P_eval = [int(x) for x in a.p_eval.split(",")]
    results = {}
    for name, cfg in parse_cfgs(a):
        be = make_be(a, cfg)
        for seed in range(a.seeds):
            print(f"\n=== {name} seed={seed} ===", flush=True)
            fn = tasks.mqar_batch_fn(P_train, a.n_q, a.L, a.bsz)
            params, hist = train(cfg, be, fn, steps=a.steps, lr=a.lr, warmup=a.warmup, seed=seed,
                                 log_every=max(a.steps // 10, 1),
                                 log_path=os.path.join(a.out, f"mqar_{name}_s{seed}.jsonl"))
            curve = eval_mqar(params, cfg, be, P_eval, a.L, n_q=a.n_q, B=a.bsz, n_batches=8, seed=999)
            rec = dict(curve=curve, p_star=p_star(curve), final_loss=hist[-1]["loss"])
            if a.oracle and cfg.mode != "gdn2":
                oc = eval_mqar(params, cfg, be, P_eval, a.L, n_q=a.n_q, B=a.bsz, n_batches=8, seed=999, oracle=True)
                rec["oracle_curve"], rec["oracle_p_star"] = oc, p_star(oc)
            results.setdefault(name, []).append(rec)
            print("  acc(P):", {P: round(v["acc"], 3) for P, v in curve.items()}, " P*=", rec["p_star"], flush=True)
            dump(os.path.join(a.out, "mqar_capacity.json"), results)

    header = ["config"] + [f"P={P}" for P in P_eval] + ["P* mean±std"]
    rows = []
    for name, recs in results.items():
        accs = [[r["curve"][P]["acc"] for r in recs] for P in P_eval]
        ps = [r["p_star"] for r in recs]
        rows.append([name] + [f"{np.mean(x):.3f}" for x in accs] + [f"{np.mean(ps):.0f}±{np.std(ps):.0f}"])
    print("\n" + md_table(header, rows))
    with open(os.path.join(a.out, "mqar_capacity.md"), "w") as f:
        f.write(md_table(header, rows) + "\n")


if __name__ == "__main__":
    main()
