"""M1: MQAR capacity curve and P*.   P*(hybrid) >= 2 * P*(gdn2c) with gap > 3 sigma  (3 seeds).

Layout (tasks.make_mqar): [P facts x 3 tokens][filler][n_q queries x 4]; facts sit at the START of the sequence, queries at
the END, so with L=2048, seg=128 every fact is >= 1 bank old when queried. NO artificial forgetting: parameters come from
init_params directly (decay_boost = 0; tests' mk_params adds +2 which would speed forgetting ~7.4x -- never use it here).

Configs
  gdn2c          mode=gdn2, channel decay            (baseline)
  hybrid         mode=hybrid, seg=128
  hybrid_seg512  mode=hybrid, seg=512
  hybrid_oracle  TRUE oracle router: the gate is placed on the bank that holds the asked fact (tasks.oracle_weights), in
                 training and evaluation. Upper bound for what routing can give.
  hybrid_allk    PROXY for an oracle router: sketch_dim=0 and top_k = #banks (router selection error removed; this is NOT
                 a true oracle -- it still weights banks by the learned energy-ratio score). Diagnostic only.

usage: python bench/mqar_capacity.py --out results/m1 [--backend serial|pallas] [--quick]
"""
import argparse
import os

from _common import dump, m1_verdict, mean_std, pick_backend, Timer


def configs(L, quick):
    from gdn3.config import G3Config
    base = dict(vocab=256, d_model=128 if not quick else 32, n_heads=4 if not quick else 2, d_head=32 if not quick else 16,
                n_layers=2, mlp_mult=2, decay_mode="channel", sketch_dim=-1, top_k=2)
    seg = 128 if not quick else 16
    nb = -(-L // seg)
    return {
        "gdn2c": G3Config(**base, mode="gdn2", seg=seg),
        "hybrid": G3Config(**base, mode="hybrid", seg=seg),
        "hybrid_seg512": G3Config(**base, mode="hybrid", seg=4 * seg),
        "hybrid_oracle": G3Config(**base, mode="hybrid", seg=seg),     # TRUE oracle: gate weights from tasks.oracle_weights (train + eval)
        "hybrid_allk": G3Config(**{**base, "sketch_dim": 0, "top_k": nb}, mode="hybrid", seg=seg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m1")
    ap.add_argument("--backend", default="serial")
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--n_q", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--p_eval", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--only", nargs="*", default=None, help="subset of config names")
    ap.add_argument("--quick", action="store_true", help="tiny CPU smoke run (NOT a result)")
    a = ap.parse_args()
    if a.quick:
        a.L, a.n_q, a.steps, a.batch, a.seeds, a.p_eval = 128, 4, 30, 4, [0], [2, 4]
    from gdn3 import tasks
    from gdn3.train import train
    from gdn3.evaluate import eval_mqar, p_star
    cfgs = configs(a.L, a.quick)
    names = a.only or list(cfgs)
    res = {"args": vars(a), "runs": {}}
    train_P = [p for p in a.p_eval] if a.quick else sorted(set(a.p_eval + [2]))
    for name in names:
        cfg = cfgs[name]
        be = pick_backend(a.backend, cfg.d_head)
        res["runs"][name] = []
        for seed in a.seeds:
            oseg = cfg.seg if name == "hybrid_oracle" else None
            fn = tasks.mqar_batch_fn(train_P, a.n_q, a.L, a.batch, oracle_seg=oseg)
            with Timer() as tm:
                params, hist = train(cfg, be, fn, steps=a.steps, lr=a.lr, warmup=max(a.steps // 15, 1), seed=seed,
                                     log_every=max(a.steps // 10, 1), verbose=True)
            curve = eval_mqar(params, cfg, be, a.p_eval, a.L, n_q=a.n_q, B=8, n_batches=4, seed=1000 + seed, oracle_seg=oseg)
            ps = p_star(curve, a.thresh)
            print(f"[{name} seed={seed}] P*={ps} curve={ {P: round(v['acc'], 3) for P, v in curve.items()} }", flush=True)
            res["runs"][name].append(dict(seed=seed, pstar=ps, curve=curve, train_s=tm.dt, final=hist[-1] if hist else None))
            dump(os.path.join(a.out, "m1_results.json"), res)          # incremental: a crash loses one run, not all
    if "gdn2c" in res["runs"]:
        res["verdicts"] = {}
        base = [r["pstar"] for r in res["runs"]["gdn2c"]]
        for name in res["runs"]:
            if name != "gdn2c":
                res["verdicts"][name] = m1_verdict([r["pstar"] for r in res["runs"][name]], base)
        print("M1 verdicts:", res["verdicts"])
        o, l = res["runs"].get("hybrid_oracle") or res["runs"].get("hybrid_allk"), res["runs"].get("hybrid")
        if o and l:
            mo, ml = mean_std([r["pstar"] for r in o])[0], mean_std([r["pstar"] for r in l])[0]
            res["diagnosis"] = ("router-limited (oracle >> learned): try sketch_dim=0 / larger top_k" if mo > 1.5 * ml
                                else "bank-capacity-limited or passed: try smaller seg / more arch_heads")
            print("diagnosis:", res["diagnosis"])
    dump(os.path.join(a.out, "m1_results.json"), res)


if __name__ == "__main__":
    main()
