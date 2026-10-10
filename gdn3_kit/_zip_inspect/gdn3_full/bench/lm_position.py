"""M2: language-model loss by position (nats/byte) for gdn2c vs hybrid; enwik8 (downloads) or --data synthetic|<file>.
Train at --L, evaluate at --eval_L (> L tests length extrapolation).  The parallel engine unrolls a python loop over banks,
so eval_L/seg = #banks drives compile time: start with eval_L=8192 before trying 32768."""
import argparse
import os

from _common import dump, pick_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m2")
    ap.add_argument("--backend", default="serial")
    ap.add_argument("--data", default="enwik8")
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--eval_L", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.data, a.L, a.eval_L, a.steps, a.batch = "synthetic", 128, 256, 20, 2
    from gdn3.config import G3Config
    from gdn3 import tasks
    from gdn3.train import train
    from gdn3.evaluate import loss_by_position
    base = dict(vocab=256, d_model=256 if not a.quick else 32, n_heads=4 if not a.quick else 2,
                d_head=64 if not a.quick else 16, n_layers=4 if not a.quick else 2, decay_mode="channel",
                seg=128 if not a.quick else 16)
    cfgs = {"gdn2c": G3Config(**base, mode="gdn2"), "hybrid": G3Config(**base, mode="hybrid")}
    tr, va = tasks.get_bytes(a.data, "train"), tasks.get_bytes(a.data, "val")
    edges = sorted({0, *[e for e in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768) if e < a.eval_L], a.eval_L - 1})
    out = {"args": vars(a), "edges": edges, "loss": {}}
    for name, cfg in cfgs.items():
        be = pick_backend(a.backend, cfg.d_head)
        params, hist = train(cfg, be, tasks.lm_batch_fn(tr, a.L, a.batch), steps=a.steps, lr=1e-3, seed=a.seed,
                             warmup=max(a.steps // 20, 1), log_every=max(a.steps // 10, 1))
        out["loss"][name] = loss_by_position(params, cfg, be, va, a.eval_L, edges, B=2, n_batches=4 if not a.quick else 1)
        print(name, [round(x, 3) for x in out["loss"][name]], flush=True)
        dump(os.path.join(a.out, "m2_results.json"), out)


if __name__ == "__main__":
    main()
