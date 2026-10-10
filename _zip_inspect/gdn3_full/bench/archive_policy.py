"""M3: degradation when max_banks < #banks.  One hybrid model is trained once on MQAR with an unbounded archive; the budget
policy only changes how the archive is compacted, so the SAME params are evaluated under fifo / merge / leveled for each
max_banks (eval_mqar_stream: prefill capped at max_banks, then teacher-forced queries)."""
import argparse
import os

from _common import dump, pick_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m3")
    ap.add_argument("--backend", default="serial")
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--n_q", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--banks", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.L, a.P, a.n_q, a.steps, a.banks = 128, 3, 2, 20, [2, 3]
    from gdn3.config import G3Config
    from gdn3 import tasks
    from gdn3.train import train
    from gdn3.evaluate import eval_mqar_stream
    small = a.quick
    seg = 16 if small else 128
    cfg = G3Config(vocab=256, d_model=32 if small else 128, n_heads=2 if small else 4, d_head=16 if small else 32,
                   n_layers=2, mode="hybrid", seg=seg, decay_mode="channel")
    be = pick_backend(a.backend, cfg.d_head)
    params, _ = train(cfg, be, tasks.mqar_batch_fn([max(a.P // 2, 2), a.P], a.n_q, a.L, 4 if small else 16), steps=a.steps,
                      lr=2e-3, warmup=max(a.steps // 15, 1), log_every=max(a.steps // 5, 1))
    out = {"args": vars(a), "n_banks_full": -(-a.L // seg), "acc": {}}
    for pol in ("fifo", "merge", "leveled"):
        out["acc"][pol] = {}
        for nb in a.banks:
            acc = eval_mqar_stream(params, cfg.with_(bank_policy=pol), be, a.P, a.L, nb, n_q=a.n_q, B=4, n_batches=2)
            out["acc"][pol][nb] = acc
            print(f"{pol:8s} max_banks={nb:3d} acc={acc:.3f}", flush=True)
            dump(os.path.join(a.out, "m3_results.json"), out)


if __name__ == "__main__":
    main()
