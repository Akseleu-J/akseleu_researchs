"""CLI:  python -m gdn3.cli train --task mqar --mode hybrid --seg 128 ...   |   python -m gdn3.cli info"""
from __future__ import annotations
import argparse
import json
import os
import sys

from .config import G3Config, parse_overrides


def add_model_args(ap: argparse.ArgumentParser):
    g = ap.add_argument_group("model / memory")
    g.add_argument("--backend", default="pallas", choices=["pallas", "serial"])
    g.add_argument("--d-model", type=int, default=256)
    g.add_argument("--n-heads", type=int, default=2)
    g.add_argument("--d-head", type=int, default=128)
    g.add_argument("--n-layers", type=int, default=2)
    g.add_argument("--mlp-mult", type=int, default=2)
    g.add_argument("--mode", default="hybrid", choices=["gdn2", "local", "hybrid", "snap"])
    g.add_argument("--seg", type=int, default=128)
    g.add_argument("--warm", type=int, default=0)
    g.add_argument("--arch-heads", type=int, default=0)
    g.add_argument("--top-k", type=int, default=2)
    g.add_argument("--sketch-dim", type=int, default=-1)
    g.add_argument("--extra", default="", help="more G3Config overrides: 'k=v,k=v'")


def cfg_from_args(a, **over) -> G3Config:
    kw = dict(vocab=256, d_model=a.d_model, n_heads=a.n_heads, d_head=a.d_head, n_layers=a.n_layers,
              mlp_mult=a.mlp_mult, mode=a.mode, seg=a.seg, warm=a.warm, arch_heads=a.arch_heads,
              top_k=a.top_k, sketch_dim=a.sketch_dim)
    kw.update(parse_overrides(a.extra))
    kw.update(over)
    return G3Config(**kw)


def backend_from_args(a, cfg):
    from .backend import make_backend
    be = make_backend(a.backend, cfg.d_head, g_max=cfg.g_max) if a.backend == "serial" else make_backend("pallas", cfg.d_head)
    cfg.check_backend(be.bt)
    if abs(be.g_max - cfg.g_max) > 1e-6:
        print(f"[warn] cfg.g_max={cfg.g_max} differs from kernel g_max={be.g_max}; use --extra g_max={be.g_max}", file=sys.stderr)
    return be


def cmd_info(a):
    from .env import device_summary
    import jax
    print(device_summary())
    cfg = cfg_from_args(a)
    print(cfg)
    print("approx params:", f"{cfg.n_params_estimate() / 1e6:.2f}M")
    if a.backend == "pallas":
        from .env import ensure_kernel
        ag = ensure_kernel()
        print("atomic_gdn2:", getattr(ag, "__version__", "?"), ag.__file__)


def cmd_train(a):
    import numpy as np
    from . import tasks
    from .train import train, make_eval_loss
    from .evaluate import eval_mqar, p_star, loss_by_position
    cfg = cfg_from_args(a)
    be = backend_from_args(a, cfg)
    os.makedirs(a.out, exist_ok=True)
    name = a.name or f"{a.task}_{cfg.mode}_s{cfg.seg}_seed{a.seed}"
    log_path = os.path.join(a.out, name + ".jsonl")
    ckpt = os.path.join(a.out, name + ".ckpt")

    if a.task == "mqar":
        P_train = [int(x) for x in a.p_train.split(",")]
        P_eval = [int(x) for x in a.p_eval.split(",")]
        batch_fn = tasks.mqar_batch_fn(P_train, a.n_q, a.L, a.bsz)

        def eval_fn(params, step):
            cur = eval_mqar(params, cfg, be, P_eval, a.L, n_q=a.n_q, B=a.bsz, n_batches=2)
            d = {f"acc_P{P}": v["acc"] for P, v in cur.items()}
            d["P_star"] = float(p_star(cur))
            return d
    else:
        data = tasks.get_bytes(a.data, "train")
        val = tasks.get_bytes(a.data, "val")
        batch_fn = tasks.lm_batch_fn(data, a.L, a.bsz)
        ev = make_eval_loss(cfg, be)

        def eval_fn(params, step):
            import jax.numpy as jnp
            rng = np.random.default_rng(0)
            ls = []
            for _ in range(4):
                b = tasks.lm_batch_fn(val, a.L, a.bsz)(rng, 0)
                l, _ = ev(params, jnp.asarray(b["ids"]), jnp.asarray(b["mask"]))
                ls.append(float(l))
            return dict(val_loss=float(np.mean(ls)), val_bpb=float(np.mean(ls) / np.log(2)))

    params, hist = train(cfg, be, batch_fn, steps=a.steps, lr=a.lr, warmup=a.warmup, wd=a.wd, seed=a.seed,
                         log_every=a.log_every, eval_every=a.eval_every, eval_fn=eval_fn,
                         ckpt_path=ckpt, ckpt_every=a.ckpt_every, resume=a.resume, log_path=log_path)
    print(json.dumps(hist[-1]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gdn3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("info")
    add_model_args(pi)
    pt = sub.add_parser("train")
    add_model_args(pt)
    pt.add_argument("--task", default="mqar", choices=["mqar", "lm"])
    pt.add_argument("--data", default="enwik8", help="enwik8 | synthetic | path to a byte file (task=lm)")
    pt.add_argument("--L", type=int, default=4096)
    pt.add_argument("--bsz", type=int, default=8)
    pt.add_argument("--steps", type=int, default=4000)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--warmup", type=int, default=200)
    pt.add_argument("--wd", type=float, default=0.1)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--log-every", type=int, default=50)
    pt.add_argument("--eval-every", type=int, default=500)
    pt.add_argument("--ckpt-every", type=int, default=500)
    pt.add_argument("--resume", action="store_true")
    pt.add_argument("--out", default="runs")
    pt.add_argument("--name", default="")
    pt.add_argument("--p-train", default="16,32,64,128,256,384")
    pt.add_argument("--p-eval", default="32,64,128,256,384,512,768")
    pt.add_argument("--n-q", type=int, default=16)
    a = ap.parse_args(argv)
    {"info": cmd_info, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
