"""Capacity curve acc(P, L) for a trained model; P* = max P with acc >= thr (per L).
   python -m bench.eval_mqar --ckpt runs/x/final.pkl --model gdn3 --Ps 64 128 256 512 1024 --Ls 4096 16384"""
import argparse, json, pickle, time
import numpy as np, jax, jax.numpy as jnp
from gdn3 import make_backend, forward
from bench.mqar import make_batch, mqar_loss_acc
from train.common import cfg_from_args, add_model_args


def evaluate(params, cfg, backend, Ps, Ls, bsz=4, n_q=32, nk=2048, nv=64, nf=32, seed=123, thr=0.9, log=print):
    fwd = jax.jit(lambda p, ids: forward(p, cfg, backend, ids))
    res = {}
    for L in Ls:
        for P in Ps:
            if 2 * P + 3 * n_q > L:
                continue
            rng = np.random.default_rng(seed + P); accs = []
            for _ in range(2):
                ids, m = make_batch(rng, bsz, L, P, n_q, nk, nv, nf)
                accs.append(float(mqar_loss_acc(fwd(params, jnp.asarray(ids)), jnp.asarray(ids), jnp.asarray(m))[1]))
            res[(L, P)] = float(np.mean(accs)); log(f"  L={L:>6} P={P:>5} acc={res[(L, P)]:.3f}")
    star = {L: max([P for P in Ps if res.get((L, P), 0) >= thr] or [0]) for L in Ls}
    log(f"P*(acc>={thr}) per L: {star}")
    return res, star


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); add_model_args(ap)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--backend", default="atomic")
    ap.add_argument("--Ps", type=int, nargs="+", default=[64, 128, 256, 512, 1024]); ap.add_argument("--Ls", type=int, nargs="+", default=[4096])
    ap.add_argument("--out", default=None); a = ap.parse_args()
    cfg = cfg_from_args(a); be = make_backend(a.backend, cfg.d_head, **({"g_max": cfg.g_max} if a.backend == "atomic" else {}))
    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(a.ckpt, "rb")))
    res, star = evaluate(params, cfg, be, a.Ps, a.Ls)
    if a.out: json.dump({"acc": {f"{L}_{P}": v for (L, P), v in res.items()}, "Pstar": star}, open(a.out, "w"), indent=1)
