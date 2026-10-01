"""Language-model training (GDN-2 vs GDN-2c vs GDN-3). Reports val loss, loss-by-position and ms/step.
   python -m train.train_lm --model gdn3 --data enwik8 --L 8192 --bsz 4 --steps 20000 --seg 128 --out runs/lm_gdn3_s0 --seed 0
   --data  enwik8 | bytes:/path.txt | npy:/path/tokens.npy   (for npy pass --vocab = tokenizer size)
Decision rule (docs/EXPERIMENTS.md): GDN-3 must be within +0.005 nats of GDN-2c on val AND better on late positions."""
import argparse, time
import numpy as np, jax, jax.numpy as jnp
from gdn3 import make_backend, init_model, forward, count_params
from gdn3.model import xent
from train.common import add_model_args, cfg_from_args, make_optimizer, make_step, RunLog
from train.data import load_source, Sampler, val_blocks


def main():
    ap = argparse.ArgumentParser(); add_model_args(ap)
    ap.add_argument("--backend", default="atomic"); ap.add_argument("--data", default="enwik8"); ap.add_argument("--L", type=int, default=8192)
    ap.add_argument("--bsz", type=int, default=4); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="runs/lm")
    ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--val_every", type=int, default=1000); ap.add_argument("--val_blocks", type=int, default=32)
    ap.add_argument("--ckpt_every", type=int, default=2000); a = ap.parse_args()
    cfg = cfg_from_args(a); be = make_backend(a.backend, cfg.d_head, **({"g_max": cfg.g_max} if a.backend == "atomic" else {}))
    src = load_source(a.data); samp = Sampler(src["train"], a.L, a.bsz, a.seed); val = val_blocks(src["val"], a.L, a.val_blocks)
    params = init_model(jax.random.PRNGKey(a.seed), cfg); print(f"{a.model}: {count_params(params)/1e6:.2f}M params, seg={cfg.seg}, bank layers "
                                                              f"{[i for i in range(cfg.n_layers) if cfg.has_banks(i)]}")
    opt = make_optimizer(params, a.lr, a.warmup, a.steps); ost = opt.init(params)
    nll_fn = lambda p, ids: xent(forward(p, cfg, be, ids)[:, :-1], ids[:, 1:])
    step = make_step(lambda p, ids: nll_fn(p, ids).mean(), opt); nll_j = jax.jit(nll_fn)
    edges = [0] + [e for e in (512, 1024, 2048, 4096, 8192, 16384, 32768) if e < a.L] + [a.L - 1]

    def evaluate():
        nll = np.concatenate([np.asarray(nll_j(params, jnp.asarray(val[i:i + a.bsz]))) for i in range(0, len(val), a.bsz)], 0)
        return float(nll.mean()), [float(nll[:, x:y].mean()) for x, y in zip(edges[:-1], edges[1:])]

    log = RunLog(a.out, vars(a)); gn_ema = jnp.float32(0.0); skipped = 0; t0 = time.perf_counter()
    for i in range(1, a.steps + 1):
        params, ost, loss, gn, ok, gn_ema = step(params, ost, gn_ema, jnp.asarray(samp.next())); skipped += int(not bool(ok))
        if i % a.log_every == 0:
            ms = (time.perf_counter() - t0) / i * 1e3; print(f"{i:6d} loss={float(loss):.4f} gn={float(gn):.2f} skipped={skipped} {ms:.0f} ms/step"); log.log(step=i, loss=float(loss), gn=float(gn), ms=ms)
        if i % a.val_every == 0 or i == a.steps:
            v, bins = evaluate(); print(f"       [val] {v:.4f}  by-pos {dict(zip(edges[1:], [round(b, 3) for b in bins]))}"); log.log(step=i, val=v, val_by_pos=bins, edges=edges)
        if i % a.ckpt_every == 0:
            log.ckpt(params, f"step{i}.pkl")
    log.ckpt(params, "final.pkl")


if __name__ == "__main__":
    main()
