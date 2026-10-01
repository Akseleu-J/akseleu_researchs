"""Train on a mixture of MQAR capacities, then evaluate the capacity curve.
   python -m train.train_mqar --model gdn3 --steps 6000 --L 4096 --seg 128 --out runs/mqar_gdn3_s0 --seed 0
Compare gdn2 / gdn2c / gdn3 (same L, steps, seed set). Report P* from the final eval."""
import argparse, time
import numpy as np, jax, jax.numpy as jnp
from gdn3 import make_backend, init_model, forward, count_params
from bench.mqar import make_batch, mqar_loss_acc, vocab_of
from bench.eval_mqar import evaluate
from train.common import add_model_args, cfg_from_args, make_optimizer, make_step, RunLog


def main():
    ap = argparse.ArgumentParser(); add_model_args(ap)
    ap.add_argument("--backend", default="atomic"); ap.add_argument("--L", type=int, default=4096); ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--P_train", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512]); ap.add_argument("--n_q", type=int, default=32)
    ap.add_argument("--nk", type=int, default=2048); ap.add_argument("--nv", type=int, default=64); ap.add_argument("--nf", type=int, default=32)
    ap.add_argument("--eval_Ps", type=int, nargs="+", default=[64, 128, 256, 512, 1024]); ap.add_argument("--eval_Ls", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="runs/mqar"); ap.add_argument("--log_every", type=int, default=100)
    a = ap.parse_args()
    cfg = cfg_from_args(a, vocab=vocab_of(a.nk, a.nv, a.nf))
    be = make_backend(a.backend, cfg.d_head, **({"g_max": cfg.g_max} if a.backend == "atomic" else {}))
    params = init_model(jax.random.PRNGKey(a.seed), cfg); print(f"{a.model}: {count_params(params)/1e6:.2f}M params, banks in layers "
                                                              f"{[i for i in range(cfg.n_layers) if cfg.has_banks(i)]}, seg={cfg.seg}")
    opt = make_optimizer(params, a.lr, a.warmup, a.steps); ost = opt.init(params)

    def loss_fn(p, ids, m):
        l, _ = mqar_loss_acc(forward(p, cfg, be, ids), ids, m); return l

    step = make_step(loss_fn, opt); log = RunLog(a.out, vars(a)); rng = np.random.default_rng(a.seed); gn_ema = jnp.float32(0.0); t0 = time.perf_counter(); skipped = 0
    for i in range(1, a.steps + 1):
        ids, m = make_batch(rng, a.bsz, a.L, a.P_train, a.n_q, a.nk, a.nv, a.nf)
        params, ost, loss, gn, ok, gn_ema = step(params, ost, gn_ema, jnp.asarray(ids), jnp.asarray(m)); skipped += int(not bool(ok))
        if i % a.log_every == 0:
            ms = (time.perf_counter() - t0) / i * 1e3; print(f"{i:6d} loss={float(loss):.4f} gn={float(gn):.2f} skipped={skipped} {ms:.0f} ms/step"); log.log(step=i, loss=float(loss), gn=float(gn), ms=ms)
    log.ckpt(params, "final.pkl")
    res, star = evaluate(params, cfg, be, a.eval_Ps, a.eval_Ls, nk=a.nk, nv=a.nv, nf=a.nf, n_q=a.n_q)
    log.log(final=True, Pstar={str(k): v for k, v in star.items()}, acc={f"{L}_{P}": v for (L, P), v in res.items()})


if __name__ == "__main__":
    main()
