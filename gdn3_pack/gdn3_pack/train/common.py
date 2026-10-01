"""Shared training utilities: model args, optimizer, skip-step on non-finite/spiky grads, JSONL logging, checkpoints."""
import json, os, pickle, time
import numpy as np, jax, jax.numpy as jnp, optax
from gdn3 import GDN3Config


def add_model_args(ap):
    ap.add_argument("--model", choices=["gdn2", "gdn2c", "gdn3"], default="gdn3",
                    help="gdn2: banks off, scalar decay per head (report baseline); gdn2c: banks off, per-channel decay; gdn3: banks + per-channel decay")
    ap.add_argument("--d_model", type=int, default=512); ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=128); ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--mlp_mult", type=int, default=2); ap.add_argument("--seg", type=int, default=128)
    ap.add_argument("--bank_every", type=int, default=2); ap.add_argument("--gate_init", type=float, default=0.1)
    ap.add_argument("--n_landmarks", type=int, default=0); ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--remat", action="store_true")
    return ap


def cfg_from_args(a, vocab=None):
    kw = dict(vocab=vocab or a.vocab, d_model=a.d_model, n_heads=a.n_heads, d_head=a.d_head, n_layers=a.n_layers,
              mlp_mult=a.mlp_mult, seg=a.seg, bank_every=a.bank_every, gate_init=a.gate_init, n_landmarks=a.n_landmarks, remat=a.remat)
    if a.model == "gdn2":
        kw.update(decay_mode="head", bank_every=0, bank_layers=(), n_landmarks=0)
    elif a.model == "gdn2c":
        kw.update(decay_mode="channel", bank_every=0, bank_layers=(), n_landmarks=0)
    return GDN3Config(**kw)


def wd_mask(params):
    def f(path, x):
        name = "/".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
        return name.endswith("/w") and "conv_" not in name
    return jax.tree_util.tree_map_with_path(f, params)


def make_optimizer(params, lr, warmup, total, wd=0.1, clip=1.0, final_frac=0.1):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(total, warmup + 1), lr * final_frac)
    return optax.chain(optax.clip_by_global_norm(clip), optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=wd, mask=wd_mask(params)))


def make_step(loss_fn, opt, spike=5.0):
    @jax.jit
    def step(params, ost, gn_ema, *batch):
        loss, gr = jax.value_and_grad(loss_fn)(params, *batch)
        gn = optax.global_norm(gr)
        ok = jnp.isfinite(loss) & jnp.isfinite(gn) & ((gn_ema <= 0) | (gn < spike * gn_ema))
        upd, ost2 = opt.update(gr, ost, params); p2 = optax.apply_updates(params, upd)
        keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), new, old)
        gn_new = jnp.where(ok, jnp.where(gn_ema > 0, 0.98 * gn_ema + 0.02 * gn, gn), gn_ema)
        return keep(p2, params), keep(ost2, ost), loss, gn, ok, gn_new
    return step


class RunLog:
    def __init__(self, out_dir, cfg_dict):
        os.makedirs(out_dir, exist_ok=True); self.dir = out_dir; self.f = open(os.path.join(out_dir, "log.jsonl"), "a")
        json.dump(cfg_dict, open(os.path.join(out_dir, "config.json"), "w"), indent=1, default=str)

    def log(self, **kw):
        self.f.write(json.dumps(kw, default=float) + "\n"); self.f.flush()

    def ckpt(self, params, name):
        pickle.dump(jax.device_get(params), open(os.path.join(self.dir, name), "wb"))
