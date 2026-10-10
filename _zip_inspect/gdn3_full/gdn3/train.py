"""Training loop: AdamW + warmup/cosine, global-norm clip, skip-step on non-finite / spiky gradients, no weight decay on
norms / biases / embeddings / router scalars / decay grid, JSONL log, resumable checkpoints."""
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .model import init_params, forward, masked_ce, count_params

WD_KEYS = {"wq", "wk", "wv", "wb", "ww", "wd", "wog", "wo", "w_gate", "w_up", "w_down", "wgam", "we", "wke", "wve",
           "wbe", "lam_w1", "lam_w2"}


def wd_mask(params):
    return jax.tree_util.tree_map_with_path(lambda path, x: getattr(path[-1], "key", None) in WD_KEYS, params)


def make_optimizer(params, lr, warmup, total, wd=0.1, clip=1.0, end_frac=0.05):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(total, warmup + 1), lr * end_frac)
    return optax.chain(optax.clip_by_global_norm(clip),
                       optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=wd, mask=wd_mask(params)))


def save_ckpt(path, params, ost, step, meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(dict(params=jax.device_get(params), opt=jax.device_get(ost), step=step, meta=meta or {}), f)
    os.replace(path + ".tmp", path)


def load_ckpt(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train(cfg, be, batch_fn, *, steps=2000, lr=1e-3, warmup=200, wd=0.1, clip=1.0, seed=0, log_every=50,
          eval_every=0, eval_fn=None, ckpt_path=None, ckpt_every=500, resume=False, spike_factor=5.0,
          log_path=None, params=None, engine="auto", verbose=True):
    """batch_fn(rng, step) -> dict(ids (B,L) int32, mask (B,L) bool).  eval_fn(params, step) -> dict.  Returns (params, hist)."""
    if params is None:
        params = init_params(jax.random.PRNGKey(seed), cfg)
    opt = make_optimizer(params, lr, warmup, steps, wd, clip)
    ost, start = opt.init(params), 0
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = load_ckpt(ckpt_path)
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        ost = jax.tree_util.tree_map(jnp.asarray, ck["opt"])
        start = int(ck["step"])

    def loss_fn(p, ids, mask, orc=None):
        return masked_ce(forward(p, ids, cfg, be, engine, oracle=orc)[0], ids, mask)

    @jax.jit
    def step_fn(p, s, ids, mask, gn_ema, orc=None):
        (loss, acc), gr = jax.value_and_grad(loss_fn, has_aux=True)(p, ids, mask, orc)
        gn = optax.global_norm(gr)
        ok = jnp.isfinite(loss) & jnp.isfinite(gn) & ((gn_ema <= 0) | (gn < spike_factor * gn_ema))
        upd, s2 = opt.update(gr, s, p)
        p2 = optax.apply_updates(p, upd)
        keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), new, old)
        gn_new = jnp.where(ok, jnp.where(gn_ema > 0, 0.98 * gn_ema + 0.02 * gn, gn), gn_ema)
        return keep(p2, p), keep(s2, s), loss, acc, gn, ok, gn_new

    rng = np.random.default_rng(seed * 7919 + start)
    gn_ema, hist, skipped = jnp.float32(0.0), [], 0
    logf = open(log_path, "a") if log_path else None
    if verbose:
        print(f"[train] mode={cfg.mode} cell={cfg.cell} seg={cfg.seg} params={count_params(params) / 1e6:.3f}M steps={steps}")
    t0 = time.perf_counter()
    for i in range(start + 1, steps + 1):
        b = batch_fn(rng, i)
        params, ost, loss, acc, gn, ok, gn_ema = step_fn(params, ost, jnp.asarray(b["ids"]), jnp.asarray(b["mask"]), gn_ema,
                                                         jnp.asarray(b["oracle"]) if "oracle" in b else None)
        skipped += int(not bool(ok))
        rec = None
        if i % log_every == 0 or i == steps:
            rec = dict(step=i, loss=float(loss), acc=float(acc), gnorm=float(gn), skipped=skipped,
                       ms_per_step=(time.perf_counter() - t0) / max(i - start, 1) * 1e3)
            hist.append(rec)
            if verbose:
                print(f"{i:6d} loss={rec['loss']:.4f} acc={rec['acc']:.4f} gnorm={rec['gnorm']:.2f} skip={skipped} "
                      f"{rec['ms_per_step']:.1f} ms/step", flush=True)
        if eval_every and eval_fn is not None and (i % eval_every == 0 or i == steps):
            ev = eval_fn(params, i)
            rec = dict(rec or dict(step=i), **{f"eval_{k}": v for k, v in ev.items()})
            hist.append(dict(step=i, **{f"eval_{k}": v for k, v in ev.items()}))
            if verbose:
                print("       [eval]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}, flush=True)
        if rec and logf:
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if ckpt_path and ckpt_every and i % ckpt_every == 0:
            save_ckpt(ckpt_path, params, ost, i, dict(cfg=cfg.__dict__))
    if ckpt_path:
        save_ckpt(ckpt_path, params, ost, steps, dict(cfg=cfg.__dict__))
    if logf:
        logf.close()
    return params, hist
