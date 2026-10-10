"""Training loop: AdamW + warmup/cosine, global-norm clip, skip-step on non-finite / spiky gradients,
no weight-decay on norms/biases/embeddings/archive scalars, JSONL logging, resumable checkpoints."""
from __future__ import annotations
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .model import init_params, forward, masked_ce, count_params

WD_KEYS = {"wq", "wk", "wv", "wb", "ww", "wd", "wog", "wo", "w_gate", "w_up", "w_down"}


def wd_mask(params):
    def f(path, x):
        last = path[-1]
        return getattr(last, "key", None) in WD_KEYS
    return jax.tree_util.tree_map_with_path(f, params)


def make_optimizer(params, lr, warmup, total, wd=0.1, clip=1.0, lr_end_frac=0.05):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, total, lr * lr_end_frac)
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


def train(cfg, be, batch_fn, *, steps=2000, lr=1e-3, warmup=200, wd=0.1, clip=1.0, seed=0,
          log_every=50, eval_every=0, eval_fn=None, ckpt_path=None, ckpt_every=500, resume=False,
          spike_factor=5.0, log_path=None, params=None, verbose=True):
    """batch_fn(rng, step) -> dict(ids=(B,L) int32, mask=(B,L) bool).
    eval_fn(params, step) -> dict (logged).  Returns (params, history)."""
    key = jax.random.PRNGKey(seed)
    if params is None:
        params = init_params(key, cfg)
    opt = make_optimizer(params, lr, warmup, steps, wd, clip)
    ost = opt.init(params)
    start = 0
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = load_ckpt(ckpt_path)
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        ost = jax.tree_util.tree_map(jnp.asarray, ck["opt"])
        start = int(ck["step"])
        if verbose:
            print(f"[resume] step {start} from {ckpt_path}")

    def loss_fn(p, ids, mask):
        logits = forward(p, ids, cfg, be)[0]
        return masked_ce(logits, ids, mask)

    @jax.jit
    def step_fn(p, s, ids, mask, gn_ema):
        (loss, acc), gr = jax.value_and_grad(loss_fn, has_aux=True)(p, ids, mask)
        gn = optax.global_norm(gr)
        ok = jnp.isfinite(loss) & jnp.isfinite(gn) & ((gn_ema <= 0) | (gn < spike_factor * gn_ema))
        upd, s2 = opt.update(gr, s, p)
        p2 = optax.apply_updates(p, upd)
        keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), new, old)
        gn_new = jnp.where(ok, jnp.where(gn_ema > 0, 0.98 * gn_ema + 0.02 * gn, gn), gn_ema)
        return keep(p2, p), keep(s2, s), loss, acc, gn, ok, gn_new

    rng = np.random.default_rng(seed * 7919 + start)
    gn_ema = jnp.float32(0.0)
    hist, skipped = [], 0
    logf = open(log_path, "a") if log_path else None
    if verbose:
        print(f"[train] mode={cfg.mode} seg={cfg.seg} params={count_params(params)/1e6:.2f}M steps={steps} lr={lr}")
    t0 = time.perf_counter()
    for i in range(start + 1, steps + 1):
        b = batch_fn(rng, i)
        params, ost, loss, acc, gn, ok, gn_ema = step_fn(
            params, ost, jnp.asarray(b["ids"]), jnp.asarray(b["mask"]), gn_ema)
        skipped += int(not bool(ok))
        rec = None
        if i % log_every == 0 or i == steps:
            dt = (time.perf_counter() - t0) / max(i - start, 1)
            rec = dict(step=i, loss=float(loss), acc=float(acc), gnorm=float(gn), skipped=skipped, ms_per_step=dt * 1e3)
            hist.append(rec)
            if verbose:
                print(f"{i:6d} loss={rec['loss']:.4f} acc={rec['acc']:.4f} gnorm={rec['gnorm']:.2f} "
                      f"skip={skipped} {rec['ms_per_step']:.1f} ms/step", flush=True)
        if eval_every and eval_fn is not None and (i % eval_every == 0 or i == steps):
            ev = eval_fn(params, i)
            rec = dict(rec or dict(step=i), **{f"eval_{k}": v for k, v in ev.items()})
            if verbose:
                print("       [eval]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}, flush=True)
            hist.append(dict(step=i, **{f"eval_{k}": v for k, v in ev.items()}))
        if rec and logf:
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if ckpt_path and ckpt_every and i % ckpt_every == 0:
            save_ckpt(ckpt_path, params, ost, i, meta=dict(cfg=cfg.__dict__))
    if ckpt_path:
        save_ckpt(ckpt_path, params, ost, steps, meta=dict(cfg=cfg.__dict__))
    if logf:
        logf.close()
    return params, hist


def make_eval_loss(cfg, be):
    @jax.jit
    def f(params, ids, mask):
        return masked_ce(forward(params, ids, cfg, be)[0], ids, mask)
    return f
