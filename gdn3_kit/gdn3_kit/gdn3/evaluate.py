"""Evaluation: MQAR capacity curve, P*, loss-by-position, streaming (capped archive) evaluation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .model import forward, masked_ce, nll_by_position
from .tasks import make_mqar, oracle_weights
from .cache import prefill


def eval_mqar(params, cfg, be, P_list, L, n_q=16, B=8, n_batches=4, seed=1234, oracle=False):
    """Accuracy at the answer tokens for each P.  One compilation (fixed L)."""
    n = -(-L // cfg.seg)

    @jax.jit
    def run(p, ids, mask, orc):
        lg = forward(p, ids, cfg, be, oracle=orc)[0]
        return masked_ce(lg, ids, mask)

    @jax.jit
    def run_plain(p, ids, mask):
        return masked_ce(forward(p, ids, cfg, be)[0], ids, mask)

    rng = np.random.default_rng(seed)
    out = {}
    for P in P_list:
        accs, losses = [], []
        for _ in range(n_batches):
            ids, mask, qi = make_mqar(rng, B, P, n_q, L)
            if oracle:
                orc = jnp.asarray(oracle_weights(qi, P, n_q, L, cfg.seg, n))
                l, a = run(params, jnp.asarray(ids), jnp.asarray(mask), orc)
            else:
                l, a = run_plain(params, jnp.asarray(ids), jnp.asarray(mask))
            accs.append(float(a)); losses.append(float(l))
        out[int(P)] = dict(acc=float(np.mean(accs)), loss=float(np.mean(losses)))
    return out


def p_star(curve: dict, thresh: float = 0.9) -> int:
    """Largest P whose accuracy >= thresh (curve: {P: {'acc': ..}}); 0 if none."""
    ok = [P for P, v in curve.items() if v["acc"] >= thresh]
    return max(ok) if ok else 0


def eval_mqar_stream(params, cfg, be, P, L, max_banks, n_q=16, B=4, n_batches=2, seed=4321):
    """Same task through prefill()+decode steps with a CAPPED archive (policy cfg.bank_policy)."""
    @jax.jit
    def run(p, ids):
        lg, _ = prefill(p, ids, cfg, be, max_banks)
        return lg

    rng = np.random.default_rng(seed)
    hits = tot = 0
    for _ in range(n_batches):
        ids, mask, _ = make_mqar(rng, B, P, n_q, L)
        lg = np.asarray(run(params, jnp.asarray(ids)))
        pred = lg[:, :-1].argmax(-1)
        m = mask[:, 1:]
        hits += int(((pred == ids[:, 1:]) & m).sum())
        tot += int(m.sum())
    return hits / max(tot, 1)


def loss_by_position(params, cfg, be, data, L, edges, B=2, n_batches=8, seed=0):
    """Mean NLL (nats/byte) in position bins for windows of length L. edges: increasing bin edges, last <= L-1."""
    @jax.jit
    def run(p, ids):
        return nll_by_position(forward(p, ids, cfg, be)[0], ids)

    rng = np.random.default_rng(seed)
    acc = np.zeros(len(edges) - 1)
    for _ in range(n_batches):
        starts = rng.integers(0, len(data) - L - 1, B)
        ids = np.stack([data[s:s + L] for s in starts]).astype(np.int32)
        nll = np.asarray(run(params, jnp.asarray(ids)))                  # (B, L-1)
        for i in range(len(edges) - 1):
            acc[i] += nll[:, edges[i]:edges[i + 1]].mean()
    return (acc / n_batches).tolist()
