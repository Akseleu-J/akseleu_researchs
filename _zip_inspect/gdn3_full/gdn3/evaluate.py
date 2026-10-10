"""Metrics: MQAR capacity curve + P* (the M1 criterion P*(hybrid) >= 2 P*(gdn2c)), streaming evaluation with a capped
archive, loss by position."""
import jax
import jax.numpy as jnp
import numpy as np

from .model import forward, masked_ce, nll_by_position, prefill
from .tasks import make_mqar, oracle_weights


def eval_mqar(params, cfg, be, P_list, L, n_q=16, B=8, n_batches=4, seed=1234, engine="auto", oracle_seg=None):
    """acc / loss at the answer tokens for each P (one compilation: fixed L)."""
    @jax.jit
    def run(p, ids, mask, orc=None):
        return masked_ce(forward(p, ids, cfg, be, engine, oracle=orc)[0], ids, mask)
    rng, out = np.random.default_rng(seed), {}
    for P in P_list:
        ls, accs = [], []
        for _ in range(n_batches):
            ids, mask, qi = make_mqar(rng, B, P, n_q, L)
            orc = jnp.asarray(oracle_weights(qi, P, n_q, L, oracle_seg)) if oracle_seg else None
            l, a = run(params, jnp.asarray(ids), jnp.asarray(mask), orc)
            ls.append(float(l)); accs.append(float(a))
        out[int(P)] = dict(acc=float(np.mean(accs)), loss=float(np.mean(ls)))
    return out


def p_star(curve, thresh=0.9):
    """Largest P with acc >= thresh (curve {P: {'acc': ..}}); 0 if none."""
    ok = [P for P, v in curve.items() if v["acc"] >= thresh]
    return max(ok) if ok else 0


def eval_mqar_stream(params, cfg, be, P, L, max_banks, n_q=16, B=4, n_batches=2, seed=4321):
    """Same task through prefill (facts + filler, archive CAPPED at max_banks by cfg.bank_policy) + token-scan over the
    queries (teacher forced).  Only the query part sees the capped archive, so this measures the budget, not the prefill."""
    rng, hits, tot = np.random.default_rng(seed), 0, 0
    nqt = 4 * n_q
    for _ in range(n_batches):
        ids, mask, _ = make_mqar(rng, B, P, n_q, L)
        ids_j = jnp.asarray(ids)
        _, caches = prefill(params, ids_j[:, :L - nqt], cfg, be, max_banks)
        lg, _ = forward(params, ids_j[:, L - nqt:], cfg, be, "scan", caches, True, max_banks)
        pred = np.asarray(jnp.argmax(lg[:, :-1], -1))
        tgt, m = ids[:, L - nqt + 1:], mask[:, L - nqt + 1:]
        hits += int(((pred == tgt) & m).sum()); tot += int(m.sum())
    return hits / max(tot, 1)


def loss_by_position(params, cfg, be, data, L, edges, B=2, n_batches=8, seed=0, engine="auto"):
    """Mean NLL (nats/byte) in position bins for windows of length L (edges increasing, last <= L-1)."""
    @jax.jit
    def run(p, ids):
        return nll_by_position(forward(p, ids, cfg, be, engine)[0], ids)
    rng, acc = np.random.default_rng(seed), np.zeros(len(edges) - 1)
    for _ in range(n_batches):
        st = rng.integers(0, len(data) - L - 1, B)
        nll = np.asarray(run(params, jnp.asarray(np.stack([data[s:s + L] for s in st]).astype(np.int32))))
        for i in range(len(edges) - 1):
            acc[i] += nll[:, edges[i]:edges[i + 1]].mean()
    return (acc / n_batches).tolist()
