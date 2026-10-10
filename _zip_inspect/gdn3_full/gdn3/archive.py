"""Dynamic bank archive with static shapes (N slots, chronological, compacted left) + budget policies.

One ROW = one (batch, head) archive; the batched functions vmap over (B, Ha).  Fields per slot:
  S | (Sq,sc,cv,ci)  bank matrix (float, or int8 + scale + exact outliers = D1 LeapQuant)
  T (D,m)  JL sketch ;  Sf ||S||_F^2 ;  span (D) total log-decay inside the bank's own span
  acc (D)  log-decay accumulated since the bank's end (B6: read weight exp(acc)) ;  sid  segment id (-1 empty) ;  lvl  Fenwick level
Policies when the archive is full (or compacted):
  fifo    drop the oldest bank
  merge   S = diag(exp(span_new)) S_old + S_new  (decay-correct superposition; exact when beta=0), merge the two oldest
  leveled (alias fenwick) log-linear, LSM-style (not a Fenwick tree): more than per_level banks on one level -> merge the two oldest of that level (level+1);
          if still full, merge the two oldest.
"""
import jax
import jax.numpy as jnp

from . import ops
from .routing import frob2, bank_gate, log_ratio, read_core


def init_arch(cfg, B, Ha, N, R_unused=None):
    D, m, z = cfg.d_head, cfg.sketch_m, jnp.zeros
    a = dict(span=z((B, Ha, N, D)), acc=z((B, Ha, N, D)), sid=-jnp.ones((B, Ha, N), jnp.int32),
             lvl=z((B, Ha, N), jnp.int32), Sf=z((B, Ha, N)))
    if m:
        a["T"] = z((B, Ha, N, D, m))
    if cfg.quant == "int8":
        K = min(cfg.quant_k, D * D)
        a.update(Sq=z((B, Ha, N, D, D), jnp.int8), sc=jnp.ones((B, Ha, N, 1, 1)),
                 cv=z((B, Ha, N, K)), ci=z((B, Ha, N, K), jnp.int32))
    else:
        a["S"] = z((B, Ha, N, D, D))
    return a


def entry_S(cfg, e):
    """Float bank matrix (...,D,D) of an entry / stack of entries."""
    if cfg.quant == "int8":
        return ops.leap_dequantize(e["Sq"], e["sc"], (e["cv"], e["ci"]))
    return e["S"]


def entry_new(cfg, R, S, span, sid, acc=None, lvl=0):
    e = dict(span=span, acc=jnp.zeros_like(span) if acc is None else acc, sid=jnp.asarray(sid, jnp.int32),
             lvl=jnp.asarray(lvl, jnp.int32), Sf=frob2(S))
    if R is not None:
        e["T"] = S @ R
    if cfg.quant == "int8":
        Sq, sc = ops.leap_quantize(S)
        cv, ci = ops.leap_compensator(S, cfg.quant_k)
        e.update(Sq=Sq, sc=sc, cv=cv, ci=ci)
    else:
        e["S"] = S
    return e


def merge_entries(cfg, R, a, b):
    """older a, newer b -> bank over the union span (decay-only transition: exact for beta=0, lossy otherwise)."""
    S = jnp.exp(b["span"])[:, None] * entry_S(cfg, a) + entry_S(cfg, b)
    return entry_new(cfg, R, S, a["span"] + b["span"], b["sid"], acc=b["acc"], lvl=jnp.maximum(a["lvl"], b["lvl"]) + 1)


def _tw(pred, x, y):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(pred, a, b), x, y)


def _take(arch, i):
    return jax.tree_util.tree_map(lambda x: x[i], arch)


def _bc(mask, a):
    return mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim))


def _remove_at(cfg, R, arch, i, merge):
    """Remove slot i (merge=False) or merge slots i,i+1 into slot i (merge=True); compacts left."""
    N = arch["sid"].shape[0]
    idx = jnp.arange(N)
    cnt = jnp.sum(arch["sid"] >= 0)
    src = jnp.minimum(idx + (idx > i), N - 1)
    if merge:
        m = merge_entries(cfg, R, _take(arch, i), _take(arch, jnp.minimum(i + 1, N - 1)))
        out = jax.tree_util.tree_map(lambda a, mv: jnp.where(_bc(idx == i, a), mv[None], a[src]), arch, m)
    else:
        out = jax.tree_util.tree_map(lambda a: a[src], arch)
    keep = idx < cnt - 1
    out = jax.tree_util.tree_map(lambda a: jnp.where(_bc(keep, a), a, jnp.zeros_like(a)), out)
    out["sid"] = jnp.where(keep, out["sid"], -1)
    return out


def append_row(cfg, R, arch, S_new, span_new, sid_new):
    N = arch["sid"].shape[0]
    idx = jnp.arange(N)
    new = entry_new(cfg, R, S_new, span_new, sid_new)
    full = jnp.sum(arch["sid"] >= 0) >= N
    room = _remove_at(cfg, R, arch, 0, merge=(cfg.bank_policy != "fifo"))
    arch = _tw(full, room, arch)
    oh = idx == jnp.sum(arch["sid"] >= 0)
    arch = jax.tree_util.tree_map(lambda a, n: jnp.where(_bc(oh, a), n[None], a), arch, new)
    if cfg.bank_policy in ("fenwick", "leveled"):
        levels = jnp.arange(cfg.max_levels + 2)
        for _ in range(cfg.max_levels):
            valid, lv = arch["sid"] >= 0, arch["lvl"]
            over = jnp.sum((lv[None, :] == levels[:, None]) & valid[None, :], axis=1) > cfg.per_level
            i = jnp.argmax((lv == jnp.argmax(over)) & valid)
            arch = _tw(jnp.any(over), _remove_at(cfg, R, arch, i, merge=True), arch)
    return arch


def append_batched(cfg, R, arch, S_new, span_new, sid_new):
    """arch leaves (B,Ha,N,...); S_new (B,Ha,D,D); span_new (B,Ha,D); sid_new (B,Ha)."""
    f = lambda a, s, sp, sid: append_row(cfg, R, a, s, sp, sid)
    return jax.vmap(jax.vmap(f))(arch, S_new, span_new, sid_new)


def read_row(cfg, R, q, arch, lamvec, ccql, tau, lam0, null, nseg, scale):
    """Archive read for ONE token of one head. q (D,) -> (D,).  Sparse: only the top_k selected banks are touched."""
    D = q.shape[-1]
    valid = arch["sid"] >= 0
    if "T" in arch:
        e = jnp.sum(jnp.einsum("d,ndm->nm", q, arch["T"], precision=jax.lax.Precision.HIGHEST) ** 2, -1)
    else:
        e = jnp.sum(jnp.einsum("d,nde->ne", q, arch["S"], precision=jax.lax.Precision.HIGHEST) ** 2, -1)
    logr = log_ratio(e, arch["Sf"], jnp.sum(q * q), D)
    age = jnp.maximum(nseg - 1 - arch["sid"], 0).astype(jnp.float32)
    if cfg.adaptive_lam:
        lam_eff = lamvec[jnp.clip(jnp.floor(jnp.log2(1.0 + age)).astype(jnp.int32), 0, cfg.n_lv - 1)]
    else:
        lam_eff = lam0
    a = bank_gate(tau * logr - lam_eff * jnp.log1p(age), valid, tau * null, cfg.top_k)
    w_, idx = jax.lax.top_k(a, min(cfg.top_k, a.shape[0]))
    sel = jax.tree_util.tree_map(lambda x: x[idx], {k: v for k, v in arch.items() if k not in ("T",)})
    S = entry_S(cfg, sel)                                                     # (K,D,D)
    qe = q[None] * (jnp.exp(jnp.minimum(sel["acc"], 0.0)) if cfg.bank_decay else 1.0)
    hi = jax.lax.Precision.HIGHEST
    r = read_core(qe, lambda x: jnp.einsum("kd,kde->ke", x, S, precision=hi),
                  lambda x: jnp.einsum("ke,kde->kd", x, S, precision=hi),
                  jnp.sqrt(jnp.sum(S * S, -1)), sel["Sf"][:, None], ccql, scale, cfg)
    return jnp.sum(w_[:, None] * r, axis=0)


def read_batched(cfg, R, q, arch, lamvec, ccql, tau, lam0, null, nseg, scale):
    """q (B,Ha,D); arch (B,Ha,N,..); lamvec (B,Ha,nl); ccql (B,Ha); tau,lam0,null (Ha,); nseg (B,Ha)."""
    f = lambda q_, a_, lv, cl, t_, l0, nu, ns: read_row(cfg, R, q_, a_, lv, cl, t_, l0, nu, ns, scale)
    inner = jax.vmap(f, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))
    return jax.vmap(inner, in_axes=(0, 0, 0, 0, None, None, None, 0))(q, arch, lamvec, ccql, tau, lam0, null, nseg)


def export_archive(cfg, R, S, Gend, N):
    """Parallel-engine banks S (B,Ha,n,D,D) with cumulative Gend (B,Ha,n,D) -> streaming archive with N slots, applying the
    budget policy bank by bank (lax.scan over banks, same code path as the streaming append)."""
    B, Ha, n = S.shape[:3]
    span = Gend - jnp.concatenate([jnp.zeros_like(Gend[:, :, :1]), Gend[:, :, :-1]], axis=2)

    def body(a, x):
        S_j, sp, j = x
        a = dict(a, acc=a["acc"] + sp[:, :, None, :])          # existing banks decay through bank j's span
        return append_batched(cfg, R, a, S_j, sp, jnp.full((B, Ha), j, jnp.int32)), None

    arch, _ = jax.lax.scan(body, init_arch(cfg, B, Ha, N),
                           (jnp.moveaxis(S, 2, 0), jnp.moveaxis(span, 2, 0), jnp.arange(n, dtype=jnp.int32)))
    return arch
