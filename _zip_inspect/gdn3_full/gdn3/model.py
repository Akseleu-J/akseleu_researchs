"""GDN-3 language model: embedding -> n_layers x (mixer + SwiGLU) -> RMSNorm -> tied head.
Engines: 'parallel' (kernel backend, fixed segmentation) | 'scan' (token scan = streaming code path) | 'auto'.
Any sequence length: the parallel engine right-pads to cfg.align(bt) (causal => the pad cannot influence real positions)."""
import math
import jax
import jax.numpy as jnp

from .layer import init_layer, layer_pre, layer_post, rmsnorm
from .mixer import mixer_parallel, mixer_scan, init_carry


def init_params(key, cfg):
    ks = jax.random.split(key, cfg.n_layers + 1)
    return dict(embed=jax.random.normal(ks[0], (cfg.vocab, cfg.d_model), jnp.float32) * 0.02,
                layers=[init_layer(ks[i + 1], cfg) for i in range(cfg.n_layers)],
                final_norm=jnp.ones((cfg.d_model,), jnp.float32))


def count_params(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def _pick_engine(cfg, be, engine):
    if engine == "auto":
        return "parallel" if (be is not None and cfg.parallel_ok()) else "scan"
    return engine


def layer_fwd(p, cfg, be, x, engine, carry=None, force_gate=None, want_cache=False, N=0, oracle=None):
    B = x.shape[0]
    xn, f, conv = layer_pre(p, cfg, x, None if carry is None else carry["conv"])
    if engine == "parallel":
        o, mix = mixer_parallel(p, cfg, be, f, force_gate, want_cache, N, oracle)
    else:
        mix0 = init_carry(p, cfg, B, N) if carry is None else carry["mix"]
        o, mix = mixer_scan(p, cfg, f, mix0, force_gate)
    x = layer_post(p, cfg, x, xn, o)
    return x, (dict(conv=conv, mix=mix) if (want_cache or engine == "scan") else None)


def forward(params, ids, cfg, be=None, engine="auto", carries=None, want_cache=False, max_banks=0, force_gate=None, oracle=None):
    """ids (B,L) int -> (logits (B,L,V), caches|None).  carries: per-layer caches to continue from (scan engine only)."""
    engine = _pick_engine(cfg, be, engine)
    B, L0 = ids.shape
    if oracle is not None and (engine != "parallel" or not cfg.ha):
        raise ValueError("oracle gate weights need engine='parallel' and an archive (mode != gdn2)")
    if engine == "parallel":
        if carries is not None:
            raise ValueError("continuing from a cache needs engine='scan'")
        cfg.check_backend(be.bt)
        al = cfg.align(be.bt)
        Lp = -(-L0 // al) * al
        if Lp != L0:
            if want_cache:
                raise ValueError("want_cache with the parallel engine needs L % align == 0")
            ids = jnp.pad(ids, ((0, 0), (0, Lp - L0)), constant_values=cfg.pad_id)
            if oracle is not None:
                raise ValueError("oracle needs L % align == 0 (no padding)")
    N = max_banks or cfg.slots(ids.shape[1] if carries is None else 1 << 20) if cfg.ha else 1
    if carries is not None and cfg.ha:
        N = carries[0]["mix"]["arch"]["sid"].shape[-1]
    x = params["embed"][ids]
    out = []
    for i, p in enumerate(params["layers"]):
        carry = None if carries is None else carries[i]
        run = lambda p_, x_, c_, i=i: layer_fwd(p_, cfg, be, x_, engine, c_, force_gate, want_cache, N, oracle)
        if cfg.remat and not want_cache and engine == "parallel":
            x = jax.checkpoint(lambda p_, x_: run(p_, x_, None)[0])(p, x)
            out.append(None)
        else:
            x, c = run(p, x, carry)
            out.append(c)
    logits = rmsnorm(x, params["final_norm"]) @ params["embed"].T
    return logits[:, :L0], (out if (want_cache or engine == "scan") else None)


def prefill(params, ids, cfg, be, max_banks, force_gate=None):
    """Prompt -> (logits, caches). Parallel (kernel) over the aligned prefix when possible + token scan over the tail,
    the archive is capped at max_banks slots by cfg.bank_policy."""
    B, L = ids.shape
    if be is None or not cfg.parallel_ok():
        return forward(params, ids, cfg, be, "scan", None, True, max_banks, force_gate)
    al = cfg.align(be.bt)
    La = (L // al) * al
    outs, caches = [], None
    if La > 0:
        lg, caches = forward(params, ids[:, :La], cfg, be, "parallel", None, True, max_banks, force_gate)
        outs.append(lg)
    if La < L:
        lg, caches = forward(params, ids[:, La:], cfg, be, "scan", caches, True, max_banks, force_gate)
        outs.append(lg)
    return jnp.concatenate(outs, 1), caches


def decode_step(params, caches, tok, cfg, force_gate=None):
    """tok (B,) -> (logits (B,V), caches).  O(1) in context length (archive read touches only top_k banks)."""
    lg, c = forward(params, tok[:, None], cfg, None, "scan", caches, True, 0, force_gate)
    return lg[:, 0], c


def init_caches(params, cfg, B, max_banks):
    return [dict(conv=jnp.zeros((3, B, cfg.d_conv - 1, cfg.proj)), mix=init_carry(p, cfg, B, max_banks))
            for p in params["layers"]]


# ----------------------------------------------------------------------------- losses
def xent(logits, targets):
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return -jnp.take_along_axis(lp, targets[..., None], -1)[..., 0]


def masked_ce(logits, ids, mask=None):
    """Next-token CE (+accuracy). logits (B,L,V) predict ids[:,1:]; mask (B,L) marks TARGET positions."""
    lg, tgt = logits[:, :-1].astype(jnp.float32), ids[:, 1:]
    nll = xent(lg, tgt)
    hit = (jnp.argmax(lg, -1) == tgt).astype(jnp.float32)
    if mask is None:
        return nll.mean(), hit.mean()
    m = mask[:, 1:].astype(jnp.float32)
    den = jnp.maximum(m.sum(), 1.0)
    return (nll * m).sum() / den, (hit * m).sum() / den


def nll_by_position(logits, ids):
    return xent(logits[:, :-1], ids[:, 1:])


def lm_loss(params, cfg, be, ids, engine="auto"):
    return masked_ce(forward(params, ids, cfg, be, engine)[0], ids)[0]
