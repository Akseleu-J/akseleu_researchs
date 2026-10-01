"""Kernel backends. Common signature (head-major):
     call(q, k, v, w, b, g, h0=None, return_vnew=False) -> (o, h_final) | (o, h_final, v_new)
     q..g: (B,H,L,D) float32, g <= 0 per key-channel, h0/h_final: (B,H,D,D) [key,value].
  'ref'    : pure-JAX token-serial scan (autodiff, any L, any device) = ground truth recurrence
  'atomic' : atomic_gdn2 Pallas kernels (TPU v5e; interpret mode on CPU). L % 128 == 0."""
import os, sys
import jax, jax.numpy as jnp

HIGHEST = jax.lax.Precision.HIGHEST


def make_ref_backend(scale):
    scale = float(scale)

    def call(q, k, v, w, b, g, h0=None, return_vnew=False):
        B, H, L, D = q.shape
        h0 = jnp.zeros((B, H, D, D), jnp.float32) if h0 is None else h0.astype(jnp.float32)
        alpha = jnp.exp(jnp.minimum(g.astype(jnp.float32), 0.0))

        def step(h, xs):
            q_t, k_t, v_t, a_t, b_t, w_t = xs
            h = h * a_t[..., :, None]
            erase = jnp.einsum("bhd,bhdv->bhv", b_t * k_t, h, precision=HIGHEST)
            vn = w_t * v_t - erase
            h = h + k_t[..., :, None] * vn[..., None, :]
            o = jnp.einsum("bhdv,bhd->bhv", h, q_t * scale, precision=HIGHEST)
            return h, (o, vn)

        xs = tuple(jnp.moveaxis(x.astype(jnp.float32), 2, 0) for x in (q, k, v, alpha, b, w))
        hf, (o, vn) = jax.lax.scan(step, h0, xs)
        o = jnp.moveaxis(o, 0, 2)
        return (o, hf, jnp.moveaxis(vn, 0, 2)) if return_vnew else (o, hf)

    call.name = "ref"
    return call


def ensure_atomic_path(repo_dir=None):
    """Make `import atomic_gdn2` work: search a cloned repo for atomic_gdn2/__init__.py and add its parent to sys.path."""
    try:
        import atomic_gdn2  # noqa
        return True
    except ImportError:
        pass
    roots = [repo_dir] if repo_dir else ["/content/atomic-ops", "/kaggle/working/atomic-ops", os.getcwd()]
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for dp, dn, fn in os.walk(r):
            if os.path.basename(dp) == "atomic_gdn2" and "__init__.py" in fn and "config.py" in fn:
                sys.path.insert(0, os.path.dirname(dp)); return True
    return False


def _make_vnew_core(kcfg, scale):
    """EXPERIMENTAL: copy of atomic_gdn2.pipeline with one extra non-differentiable output v_new (for landmarks)."""
    from atomic_gdn2.precision import sanitize
    from atomic_gdn2.layout import to_chunks_hm, from_chunks_hm
    from atomic_gdn2.fwd import build_scores, wy_solve_leafbatched, recompute_and_scan_slim
    from atomic_gdn2.bwd import backward_mega
    cfg = kcfg; scale = float(scale)

    def _fwd(q, k, v, w, b, g, h0):
        gc = jnp.cumsum(g.astype(jnp.float32), axis=-2)
        Aqk, Akk = build_scores(q, k, b, gc, scale, cfg)
        A = wy_solve_leafbatched(Akk, cfg)
        cd = recompute_and_scan_slim(Aqk, q, k, v, w, b, gc, A, scale, h0, cfg)
        return cd["o"], cd["h_final"], dict(gc=gc, Aqk=Aqk, A=A, h_pre_all=cd["h_pre_all"],
                                            v_new_all=cd["v_new_all"], gc_last=cd["gc_last"])

    @jax.custom_vjp
    def core(q, k, v, w, b, g, h0):
        o, hf, res = _fwd(q, k, v, w, b, g, h0)
        return o, hf, res["v_new_all"]

    def core_fwd(q, k, v, w, b, g, h0):
        o, hf, res = _fwd(q, k, v, w, b, g, h0)
        vn = res["v_new_all"]
        res = dict(res); res.update(q=q, k=k, v=v, w=w, b=b, g=g, h0=h0)
        return (o, hf, vn), res

    def core_bwd(res, cts):
        do, dhf, _ = cts                                  # cotangent of v_new is ignored (use under stop_gradient)
        out = backward_mega(res["q"], res["k"], res["v"], res["w"], res["b"], res["gc"], res["A"], res["Aqk"],
                            res["h_pre_all"], res["v_new_all"], res["gc_last"],
                            do.astype(jnp.float32), dhf.astype(jnp.float32), scale, cfg)
        fin = lambda x, like: sanitize(x, 1e4).astype(like.dtype)
        return (fin(out["dq"], res["q"]), fin(out["dk"], res["k"]), fin(out["dv_raw"], res["v"]),
                fin(out["dw"], res["w"]), fin(out["db"], res["b"]), fin(out["dg"], res["g"]), fin(out["dh0"], res["h0"]))

    core.defvjp(core_fwd, core_bwd)

    def call(q, k, v, w, b, g, h0):
        B, H, L, D = q.shape; nc = L // cfg.bt
        h = h0; outs, vns = [], []
        for s in range(0, nc, cfg.max_chunks_per_call):
            e = min(nc, s + cfg.max_chunks_per_call); a, c = s * cfg.bt, e * cfg.bt
            o_ch, h, vn = core(*(to_chunks_hm(t[:, :, a:c], e - s, cfg.bt) for t in (q, k, v, w, b, g)), h)
            outs.append(from_chunks_hm(o_ch)); vns.append(from_chunks_hm(vn))
        cat = lambda xs: xs[0] if len(xs) == 1 else jnp.concatenate(xs, axis=2)
        return cat(outs), h, cat(vns)

    return call


def make_atomic_backend(scale, g_max=None, vmem_mb=150.0, interpret=None, **cfg_kw):
    """Wrap atomic_gdn2. Call ensure_atomic_path() first if the repo is not installed."""
    import atomic_gdn2 as ag
    kw = dict(bt=128, score_bs=128, bs2=32, mb=8, diag="btl", dot_mode="highest",
              solve_dot_mode="bf16x3", b3_dot_mode="bf16x3", vmem_mb=vmem_mb)
    if interpret is not None:
        kw["interpret"] = interpret
    kw.update(cfg_kw)
    kcfg = ag.make_cfg(**kw)
    if g_max is not None and g_max * kcfg.bs2 / 2.0 >= 88.0:
        raise ValueError(f"g_max={g_max} too large for bs2={kcfg.bs2}: BTL half-span {g_max*kcfg.bs2/2:.1f} >= 88")
    core = ag.make_blr_trainable(kcfg, scale)
    vn_call = _make_vnew_core(kcfg, scale)

    def call(q, k, v, w, b, g, h0=None, return_vnew=False):
        f32 = lambda t: t.astype(jnp.float32)
        q, k, v, w, b, g = map(f32, (q, k, v, w, b, g))
        if return_vnew:
            B, H, L, D = q.shape
            h0_ = jnp.zeros((B, H, D, D), jnp.float32) if h0 is None else h0
            return vn_call(q, k, v, w, b, g, h0_)
        return core(q, k, v, w, b, g, h0)

    call.name = "atomic"; call.kcfg = kcfg
    return call


def make_backend(kind, d_head, **kw):
    scale = 1.0 / (d_head ** 0.5)
    if kind == "ref":
        return make_ref_backend(scale)
    if kind == "atomic":
        ensure_atomic_path(kw.pop("repo_dir", None))
        return make_atomic_backend(scale, **kw)
    raise ValueError(kind)
