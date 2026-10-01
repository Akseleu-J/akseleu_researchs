"""Streaming (prefill / decode) with a DYNAMIC number of banks.
Tokens are fed in blocks of any size; every `cfg.seg` tokens the finished segment bank is appended to a BankStore
(unbounded = exact == batch forward; budget/fenwick = merged, bounded memory).
Not jit-compiled as a whole (python control flow over segment boundaries); jit `backend` calls internally if desired.
Landmarks are not supported in streaming yet."""
import math
import jax, jax.numpy as jnp
from .layer import layer_pre, layer_post, rmsnorm
from .read import read_banks
from .bank import BankStore


class StreamMixer:
    def __init__(self, p, cfg, backend, B, banks, budget=None, policy="none", per_level=2):
        assert not (banks and cfg.n_landmarks > 0), "landmarks unsupported in streaming"
        H, D = cfg.n_heads, cfg.d_head
        self.p, self.cfg, self.be, self.banks = p, cfg, backend, banks
        self.h = jnp.zeros((B, H, D, D), jnp.float32)
        self.cur = jnp.zeros((B, H, D, D), jnp.float32); self.cur_gs = jnp.zeros((B, H, D), jnp.float32)
        self.Gcum = jnp.zeros((B, H, D), jnp.float32); self.pos = 0
        self.store = BankStore(budget, policy, per_level)

    def feed(self, q, k, v, w, b, g):
        cfg = self.cfg; C = cfg.seg; T = q.shape[2]; outs = []; i = 0
        while i < T:
            m = min(C - self.pos % C, T - i) if self.banks else T - i
            sl = slice(i, i + m)
            qp, kp, vp, wp, bp, gp = (t[:, :, sl] for t in (q, k, v, w, b, g))
            o, self.h = self.be(qp, kp, vp, wp, bp, gp, self.h)[:2]
            Gtok = self.Gcum[:, :, None, :] + jnp.cumsum(gp.astype(jnp.float32), axis=2)
            if self.banks:
                self.cur = self.be(jnp.zeros_like(kp), kp, vp, wp, bp, gp, self.cur)[1]
                self.cur_gs = self.cur_gs + gp.sum(2)
                if len(self.store):
                    S, Gend = self.store.arrays()
                    r = read_banks(qp, S, Gtok, Gend, jnp.full((m,), S.shape[2]), jnp.exp(self.p["log_tau"]),
                                   self.p["theta"], 1.0 / math.sqrt(cfg.d_head), cfg.bank_decay, cfg.read_precision)
                    o = o + jnp.tanh(self.p["gate_raw"])[None, :, None, None] * r
            self.Gcum = Gtok[:, :, -1]; self.pos += m; i += m
            if self.banks and self.pos % C == 0:
                self.store.append(self.cur, self.cur_gs, self.Gcum)
                self.cur = jnp.zeros_like(self.cur); self.cur_gs = jnp.zeros_like(self.cur_gs)
            outs.append(o)
        return outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=2)


class StreamingLM:
    def __init__(self, params, cfg, backend, B=1, budget=None, policy="none", per_level=2):
        self.params, self.cfg = params, cfg
        self.mix = [StreamMixer(p, cfg, backend, B, cfg.has_banks(i) and "log_tau" in p, budget, policy, per_level)
                    for i, p in enumerate(params["layers"])]
        self.conv = [None] * cfg.n_layers

    def n_banks(self):
        return [len(m.store) for m in self.mix]

    def feed(self, tokens):
        x = self.params["embed"][tokens]
        for i, p in enumerate(self.params["layers"]):
            xn, (q, k, v, w, b, g), self.conv[i] = layer_pre(p, self.cfg, x, self.conv[i])
            o = self.mix[i].feed(q, k, v, w, b, g)
            x = layer_post(p, self.cfg, x, xn, o)
        return rmsnorm(x, self.params["final_norm"]) @ self.params["embed"].T
