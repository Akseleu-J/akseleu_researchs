"""Mechanism benchmark (pure numpy, CPU, seconds). NOT a trained model.
Question: with the exact delta-rule recurrence, how many key->value facts can be retrieved
  (a) from ONE continuous state (GDN-2),  (b) from N independent segment banks + read-norm routing (GDN-3 archive),
  (c) from budgeted (merged) banks?
Facts: unit random keys/values, beta=1, w=1. Queries come AFTER all writes (worst case for one state).
Run:  python -m bench.mechanism_np
"""
import numpy as np

NEG = -1e30


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def delta_scan(q, k, v, w, b, g, h0, scale=1.0):
    """Token-serial gated delta rule, one head. q,k,v,w,b,g: (L,D). Mirrors atomic_gdn2.token_serial_ref."""
    h = h0.copy(); o = np.zeros_like(q); vn_all = np.zeros_like(q)
    al = np.exp(np.minimum(g, 0.0))
    for t in range(len(q)):
        h = h * al[t][:, None]
        vn = w[t] * v[t] - (b[t] * k[t]) @ h
        h = h + np.outer(k[t], vn)
        o[t] = (q[t] * scale) @ h; vn_all[t] = vn
    return o, h, vn_all


def read_banks_np(q, S, Gtok, Gend, seg_idx, tau, theta, scale=1.0, use_decay=True):
    """Online-softmax read with a 'null' slot (value 0, logit theta). Mirrors gdn3.read.read_banks."""
    L, D = q.shape; n = S.shape[0]
    m = np.full(L, float(theta)); z = np.ones(L); acc = np.zeros((L, D))
    for j in range(n):
        qe = q * np.exp(np.minimum(Gtok - Gend[j], 0.0)) if use_decay else q
        r = (qe @ S[j]) * scale
        s = tau * np.sqrt((r * r).mean(-1) + 1e-6)
        ok = (j < seg_idx).astype(np.float64)
        s = np.where(ok > 0, s, NEG)
        mn = np.maximum(m, s); a = np.exp(m - mn); e = np.exp(s - mn) * ok
        z = z * a + e; acc = acc * a[:, None] + e[:, None] * r; m = mn
    return acc / z[:, None]


def merge_pair(Sa, gsa, Sb, gsb):
    """older a, newer b -> bank of the union span (decay-only approximation of the transition)."""
    return np.exp(gsb)[:, None] * Sa + Sb, gsa + gsb


class Store:
    def __init__(self, budget=None, policy="fenwick", per_level=2):
        self.e = []; self.budget, self.policy, self.pl = budget, policy, per_level

    def append(self, S, gs, Gend):
        self.e.append(dict(S=S, gs=gs, Gend=Gend, lvl=0)); self._compact()

    def _merge(self, i):
        a, b = self.e[i], self.e[i + 1]
        S, gs = merge_pair(a["S"], a["gs"], b["S"], b["gs"])
        self.e[i:i + 2] = [dict(S=S, gs=gs, Gend=b["Gend"], lvl=max(a["lvl"], b["lvl"]) + 1)]

    def _compact(self):
        if self.policy == "fenwick":
            ch = True
            while ch:
                ch = False
                lv = [x["lvl"] for x in self.e]
                for l in sorted(set(lv)):
                    idx = [i for i, x in enumerate(lv) if x == l]
                    if len(idx) > self.pl:
                        self._merge(idx[0]); ch = True; break
        if self.budget:
            while len(self.e) > self.budget:
                self._merge(0)

    def arrays(self):
        return (np.stack([x["S"] for x in self.e]), np.stack([x["Gend"] for x in self.e]))


def make_task(rng, P, D, Q, decay=0.0):
    keys = unit(rng.normal(size=(P, D))); vals = unit(rng.normal(size=(P, D)))
    qi = rng.integers(0, P, Q)
    L = P + Q
    q = np.zeros((L, D)); k = np.zeros((L, D)); v = np.zeros((L, D))
    w = np.zeros((L, D)); b = np.zeros((L, D))
    k[:P] = keys; v[:P] = vals; w[:P] = 1.0; b[:P] = 1.0
    q[P:] = keys[qi]
    g = -np.full((L, D), decay)
    return q, k, v, w, b, g, vals, qi


def acc_of(o, vals, qi):
    sim = unit(o + 1e-12) @ vals.T
    return float((sim.argmax(-1) == qi).mean())


def run(D=64, C=16, Ps=(32, 64, 128, 256, 512, 1024), Q=128, seeds=3, tau=60.0, theta=0.5, decay=0.0):
    rows = []
    for P in Ps:
        r = dict(main=[], routed=[], uniform=[], oracle=[], b8=[], b4=[], b2=[], fen=[], fen_n=[])
        for s in range(seeds):
            rng = np.random.default_rng(1000 * s + P)
            q, k, v, w, b, g, vals, qi = make_task(rng, P, D, Q, decay)
            L = len(q); Lp = ((L + C - 1) // C) * C
            pad = lambda x: np.concatenate([x, np.zeros((Lp - L,) + x.shape[1:])], 0)
            q, k, v, w, b, g = map(pad, (q, k, v, w, b, g))
            z0 = np.zeros((D, D))
            o_main, _, _ = delta_scan(q, k, v, w, b, g, z0)
            r["main"].append(acc_of(o_main[P:L], vals, qi))
            n = Lp // C; Gtok = np.cumsum(g, 0); seg = np.arange(Lp) // C
            S = np.zeros((n, D, D)); gs = np.zeros((n, D)); Gend = np.zeros((n, D))
            for j in range(n):
                sl = slice(j * C, (j + 1) * C)
                _, S[j], _ = delta_scan(np.zeros((C, D)), k[sl], v[sl], w[sl], b[sl], g[sl], z0)
                gs[j] = g[sl].sum(0); Gend[j] = Gtok[(j + 1) * C - 1]
            r["routed"].append(acc_of(read_banks_np(q, S, Gtok, Gend, seg, tau, theta)[P:L], vals, qi))
            oun = np.zeros((Lp, D))
            for j in range(n):
                qe = q * np.exp(np.minimum(Gtok - Gend[j], 0)); oun += (j < seg)[:, None] * (qe @ S[j])
            r["uniform"].append(acc_of(oun[P:L] / np.maximum(1, seg[P:L])[:, None], vals, qi))
            oo = np.zeros((L - P, D))
            for t, p in enumerate(qi):
                j = p // C; qe = q[P + t] * np.exp(np.minimum(Gtok[P + t] - Gend[j], 0)); oo[t] = qe @ S[j]
            r["oracle"].append(acc_of(oo, vals, qi))
            nb = P // C
            for name, kw in (("b8", dict(budget=8, policy="oldest")), ("b4", dict(budget=4, policy="oldest")),
                             ("b2", dict(budget=2, policy="oldest")), ("fen", dict(budget=None, policy="fenwick"))):
                st = Store(**kw)
                for j in range(nb):
                    st.append(S[j], gs[j], Gend[j])
                SS, GG = st.arrays()
                if name == "fen": r["fen_n"].append(len(st.e))
                o = read_banks_np(q[P:L], SS, Gtok[P:L], GG, np.full(L - P, len(st.e)), tau, theta)
                r[name].append(acc_of(o, vals, qi))
        row = {k_: float(np.mean(v_)) for k_, v_ in r.items()}; row["P"] = P
        rows.append(row)
    return rows


if __name__ == "__main__":
    D, C = 64, 16
    print(f"mechanism: D={D}, facts per bank C={C}, queries after all writes; acc = top-1 retrieval of the true value")
    print(f"{'P':>5} {'N':>4} | {'main(1 state)':>13} {'banks routed':>12} {'uniform':>8} {'oracle':>7} | "
          f"{'budget8':>7} {'budget4':>7} {'budget2':>7} {'fenwick':>7} (N_f)")
    for r in run(D, C):
        print(f"{r['P']:>5} {r['P']//C:>4} | {r['main']:>13.2f} {r['routed']:>12.2f} {r['uniform']:>8.2f} {r['oracle']:>7.2f} | "
              f"{r['b8']:>7.2f} {r['b4']:>7.2f} {r['b2']:>7.2f} {r['fen']:>7.2f} ({r['fen_n']:.0f})")
