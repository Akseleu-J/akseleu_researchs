"""Dynamic bank store for streaming / long-context evaluation.
Banks are exact while the budget allows. Beyond it, banks are MERGED:  S = diag(exp(gsum_new)) S_old + S_new.
This ignores the erase part of the transition (exact for beta=0; lossy otherwise, see tests) -> merging trades capacity
for memory. Policies: 'fenwick' (<= per_level banks per level, N ~ log n) and/or hard `budget` (merge the two oldest)."""
import jax.numpy as jnp


def merge_pair(Sa, gsa, Sb, gsb):
    """older a, newer b -> union-span bank. S: (B,H,D,D); gs: (B,H,D) total log-decay over the span."""
    return jnp.exp(gsb)[..., :, None] * Sa + Sb, gsa + gsb


class BankStore:
    def __init__(self, budget=None, policy="fenwick", per_level=2):
        assert policy in ("fenwick", "oldest", "none")
        self.entries, self.budget, self.policy, self.per_level = [], budget, policy, per_level

    def __len__(self):
        return len(self.entries)

    def append(self, S, gs, Gend):
        self.entries.append(dict(S=S, gs=gs, Gend=Gend, lvl=0)); self._compact()

    def _merge(self, i):
        a, b = self.entries[i], self.entries[i + 1]
        S, gs = merge_pair(a["S"], a["gs"], b["S"], b["gs"])
        self.entries[i:i + 2] = [dict(S=S, gs=gs, Gend=b["Gend"], lvl=max(a["lvl"], b["lvl"]) + 1)]

    def _compact(self):
        if self.policy == "fenwick":
            changed = True
            while changed:
                changed = False
                lv = [e["lvl"] for e in self.entries]
                for l in sorted(set(lv)):
                    idx = [i for i, x in enumerate(lv) if x == l]
                    if len(idx) > self.per_level:
                        self._merge(idx[0]); changed = True; break
        if self.budget:
            while len(self.entries) > self.budget:
                self._merge(0)

    def arrays(self):
        """S (B,H,n,D,D), Gend (B,H,n,D)"""
        return (jnp.stack([e["S"] for e in self.entries], 2), jnp.stack([e["Gend"] for e in self.entries], 2))
