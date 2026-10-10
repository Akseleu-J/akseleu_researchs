"""A REAL Fenwick tree over the bank stream: O(log N) append and O(log N) decay-aware range reads.

Element j = (S_j (D,D), span_j (D,))  where span_j <= 0 is the total log-decay inside bank j.
Monoid (older a, newer b):  a o b = ( exp(b.span)[:,None] * a.S + b.S ,  a.span + b.span ),  identity = (0, 0)
(exactly the decay-correct superposition of archive.merge_entries, so it is associative).
Node i (1-indexed) covers banks (i - lowbit(i), i].
  append(n+1)  : node = e o-combined with nodes n, n-lowbit(n), ... down to the node boundary     -> O(log N)
  prefix(i)    : combine nodes i, i-lowbit(i), ... (older nodes on the left)                         -> O(log N)
  range(a, b]  : decayed sum of banks a+1..b seen from the END of bank b:
                   S = P_b.S - exp(P_b.span - P_a.span)[:,None] * P_a.S ,   span = P_b.span - P_a.span
                 (decay is a group, so a "minus" exists; the exponent is <= 0 => numerically stable)       -> O(log N)
Indices are static python ints (host / trace-time structure); array math is backend-agnostic (numpy or jax.numpy via xp).
NOT yet wired into the router: archive.py keeps top-k content routing over individual banks (O(N) scoring).
"""
import numpy as _np


def _low(i):
    return i & (-i)


class Fenwick:
    def __init__(self, D, xp=_np):
        self.xp, self.D = xp, D
        self.n = 0
        self.S, self.span = [None], [None]            # 1-indexed node arrays

    def _ident(self):
        return self.xp.zeros((self.D, self.D)), self.xp.zeros((self.D,))

    def combine(self, a, b):
        xp = self.xp
        return xp.exp(b[1])[:, None] * a[0] + b[0], a[1] + b[1]

    def append(self, S, span):
        """Add the newest bank. Touches O(log n) nodes."""
        i = self.n + 1
        node = (S, span)
        j, stop = i - 1, i - _low(i)
        while j > stop:                                # older sub-nodes tile (stop, i-1]
            node = self.combine((self.S[j], self.span[j]), node)
            j -= _low(j)
        self.S.append(node[0]); self.span.append(node[1]); self.n = i

    def prefix(self, i):
        """Combined (S, span) of banks 1..i (S as seen at the end of bank i)."""
        acc, j = self._ident(), i
        while j > 0:
            acc = self.combine((self.S[j], self.span[j]), acc)
            j -= _low(j)
        return acc

    def range(self, a, b):
        """Decayed sum of banks a+1..b as seen at the end of bank b (0 <= a <= b <= n)."""
        Pb, Pa = self.prefix(b), self.prefix(a)
        sp = Pb[1] - Pa[1]
        return Pb[0] - self.xp.exp(sp)[:, None] * Pa[0], sp

    def last(self, m):
        """Decayed sum of the m most recent banks (recency window read)."""
        return self.range(max(self.n - m, 0), self.n)

    def touched(self, i):
        """#nodes visited by prefix(i) (<= log2(n)+1) -- for complexity checks."""
        c, j = 0, i
        while j > 0:
            c += 1; j -= _low(j)
        return c
