"""Numpy-only checks of gdn3.fenwick against brute force (runs anywhere, no jax):  python tests/test_fenwick.py"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gdn3.fenwick import Fenwick


def brute(Ss, spans, a, b):
    """decayed sum of banks a+1..b (1-indexed) seen at end of bank b."""
    S, sp = np.zeros_like(Ss[0]), np.zeros_like(spans[0])
    for j in range(a, b):                              # python index j == bank j+1
        S = np.exp(spans[j])[:, None] * S + Ss[j]
        sp = sp + spans[j]
    return S, sp


def main():
    r = np.random.default_rng(0); D, n = 6, 37
    Ss = [r.standard_normal((D, D)) * 0.3 for _ in range(n)]
    spans = [-r.uniform(0, 0.6, D) for _ in range(n)]
    fw, worst = Fenwick(D), 0.0
    for t in range(n):
        fw.append(Ss[t], spans[t])
        m = fw.n
        assert fw.touched(m) <= math.log2(m) + 1 + 1e-9
        for a in {0, m // 3, m // 2, max(m - 1, 0)}:
            for b in {m, max(m - 2, a)}:
                if a <= b:
                    S, sp = fw.range(a, b); Sb, spb = brute(Ss, spans, a, b)
                    worst = max(worst, np.abs(S - Sb).max(), np.abs(sp - spb).max())
        S, _ = fw.last(5); Sb, _ = brute(Ss, spans, max(m - 5, 0), m)
        worst = max(worst, np.abs(S - Sb).max())
    # monoid associativity
    a, b, c = [(r.standard_normal((D, D)), -r.uniform(0, .5, D)) for _ in range(3)]
    l, rr = fw.combine(fw.combine(a, b), c), fw.combine(a, fw.combine(b, c))
    worst = max(worst, np.abs(l[0] - rr[0]).max(), np.abs(l[1] - rr[1]).max())
    print(f"max abs err vs brute force: {worst:.2e}")
    ok = worst < 1e-10
    print("PASS" if ok else "FAIL"); return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
