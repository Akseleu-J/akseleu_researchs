"""Shared helpers for the benches (no jax import at module level -> aggregation logic is unit-testable offline)."""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dump(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def mean_std(xs):
    xs = [float(x) for x in xs]
    n = len(xs)
    m = sum(xs) / max(n, 1)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd


def m1_verdict(pstar_hybrid, pstar_base, k=2.0, nsig=3.0):
    """M1: mean(P*(hybrid)) >= k * mean(P*(base)) and gap = mean_h - k*mean_b exceeds nsig * SE(gap).
    SE(gap) = sqrt(var_h/n_h + k^2 var_b/n_b). With zero variance (discrete P* grid, identical seeds) gap >= 0 suffices
    -- report the raw numbers, do not over-read a 3-seed verdict."""
    mh, sh = mean_std(pstar_hybrid)
    mb, sb = mean_std(pstar_base)
    gap = mh - k * mb
    se = math.sqrt(sh ** 2 / max(len(pstar_hybrid), 1) + (k * sb) ** 2 / max(len(pstar_base), 1))
    ok = (gap >= 0) and (gap > nsig * se if se > 0 else gap >= 0)
    return dict(mean_hybrid=mh, std_hybrid=sh, mean_base=mb, std_base=sb, ratio=(mh / mb if mb > 0 else float("inf")),
                gap=gap, se=se, passed=bool(ok and mh > 0))


def pick_backend(kind, d_head):
    from gdn3.backend import make_backend
    return make_backend(kind, d_head)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
