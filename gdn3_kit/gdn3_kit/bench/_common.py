import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdn3.config import G3Config, parse_overrides  # noqa: E402


def base_args(ap: argparse.ArgumentParser):
    ap.add_argument("--backend", default="pallas", choices=["pallas", "serial"])
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--d-head", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--mlp-mult", type=int, default=2)
    ap.add_argument("--cfg", action="append", default=[],
                    help="name:key=val,key=val  (e.g. 'hyb128:mode=hybrid,seg=128'); repeatable")
    ap.add_argument("--out", default="results")


def parse_cfgs(a):
    """-> list of (name, G3Config)."""
    specs = a.cfg or ["gdn2:mode=gdn2", "hybrid128:mode=hybrid,seg=128"]
    out = []
    for s in specs:
        name, _, ov = s.partition(":")
        kw = dict(vocab=256, d_model=a.d_model, n_heads=a.n_heads, d_head=a.d_head, n_layers=a.n_layers,
                  mlp_mult=a.mlp_mult)
        kw.update(parse_overrides(ov))
        out.append((name, G3Config(**kw)))
    return out


def make_be(a, cfg):
    from gdn3.backend import make_backend
    if a.backend == "serial":
        be = make_backend("serial", cfg.d_head, g_max=cfg.g_max)
    else:
        be = make_backend("pallas", cfg.d_head)
    cfg.check_backend(be.bt)
    return be


def dump(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def md_table(header, rows):
    w = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = lambda r: "| " + " | ".join(str(x).ljust(wi) for x, wi in zip(r, w)) + " |"
    return "\n".join([fmt(header), "|" + "|".join("-" * (wi + 2) for wi in w) + "|"] + [fmt(r) for r in rows])


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
