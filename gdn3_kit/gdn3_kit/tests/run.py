"""Plain runner (no pytest needed):  python tests/run.py --backend serial|pallas|both [--only substr] [-x]"""
import argparse
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("test_gdn3", os.path.join(HERE, "test_gdn3.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="serial", choices=["serial", "pallas", "both"])
    ap.add_argument("--only", default="")
    ap.add_argument("-x", action="store_true", help="stop at first failure")
    a = ap.parse_args()
    kinds = ["serial", "pallas"] if a.backend == "both" else [a.backend]
    import jax
    print(f"jax {jax.__version__} backend={jax.default_backend()} devices={jax.devices()}")
    tests = [(n, f) for n, f in vars(T).items() if n.startswith("test_") and callable(f) and a.only in n]
    res = dict(pass_=0, fail=0, skip=0)
    for kind in kinds:
        for name, fn in tests:
            t0 = time.perf_counter()
            try:
                fn(kind)
                status = "PASS"
                res["pass_"] += 1
            except T.Skip as e:
                status = f"SKIP ({e})"
                res["skip"] += 1
            except Exception:
                status = "FAIL"
                res["fail"] += 1
                traceback.print_exc()
            print(f"[{kind:6s}] {name:42s} {status:10s} {time.perf_counter() - t0:7.1f}s", flush=True)
            if status == "FAIL" and a.x:
                sys.exit(1)
    print(f"\npassed={res['pass_']} failed={res['fail']} skipped={res['skip']}")
    sys.exit(1 if res["fail"] else 0)


if __name__ == "__main__":
    main()
