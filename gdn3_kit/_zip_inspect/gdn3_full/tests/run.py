"""python tests/run.py [--only substr] [-x]"""
import argparse, importlib.util, os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("test_gdn3", os.path.join(HERE, "test_gdn3.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("-x", action="store_true")
    a = ap.parse_args()
    import jax
    print(f"jax {jax.__version__} backend={jax.default_backend()}", flush=True)
    tests = [(n, f) for n, f in vars(T).items() if n.startswith("test_") and callable(f) and a.only in n]
    ok = bad = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn(); st = "PASS"; ok += 1
        except Exception:
            st = "FAIL"; bad += 1; traceback.print_exc()
        print(f"{name:50s} {st} {time.perf_counter() - t0:7.1f}s", flush=True)
        if st == "FAIL" and a.x: break
    print(f"\npassed={ok} failed={bad}"); sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
