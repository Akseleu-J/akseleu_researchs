"""One command for a Kaggle/Colab TPU session:  python kaggle/run_all.py [--quick] [--backend pallas]
Order: unit tests -> (pallas test on TPU) -> M1 -> M3 -> speed -> M2.  Each stage is a subprocess; a failure is logged and the
next stage still runs.  --quick is a CPU smoke run of the scripts, NOT a result.  Outputs: results/*.json + results/summary.json"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(name, cmd, log):
    t = time.perf_counter()
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"=== {name}: rc={rc} {time.perf_counter() - t:.0f}s (log {log})", flush=True)
    return dict(rc=rc, seconds=time.perf_counter() - t, log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--backend", default="serial")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    py, q = sys.executable, (["--quick"] if a.quick else [])
    be = ["--backend", a.backend]
    stages = [("fenwick", [py, "tests/test_fenwick.py"]), ("unit", [py, "tests/run.py"])]
    if a.backend == "pallas":
        stages.append(("pallas", [py, "tests/test_pallas.py"]))
    stages += [("m1", [py, "bench/mqar_capacity.py", "--out", f"{a.out}/m1", *be, *q]),
               ("m3", [py, "bench/archive_policy.py", "--out", f"{a.out}/m3", *be, *q]),
               ("speed", [py, "bench/speed.py", "--out", f"{a.out}/speed", *be, *q]),
               ("m2", [py, "bench/lm_position.py", "--out", f"{a.out}/m2", *be, *q])]
    summ = {n: run(n, c, os.path.join(out, f"{n}.log")) for n, c in stages}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    sys.exit(1 if any(v["rc"] for v in summ.values()) else 0)


if __name__ == "__main__":
    main()
