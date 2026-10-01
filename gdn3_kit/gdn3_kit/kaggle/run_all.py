"""One-stop runner for Kaggle (TPU v5e).  Every stage writes into --out and is independent / resumable.

  python kaggle/run_all.py --stage env                 # install deps, locate/clone atomic_gdn2, print devices
  python kaggle/run_all.py --stage tests               # serial + pallas test-suite (~minutes)
  python kaggle/run_all.py --stage speed               # honest jit benchmark
  python kaggle/run_all.py --stage mqar --quick        # plumbing check of the capacity benchmark (~minutes)
  python kaggle/run_all.py --stage mqar                # full capacity study (hours)
  python kaggle/run_all.py --stage lm                  # enwik8 loss-by-position (needs internet for enwik8)
  python kaggle/run_all.py --stage archive             # dynamic archive under a bank budget
  python kaggle/run_all.py --stage all --quick         # everything, tiny
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def sh(cmd, env=None, log=None):
    print("\n$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    e = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    e.update(env or {})
    if log:
        with open(log, "w") as f:
            p = subprocess.Popen(cmd, cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                sys.stdout.write(line)
                f.write(line)
            p.wait()
    else:
        p = subprocess.run(cmd, cwd=ROOT, env=e)
    print(f"[done in {time.perf_counter() - t0:.0f}s, rc={p.returncode}]", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="env", choices=["env", "tests", "speed", "mqar", "lm", "archive", "all"])
    ap.add_argument("--out", default="/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--backend", default="pallas")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stages = ["env", "tests", "speed", "mqar", "lm", "archive"] if a.stage == "all" else [a.stage]
    q = ["--quick"] if a.quick else []
    rc = 0
    for s in stages:
        if s == "env":
            sh([PY, "-m", "pip", "install", "-q", "optax"])
            sh([PY, "-c", "from gdn3.env import ensure_kernel, device_summary; m=ensure_kernel(); print(m.__file__); print(device_summary())"])
        elif s == "tests":
            sh([PY, "tests/run.py", "--backend", "serial"], log=os.path.join(a.out, "tests_serial.log"))
            if a.backend == "pallas":
                rc |= sh([PY, "tests/run.py", "--backend", "pallas"], log=os.path.join(a.out, "tests_pallas.log"))
        elif s == "speed":
            sh([PY, "bench/speed.py", "--backend", a.backend, "--out", a.out,
                "--cfg", "gdn2:mode=gdn2", "--cfg", "hybrid128:mode=hybrid,seg=128",
                "--cfg", "hybrid512:mode=hybrid,seg=512", "--cfg", "local512:mode=local,seg=512",
                "--cfg", "snap512:mode=snap,seg=512", "--cfg", "hybrid512_h1:mode=hybrid,seg=512,arch_heads=1",
                "--L", "1024,2048" if a.quick else "2048,4096,8192,16384"] + (["--decode-banks", "1,8"] if a.quick else []),
               log=os.path.join(a.out, "speed.log"))
        elif s == "mqar":
            sh([PY, "bench/mqar_capacity.py", "--backend", a.backend, "--out", a.out, "--seeds", str(a.seeds),
                "--steps", str(a.steps), "--oracle",
                "--cfg", "gdn2:mode=gdn2", "--cfg", "hybrid128:mode=hybrid,seg=128",
                "--cfg", "local128:mode=local,seg=128", "--cfg", "snap128:mode=snap,seg=128"] + q,
               log=os.path.join(a.out, "mqar.log"))
        elif s == "lm":
            sh([PY, "bench/lm_position.py", "--backend", a.backend, "--out", a.out,
                "--data", "synthetic" if a.quick else "enwik8",
                "--cfg", "gdn2:mode=gdn2", "--cfg", "hybrid512:mode=hybrid,seg=512"] + q,
               log=os.path.join(a.out, "lm.log"))
        elif s == "archive":
            sh([PY, "bench/archive_policy.py", "--backend", a.backend, "--out", a.out,
                "--cfg", "hybrid:mode=hybrid,seg=128"] + q, log=os.path.join(a.out, "archive.log"))
    if os.path.isdir("/kaggle/working"):
        shutil.make_archive("/kaggle/working/gdn3_results", "zip", a.out)
        print("results zipped to /kaggle/working/gdn3_results.zip")
    sys.exit(rc)


if __name__ == "__main__":
    main()
