"""Uniform test-tier runner. This file is not invoked during project generation."""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
TIERS={"core": ["tests"], "cells": ["tests/test_cells.py","tests/test_reference.py","tests/test_dla.py","tests/test_router.py"], "persistence": ["tests/test_archive.py","tests/test_bundle.py","tests/test_oja_kv.py","tests/test_migration.py"], "hardware": ["tests","-m","tpu"]}

def version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--tier",choices=TIERS,default="core");p.add_argument("--output-dir",type=Path,default=Path("artifacts/test-results"));p.add_argument("--require-tpu",action="store_true");p.add_argument("pytest_args",nargs=argparse.REMAINDER);args=p.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=True);stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ");result_path=args.output_dir/f"{args.tier}-{stamp}.json"
    result=dict(schema=1,tier=args.tier,started_utc=stamp,python=platform.python_version(),versions={n:version(n) for n in ("jax","jaxlib","numpy","optax","pytest")},command=[],status="FAIL",exit_code=1,reason=None)
    if args.tier=="hardware":
        try:
            import jax
            has_tpu=any(d.platform=="tpu" for d in jax.devices())
        except Exception:
            has_tpu=False
        if not has_tpu:
            result.update(status="FAIL" if args.require_tpu else "SKIP",exit_code=2 if args.require_tpu else 0,reason="no TPU device visible")
        elif result["versions"]["pytest"] is None:
            result.update(reason="pytest is not installed")
        else:
            cmd=[sys.executable,"-m","pytest",*TIERS[args.tier],*(args.pytest_args or [])]
            result["command"]=cmd;run=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
            (args.output_dir/f"{args.tier}-{stamp}.log").write_text(run.stdout+run.stderr,encoding="utf-8")
            result.update(status="PASS" if run.returncode==0 else "FAIL",exit_code=run.returncode,reason="pytest exit code")
    elif result["versions"]["pytest"] is None:
        result.update(reason="pytest is not installed")
    else:
        cmd=[sys.executable,"-m","pytest",*TIERS[args.tier],*(args.pytest_args or [])]
        result["command"]=cmd;run=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        (args.output_dir/f"{args.tier}-{stamp}.log").write_text(run.stdout+run.stderr,encoding="utf-8")
        result.update(status="PASS" if run.returncode==0 else "FAIL",exit_code=run.returncode,reason="pytest exit code")
    result["completed_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
    result_path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False));return int(result["exit_code"])

if __name__=="__main__":raise SystemExit(main())
