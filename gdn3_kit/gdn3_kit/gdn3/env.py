"""Locate (or fetch) the atomic_gdn2 kernel package. Works on Kaggle / Colab / local."""
from __future__ import annotations
import importlib
import os
import subprocess
import sys

REPO_URL = "https://github.com/Akseleu-J/atomic-ops.git"


def _roots():
    env = os.environ.get("ATOMIC_GDN2_PATH")
    if env:
        yield env
    for r in ("/kaggle/working/atomic-ops", "/content/atomic-ops",
              os.path.expanduser("~/atomic-ops"), os.path.join(os.getcwd(), "atomic-ops")):
        yield r
        yield os.path.join(r, "atomic_gdn2")


def ensure_kernel(clone: bool = True):
    """Return the imported `atomic_gdn2` module (adds it to sys.path / git-clones if needed)."""
    try:
        return importlib.import_module("atomic_gdn2")
    except ImportError:
        pass
    for c in _roots():
        if os.path.isfile(os.path.join(c, "atomic_gdn2", "__init__.py")):
            sys.path.insert(0, c)
            try:
                return importlib.import_module("atomic_gdn2")
            except ImportError:
                sys.path.remove(c)
    if clone:
        dst = "/kaggle/working/atomic-ops" if os.path.isdir("/kaggle/working") else os.path.abspath("atomic-ops")
        if not os.path.isdir(dst):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dst], check=True)
        for c in (dst, os.path.join(dst, "atomic_gdn2")):
            if os.path.isfile(os.path.join(c, "atomic_gdn2", "__init__.py")):
                sys.path.insert(0, c)
                return importlib.import_module("atomic_gdn2")
    raise ImportError("atomic_gdn2 not found. Set ATOMIC_GDN2_PATH to the directory that contains the "
                      "'atomic_gdn2' package, or enable internet to git-clone " + REPO_URL)


def device_summary() -> str:
    import jax
    d = jax.devices()
    return f"jax {jax.__version__} backend={jax.default_backend()} devices={len(d)} kind={d[0].device_kind}"
