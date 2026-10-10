"""Locate (or git-clone) the atomic_gdn2 package (Kaggle / Colab / local)."""
import importlib
import os
import subprocess
import sys

REPO_URL = "https://github.com/Akseleu-J/atomic-ops.git"


def ensure_kernel(clone: bool = True):
    try:
        return importlib.import_module("atomic_gdn2")
    except ImportError:
        pass
    cands = [os.environ.get("ATOMIC_GDN2_PATH", ""), "/kaggle/working/atomic-ops", "/content/atomic-ops",
             os.path.expanduser("~/atomic-ops"), os.path.join(os.getcwd(), "atomic-ops")]
    for c in filter(None, cands):
        for root in (c, os.path.dirname(c.rstrip("/"))):
            if os.path.isfile(os.path.join(root, "atomic_gdn2", "__init__.py")):
                sys.path.insert(0, root)
                return importlib.import_module("atomic_gdn2")
    if clone:
        dst = "/kaggle/working/atomic-ops" if os.path.isdir("/kaggle/working") else os.path.abspath("atomic-ops")
        if not os.path.isdir(dst):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dst], check=True)
        sys.path.insert(0, dst)
        return importlib.import_module("atomic_gdn2")
    raise ImportError("atomic_gdn2 not found; set ATOMIC_GDN2_PATH to the directory containing it")


def device_summary() -> str:
    import jax
    d = jax.devices()
    return f"jax {jax.__version__} backend={jax.default_backend()} devices={len(d)} kind={d[0].device_kind}"
