#!/usr/bin/env bash
# Kaggle: Settings -> Accelerator: TPU VM v5e-8 (or single v5e), Internet: ON
set -e
pip install -q optax
[ -d /kaggle/working/atomic-ops ] || git clone --depth 1 https://github.com/Akseleu-J/atomic-ops.git /kaggle/working/atomic-ops
python - <<'PY'
import sys; sys.path.insert(0, "/kaggle/working/gdn3_kit")
from gdn3.env import ensure_kernel, device_summary
m = ensure_kernel(); print(m.__file__); print(device_summary())
PY
