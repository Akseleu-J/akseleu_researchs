# gdn3_kit — динамическая многоматричная память GDN-3 поверх atomic_gdn2

Пул матриц состояния `D×D`, число которых растёт с длиной контекста, **без потери качества относительно GDN-2**
(режим `hybrid` = точный GDN-2 + гейтируемое чтение top-k независимых банков).
Код — чистый JAX (без flax), ядро — `atomic_gdn2` (Pallas, TPU v5e); для тестов есть эталонный `serial`-бэкенд.

> **Статус:** не запускалось на TPU/JAX в среде разработки. Первый шаг — `python tests/run.py --backend both`.

## Структура
```
gdn3/            config, backend (serial|pallas), segments (fold/plan), routing, read (parallel), mixer, cache (stream),
                 model, tasks (MQAR/enwik8), train, evaluate, cli, env
tests/           test_gdn3.py + run.py (+ conftest.py для pytest)
bench/           mqar_capacity.py  speed.py  lm_position.py  archive_policy.py
kaggle/          run_all.py  setup.sh  gdn3_kaggle.ipynb
sim/             router_sim.py  stream_check.py  (numpy-проверки идей)  namecheck.py
docs/            DESIGN.md (дизайн, теория, протокол)  KAGGLE.md (запуск)
```

## Быстрый старт
```python
import jax
from gdn3.config import G3Config
from gdn3.backend import make_backend
from gdn3.model import init_params, forward
from gdn3.cache import prefill, decode_step

cfg = G3Config(d_model=256, n_heads=2, d_head=128, n_layers=2, mode="hybrid", seg=128)
be = make_backend("pallas", cfg.d_head)                    # или "serial" на CPU
params = init_params(jax.random.PRNGKey(0), cfg)
logits, _ = forward(params, ids, cfg, be)                  # любой L; n = ceil(L/seg) банков
logits, cache = prefill(params, ids, cfg, be, max_banks=256)       # кэш с динамическим архивом
lg, cache = decode_step(params, cache, next_token, cfg)            # O(1) шаг; архив растёт сам
```
Режимы: `gdn2 | hybrid | snap | local` (`DESIGN.md §2`). Бюджет банков при инференсе: `max_banks` и `bank_policy=fifo|merge`.
