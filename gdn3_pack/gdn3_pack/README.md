# gdn3_pack — GDN-3: динамический архив матриц-банков поверх GDN-2 (JAX, TPU v5e)

Двухпоточный слой: **main** (точный GDN-2) + **banks** (n=⌊L/C⌋ независимых состояний сегментов) с чтением по read-norm роутингу.
При γ=0 слой тождественен GDN-2. Подробности — `docs/DESIGN.md`, протокол — `docs/EXPERIMENTS.md`.

## Что проверено, а что нет (честно)
* **Запущено и проходит (numpy, CPU):** `python -m tests.test_numpy_ref` (математика слияния/разложения/онлайн-softmax) и `python -m bench.mechanism_np` (таблица в DESIGN §6).
* **Написано, НЕ запускалось** (в моей среде не было JAX): весь пакет `gdn3/`, `tests/run_all.py`, `bench/{speed,eval_mqar,mqar}.py`, `train/*`. Проверен только синтаксис (`py_compile`) и логика генератора MQAR. Первый запуск на Colab — это первый прогон: ожидайте мелких ошибок, присылайте трейс.
* Бэкенд `atomic` оборачивает ваш `atomic_gdn2` (`make_blr_trainable`), `ref` — чистый JAX-эталон. Путь с `return_vnew` (landmarks) — копия `pipeline.py`, экспериментальный.

## Структура
```
gdn3/        config, backend (ref|atomic), segments, read (роутинг), bank (BankStore/merge), landmarks (exp.),
             layer (GDN-2 слой + банки), model (LM), stream (стриминг с динамическим числом банков)
tests/       test_numpy_ref.py (без JAX), run_all.py (JAX; --atomic для Pallas)
bench/       mechanism_np.py, mqar.py, eval_mqar.py (кривая P*), speed.py
train/       train_mqar.py, train_lm.py, common.py, data.py
notebooks/   colab_gdn3.ipynb
docs/        DESIGN.md, EXPERIMENTS.md
```

## Быстрый старт (Colab TPU v5e)
Откройте `notebooks/colab_gdn3.ipynb` и идите по ячейкам, либо:
```
pip install -U "jax[tpu]" optax -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
git clone --depth 1 https://github.com/Akseleu-J/atomic-ops.git /content/atomic-ops
python -m tests.run_all --atomic
python -m train.train_mqar --model gdn3 --seg 128 --L 4096 --steps 6000 --out runs/mqar_gdn3_s0
```
Модели: `--model gdn2` (скалярный decay, как в отчёте) | `gdn2c` (поканальный decay) | `gdn3` (банки). Для сравнения нужны все три и ≥3 seed'а.

## Минимальный код
```python
from gdn3 import GDN3Config, make_backend, init_model, forward
cfg = GDN3Config(d_model=512, n_heads=4, d_head=128, n_layers=4, seg=128, bank_every=2)
be  = make_backend("atomic", cfg.d_head, g_max=cfg.g_max)      # или "ref" на CPU
params = init_model(jax.random.PRNGKey(0), cfg)
logits = forward(params, cfg, be, tokens)                        # (B, L, vocab), L % 128 == 0 для atomic
# стриминг с бюджетом банков:
from gdn3 import StreamingLM
s = StreamingLM(params, cfg, be, B=1, budget=64, policy="fenwick"); logits = s.feed(tokens_block)
```
Ограничения: `seg` кратен 128 (ядро), `g_max·32/2 < 88` (BTL), для `atomic` длина кратна 128.
