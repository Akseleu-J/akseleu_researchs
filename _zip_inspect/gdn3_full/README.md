# GDN-3 (JAX)

Gated DeltaNet-2 + пул матриц-«банков». Пакет `gdn3/`, тесты `tests/`, бенчи `bench/`, одна кнопка `kaggle/run_all.py`.

    pip install -r requirements.txt
    python tests/run.py                          # 33 интеграционных теста (serial backend, CPU ок, ~10 мин)
    python kaggle/run_all.py --quick             # smoke всех бенчей на CPU (НЕ результат)
    python kaggle/run_all.py --backend pallas    # TPU: тесты + pallas-vs-serial + M1 + M3 + speed + M2

Бенчи пишут JSON постепенно (`results/<m>/…json`), падение теряет один прогон, а не всё.
M1: `bench/mqar_capacity.py` (gdn2c / hybrid / hybrid_seg512 / hybrid_allk, 3 seed'а, критерий в `bench/_common.m1_verdict`).
ВАЖНО для M1: параметры берутся из `init_params` без `decay_boost` (в `tests/mk_params` он +2.0 — только для тестов).
