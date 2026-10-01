# Запуск на Kaggle (TPU v5e)

## 0. Подготовка
1. Загрузите `gdn3_kit.zip` как **Kaggle Dataset** (или просто распакуйте в `/kaggle/working/gdn3_kit`).
2. Notebook → Settings: **Accelerator = TPU**, **Internet = ON** (нужен для `git clone atomic-ops`, `pip install optax`
   и скачивания `enwik8`).
3. Первая ячейка:
```python
!cp -r /kaggle/input/gdn3-kit/gdn3_kit /kaggle/working/ 2>/dev/null || unzip -o -q /kaggle/input/gdn3-kit/gdn3_kit.zip -d /kaggle/working
%cd /kaggle/working/gdn3_kit
!python kaggle/run_all.py --stage env
```
`env` ставит `optax`, находит `atomic_gdn2` (или клонирует `Akseleu-J/atomic-ops` в `/kaggle/working/atomic-ops`) и печатает
устройства. Если пакет лежит в нестандартном месте: `ATOMIC_GDN2_PATH=<каталог, содержащий atomic_gdn2/>`.

## 1. Порядок (каждый шаг независим, результаты в `/kaggle/working/results`, в конце `gdn3_results.zip`)

| шаг | команда | время (оценка) | зачем |
|---|---|---|---|
| тесты | `python kaggle/run_all.py --stage tests` | 5–20 мин (компиляция Pallas) | **обязательный первый шаг**: serial + pallas |
| скорость | `python kaggle/run_all.py --stage speed` | 15–40 мин | μs/токен, compile-time, decode vs число банков |
| plumbing | `python kaggle/run_all.py --stage all --quick` | 20–40 мин | проверка, что все скрипты проходят |
| ёмкость | `python kaggle/run_all.py --stage mqar --seeds 3 --steps 4000` | часы | главный результат (P*, oracle) |
| LM | `python kaggle/run_all.py --stage lm` | часы | enwik8, loss по позициям до 32k |
| бюджет | `python kaggle/run_all.py --stage archive` | 20–40 мин | fifo/merge при N < n |

Время — грубая оценка: 2 слоя, `D=128`, `bsz=8`; точное даст `bench/speed.py`. Сессия Kaggle конечна — длинные прогоны
пишут чекпойнт (`runs/*.ckpt`) и поддерживают `--resume` в `python -m gdn3.cli train`.

## 2. Тонкая настройка вручную
```bash
# одиночное обучение MQAR (с чекпойнтом и JSONL-логом)
python -m gdn3.cli train --task mqar --mode hybrid --seg 128 --L 4096 --steps 4000 --out runs
# LM на enwik8
python -m gdn3.cli train --task lm --data enwik8 --mode hybrid --seg 512 --L 8192 --bsz 4 --steps 6000 --lr 6e-4
# свои конфиги в бенчмарке: --cfg 'имя:ключ=знач,ключ=знач'
python bench/mqar_capacity.py --cfg gdn2:mode=gdn2 --cfg h1:mode=hybrid,seg=128,arch_heads=1 --cfg k4:mode=hybrid,seg=128,top_k=4
```

## 3. Если что-то сломалось
| симптом | что делать |
|---|---|
| `ImportError: atomic_gdn2` | `ATOMIC_GDN2_PATH=...` или включить Internet |
| `seg=... must be multiples of the kernel chunk` | `seg` и `warm` кратны 128 на Pallas |
| `g_max` warning | `--extra g_max=<значение ядра>` (kernel `160/bs2`, у `make_cfg` по умолчанию 5.0) |
| OOM | `--bsz` меньше, `--L` меньше, `arch_heads` меньше; `remat=True` по умолчанию |
| долгая компиляция | норма для ядра (60–140 с), больше при большом `n` (Python-циклы по банкам: держите `n ≲ 64`) |
| `SLICE_FAILURE_SW_INJECT_ERROR` | аппаратный сбой TPU — перезапустить runtime; обучение подхватится через `--resume` |
| падает `test_decode_equals_parallel` (pallas) | это допуск bf16-проходов: пришлите значения `rel`; serial-версия должна быть < 2e-4 |
| падает serial-тест | **баг логики** — пришлите traceback целиком |

## 4. Что прислать назад
`tests_serial.log`, `tests_pallas.log`, `speed.json`, `mqar_capacity.json/.md`, `lm_position.json/.md`,
`archive_policy.json`. По ним принимается решение по дереву из `DESIGN.md §12`.
