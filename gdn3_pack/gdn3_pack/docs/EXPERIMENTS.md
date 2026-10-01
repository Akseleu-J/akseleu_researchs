# План экспериментов (с критериями остановки)

Все сравнения: **≥3 seed'а**, одинаковые токены/шаги, три модели: `gdn2` (скалярный decay, как в отчёте), `gdn2c` (поканальный decay), `gdn3`. Порог значимости = 2–3 σ между seed'ами **gdn2c**, а не фиксированное число.

## M0 — корректность (≈30 мин)
```
python -m tests.test_numpy_ref          # математика, без JAX
python -m tests.run_all                 # ref-бэкенд: причинность, gate0==GDN-2, стриминг, динамическое N, градиенты
python -m tests.run_all --atomic        # Pallas-ядро == эталон (fwd, градиенты, h0, split), модель atomic==ref
```
Остановка: любой FAIL → дальше не идти. Особенно `causal*` (с γ≠0) и `atomic == ref`.

## M1 — скорость (≈15 мин)
`python -m bench.speed --backend atomic --L 4096 8192 --seg 128 256 512 --bank_every 2 1`
Критерий: gdn3 (bank_every=2) ≤ 1.4× gdn2 по us/tok. Если больше — bank_every=4 или меньше слоёв.

## M2 — ёмкость на MQAR (основной тест)
```
for m in gdn2 gdn2c gdn3; do for s in 0 1 2; do
  python -m train.train_mqar --model $m --seed $s --L 4096 --seg 128 --steps 6000 --out runs/mqar_${m}_s$s
done; done
```
Метрика: P\*(L) = max P с acc ≥ 0.9 (в `log.jsonl`, строка `final`). Критерии:
* P\*(gdn3) ≥ 2·P\*(gdn2c) — успех; ≥ 3× — цель.
* Если P\* не вырос: смотрим γ,τ,θ (растёт ли γ?), пробуем seg=128 при d_head=128, больше шагов, curriculum по P. Если oracle-режим (подставить истинный банк) тоже низкий — перегружен банк (C велик/факты разорваны границей).
* Контроль iso-state: gdn2c с d_head=128·√N и тот же state-размер — должен проигрывать gdn3 по P\*, иначе выигрыш только во FLOP.

## M3 — реальный LM
```
python -m train.train_lm --model gdn2c --data enwik8 --L 8192 --steps 20000 --out runs/lm_gdn2c_s0
python -m train.train_lm --model gdn3  --data enwik8 --L 8192 --steps 20000 --seg 128 --out runs/lm_gdn3_s0
```
Для BPE: `--data npy:tokens.npy --vocab <размер>`. Критерии: val(gdn3) ≤ val(gdn2c)+0.005 (не хуже) и loss по поздним позициям (`val_by_pos`) ниже. Дополнительно: sweep seg ∈ {128,256,512}, bank_every ∈ {1,2,4}.

## M4 — длинный контекст
`python -m bench.eval_mqar --ckpt runs/.../final.pkl --model gdn3 --Ls 4096 16384 65536 --Ps 64 256 1024` (параметры модели те же, что при обучении). Критерий: P\* при L≫L_train не падает катастрофически; сравнить с gdn2c. Стриминг с бюджетом: `gdn3.StreamingLM(..., budget=N, policy="fenwick")`.

## Итоговая таблица (заполнить)
| модель | params | us/tok | val (enwik8) | P\*@4k | P\*@16k | val по поздним позициям |
|---|---|---|---|---|---|---|
| gdn2 | | | | | | |
| gdn2c | | | | | | |
| gdn3 | | | | | | |
