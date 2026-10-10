# GDN-3 — полный TODO до реализации

Дата: **2026-10-09**. **Все пункты ниже — будущая работа. В текущей сессии код не менялся, новые тесты не добавлялись и не запускались.**

Основание: [полный аудит с точными местами R01–R33 и сценариями T01–T37](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_IMPLEMENTATION_AUDIT.md). Текущие исходники: [manifest](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_AUDIT_MANIFEST.json).

## 0. Зафиксировать спецификацию и цель

- [ ] Выбрать один canonical package/snapshot для последующей реализации; описать переносы kit↔full↔atomic и сохранить исходные hashes.
- [ ] Записать единую GDN-2 recurrence с раздельными write/erase адресами и gates; определить scalar-paper и generalized GDN-2 варианты.
- [ ] Определить «Inf»: непрерывный поток, ε-quality на объявленных задачах, RAM/latency budget, при необходимости растущий exact external archive. Разделить рабочую память и полное хранение истории.
- [ ] Выбрать frozen-bank либо decay-aligned semantics; определить causal read/close timing, h_init scope, merge losses, oracle privilege.
- [ ] Исправить claims lossless/O(log L)/O(1), universal D/3/128–256 и неподтверждённые speedups. Объявить реально проверяемую сложность каждого пути.
- [ ] Создать capability matrix cell×engine×memory policy×precision×Ha; назначить явные off-switches.
- [ ] Закрепить atomic/JAX/jaxlib/optax/runtime versions, device и reference/kernel presets, включая WY eps.

**Готовность:** единая проверяемая спецификация и нет скрыто игнорируемых опций.

## 1. P0: устранить дефекты, мешающие доверять результатам

- [ ] **R01:** slice archive substeps по Ha, включая все extras и preconditioner; T01.
- [ ] **R02:** index-based top-k в kit, deterministic ties, общий null и sparse/dense semantics; T02.
- [ ] **R03:** исправить QED spec и написать independent Eq.6 reference; старый two-erase вариант обозначить отдельно; T03.
- [ ] **R04:** написать paper OSDN reference, write-only scaling/hypergradient/box bounds; обосновать generalized gate adaptation; T04.
- [ ] **R05:** budget-aware prefix processing, same policy from first token; auto/capability validation; T05–06.
- [ ] **R06:** исправить capped archive evaluation kit, сохранять encoding/read budgets; constructed known-retention control; T06.
- [ ] **R20:** проверить/исправить mega dgc masks на flat g и g=0, включая h0/dh_final и interleaving; T24/T18.
- [ ] **R21:** согласовать lean clip forward/VJP и independent clipped reference; T25.
- [ ] **R22:** проверить gradient semantics sanitize/clamps, добавить pre-sanitize health и domain guards; T26.
- [ ] **R23:** correctness eps=0 и separate stabilized eps reference; T28.
- [ ] **R26/R31:** единый runner discovery/counts, отличать TPU SKIP от PASS, агрегировать все rc, создать log parent до open; T33/T36.
- [ ] **R27:** M1 INVALID/INCONCLUSIVE при нездоровом baseline, малом seed count и censored P*; paired metrics; T31.
- [ ] **R32:** полные config/capability validations, invalid ranges и negative dp_extra; T23/T29.

**Готовность:** baseline serial correctness; Pallas подтверждён отдельно на конкретной TPU версии. Не закрывать stage только наличием test files.

## 2. Непрерывный streaming и долговременная память

- [ ] **B5/R18:** определить learned init cumulative vs bank-only; одинаковый reset/forward/cache/export; T22.
- [ ] **B6/R11:** объявленные frozen/decay modes, current-token clock, causal query; T13.
- [ ] **R12:** определить merge как summary superposition или full affine composition; учесть nonzero initial offset; T14.
- [ ] **B1/R07:** выбрать token information signal и масштаб; causal boundary-before-token min/max и per-head/per-sample semantics; T07.
- [ ] **B1/R07:** trainable boundary surrogate/θ с явной областью, либо fixed threshold; проверить snap; T08.
- [ ] **B1/R07:** хранить information/density metadata, выбирать соседние low-information banks для merge; T08/T14.
- [ ] **B2/R08:** active exponential/log-linear bins с bounded slots per level; исключить O(N) historical-node retention из working-memory claim; T09.
- [ ] **B2/R08:** интегрировать hierarchy в read/router/carry/export; не создавать dense L×N scores/read; подсчитать actual visits/bytes/FLOPs.
- [ ] **B2/R08:** stable range/block read, без catastrophic prefix subtraction; T10.
- [ ] **B12/R09:** content-conditioned weights/decay для реальных levels, политика extrapolation новых уровней; T11.
- [ ] **R33:** относительные/rebased clocks, overflow-safe ages/nseg/position/HOLA counts; T13/T37.
- [ ] **R25/R33:** valid mask/no-op contract, L=0, ragged batch, independent sequence reset; T30.
- [ ] **R30/R33:** versioned inference cache serialization/resume, включая conv/h/banks/levels/KV/counters; T27/T37.
- [ ] Разделить inference memory bound и training BPTT storage; описать exact gradients/remat либо TBPTT и его approximation.

**Готовность:** one-shot/chunk/decode/resume parity + measured live-byte bound и declared lossy semantics.

## 3. Exact important-token память и retrieval

- [ ] **B8/R13:** определить committed surprise для GDN-2 и compositional cells через ||ΔS||; independent score fixtures; T15.
- [ ] **B8/R13:** top-w priority admission/eviction вместо threshold FIFO, стабильные ties и exact KV field contract; T15.
- [ ] **B8/R13:** separate cache-path RMSNorm gamma; causal/current-block/null semantics без изменения recurrent q/k; T15.
- [ ] Добавить диагностику router candidate recall, HOLA retention/hits, summary decoding errors и first-failure cause.
- [ ] Если нужен любой старый факт, описать persistent external storage/index, exact item ID/span/version, update/delete/overwrite semantics, RAM/disk/I/O growth и candidate search guarantees.
- [ ] Для external path проверить copy integrity и retrieval отдельно от генерации ответа; определить отказ при отсутствии candidate.

**Готовность:** старые важные факты сохраняются по выбранной политике; ограниченный exact cache не подписан как универсальное lossless хранение.

## 4. Модули ядра и чтения — все пункты исходного приложения

- [ ] **A2/R18:** full per-channel KDA projection и decay_a gradient; head/channel/kda режимы, invariant/halfspan; T21/T28.
- [ ] **A3/R03:** корректный QED asymmetric erase address и learned nonnegative per-head strength; production JAX/kernel + gradients; T03/T18.
- [ ] **A11/R19:** generalized EDA matrix reference, one decay, e-projection/normalization, exact γ=0 bypass; T18/T23.
- [ ] **A4/R19:** DP n_h≥1, собственные extras, decay once и read timing, n_h=1 parity; T18/T23.
- [ ] **A5/R04:** exact OSDN preconditioner path, write/read address separation, box projection, chain gradients; T04/T26.
- [ ] **B4/R10:** SSE class sparse WRITE, independent partitions, class bias/read-write gates/shared partition specification; T12.
- [ ] **C1/R14:** paper centered running key covariance, stable stats/count/mean, causal scope и sigmoid strength; T16.
- [ ] Для каждого механизма добавить feature-off parity, independent forward reference, FD/VJP continuous branches и причинность.
- [ ] Для q/k≈0 и почти совпадающих directions определить eps/convention, gradient/conditioning guards.
- [ ] Объявить cell composition order/capabilities; отдельные пары после independent validation; T29.
- [ ] Провести ablations по одной функции на том же healthy baseline, с trainable-params/state/runtime accounting.

**Готовность:** каждый принятый модуль IMPLEMENTED + REFERENCE-VALIDATED; paper/variant names соответствуют формулам. Полезность — отдельный статус.

## 5. Экономика и расширение обученной модели

- [ ] **D1/R15:** выбрать simple archive_int8 как отдельный режим или реализовать LeapQuant fixed lowbit state + highprecision window updates + compensator token factors; T17.
- [ ] **D1/R15:** сохранить low-rank residual отдельно от quant scale; window-boundary decode/cache/export semantics; measured error accumulation.
- [ ] **D1/R15:** training-free/STE контракты explicit; quant не игнорируется engine=auto; T06/T17.
- [ ] **D1/R15:** считать все live/peak bytes: h/A/Sq/T/scales/outliers/buffer/temp dequant; сравнить actual speed и recall.
- [ ] **D2/R16:** OjaKV adaptation к banks/token storage, crucial exact anchors и online low-rank basis; T19.
- [ ] **D2/R16:** reorthogonalization/basis rotation/conditioning, rank selection и rank=D fallback; quality-memory tradeoff.
- [ ] **D3/R17:** whole-checkpoint StateX migration subset layers, все projections/norms/conv/scale/router/cache; T20/T27.
- [ ] **D3/R17:** гарантировать declared initial preserve-logits, активируемые extra coords, measured parameter overhead и adaptation cost.
- [ ] **R24:** profiled Pallas kernels/fusion; reference interfaces, custom_vjp каждого Pallas path, highest/bf16x3/default; T28/T29.
- [ ] **R29:** stateful speed protocol с valid filled cache, full output carry synchronized, latency regular/close/merge/quant; T35.

**Готовность:** ускорение и сжатие подтверждены на реальной модели/памяти с declared quality loss, а не только helper.

## 6. Training, данные и честные эксперименты

- [ ] **R30:** save/restore RNG, optimizer/schedule, EMA/skips, config/schema/fingerprints; uninterrupted-vs-resume T27.
- [ ] **R28:** MQAR facts по всей истории, variable density/qstart/distance, boundary facts, overwrite tasks; T32.
- [ ] **R28:** unique queries либо separate repeated-query split; teacher-forcing policy explicit.
- [ ] **R28:** causal query-key oracle и privileged oracle labels; oracle-trained ceiling отдельно от learned checkpoint routing diagnosis.
- [ ] Healthy short positive control и long retention control прежде capacity verdict.
- [ ] Pair GDN-2c/GDN-3 learned/oracle diagnostic, одинаковые data/seeds/supervised tokens/schedule; ≥3 seeds и appropriate uncertainty.
- [ ] Iso-state, iso-params и iso-FLOP/runtime controls; не выдавать увеличение storage за бесплатный архитектурный выигрыш.
- [ ] Adequate P-grid, P* censoring/uncertainty, minimum baseline health; T31.
- [ ] Long-context suite 2K→8K→32K→128K→1M через настоящий streaming, facts/distractors/overwrite/multi-hop/passkey/needle; T34.
- [ ] LM perplexity by position и downstream recall/reasoning, отдельные task/data versions.
- [ ] Stream longevity/rebase/overflow/resume soak; T37.
- [ ] Для каждого run сохранять source/checkpoint hashes, versions/device, config/backend/dtype/eps, metrics/health/bytes/latency и pass/fail/skip logs.

**Готовность:** M1 валиден; конечные long-context результаты воспроизводимы; ε/quality и resource limits объявлены.

## 7. Обязательный тестовый backlog

- [ ] T01–T06: heads/router/budget/evaluation/engine semantics.
- [ ] T07–T14: boundaries/surrogate/hierarchy/numerics/levels/decay/merge.
- [ ] T15–T23: HOLA/CCQ/LeapQuant/cells/Oja/StateX/KDA/init/config.
- [ ] T24–T30: atomic VJP/clipping/domain/checkpoint/precision/backend/padding.
- [ ] T31–T37: verdict/data/runner/long-context/speed/bootstrap/longevity.
- [ ] Independent NumPy float64 oracle для каждой формулы; production code не переиспользуется внутри reference.
- [ ] FD continuous branches + separate hard decision/surrogate tests; не сравнивать ST с FD hard boundary.
- [ ] Outputs/final state/all gradient inputs/h0/dh_final; не ограничиваться loss/logits.
- [ ] Serial обязательный tier, TPU/Pallas explicit tier, SKIP виден; точные backend/precision tolerances.
- [ ] Property tests per-sequence causality, chronological coverage, index consistency, count/levels/counters и cache roundtrip.

## 8. Условия итогового статуса GDN3-Inf

- [ ] Один canonical snapshot и reproducible dependency/kernel environment.
- [ ] Все используемые формулы validated independent reference/VJP; off baseline parity.
- [ ] Любое допустимое разбиение потока и resume сохраняет declared semantics.
- [ ] Working memory и retrieval complexity соответствуют спецификации; внешнее хранилище учтено отдельно.
- [ ] Long finite lengths и sustained stream без unexplained norm/NaN/skip/overflow отказов.
- [ ] Paired long-context quality/capacity advantage на healthy baseline; данные и CI сохранены.
- [ ] Для exact/history цели существует достаточное растущее storage и проверенный retrieval; для summaries объявлена lossiness.
- [ ] Документация заявляет только доказанные/измеренные свойства и конкретные lengths/tasks/hardware.

**Следующий практический шаг после отдельной команды на реализацию:** S0+S1, затем streaming hierarchy и recall protection. Остальные modules принимаются по independent correctness и измеренному эффекту.
