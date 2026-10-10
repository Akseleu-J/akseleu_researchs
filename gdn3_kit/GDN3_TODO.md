# GDN-3 v2 — полный план переделки и TODO

**Создан на шаге 2 · 2026-10-09.** Это исходный плановый backlog G001–G092. Шаг 3 начат по прямой команде пользователя; частичная реализация отражена в корневом STATUS.md и gdn3_v2/docs/CAPABILITIES.md. Галочки здесь намеренно не считать доказательством: каждый пункт закрывается по фактическому коду и evidence, а тесты/обучение/бенчмарки остаются отдельным шагом 4.

Нормативные документы: [архитектура v2](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_ARCHITECTURE_V2.md), [первоисточники](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_SOURCES.md), [точные места R01–R33](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_IMPLEMENTATION_AUDIT.md). Старый план: [TODO STEP1](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_TODO_STEP1.md).

## Как читать

- **Gxxx** — work package с зависимостями и критерием приёмки. **Rxx** — существующий finding с точными source lines. **Txx** — сценарий из аудита/дополнений ниже.
- Пути в поле «Компонент» — предлагаемые подсистемы будущего gdn3_v2, не существующие файлы. Core source ID раскрыт в реестре.
- P0: нельзя доверять baseline/результатам; P1: обязательная архитектура/интеграция; P2: optional research/отложенный механизм.
- Dependencies означают готовность contract/implementation для авторинга следующего компонента; validation закрывается только после будущего запуска. G077–G088 относятся к шагу 4.
- S8 reference work можно выполнять после G005; он не блокирует 16 core. Optional preserve migration G066 и bank Oja G063 не блокируют paper-reinit/KV integration.
- При implementation каждый G получает owner, actual file links, commit/evidence и статус. Чекбокс не закрывается только потому, что файл создан.

## S0 — фундамент и baseline

**Выход этапа:** Единый ABI, валидируемая конфигурация и независимый baseline; validation пока запланирована.

- [ ] **G001 · P0 — Создать отдельный gdn3_v2 package; сохранить donor snapshots и журнал переносов.**
  - Компонент: config/model. Зависимости: —. Источник/модуль: A1.
  - Расхождения: R01–R33. Проверки: T27/T36.
  - Приёмка: Никакого wholesale копирования full; каждый перенос связан с finding/source.

- [ ] **G002 · P0 — Закрепить совместимые JAX/jaxlib/optax/atomic commit и аппаратный preset; сохранить lock и лицензии donor code.**
  - Компонент: runtime. Зависимости: G001. Источник/модуль: A1.
  - Расхождения: R24/R31. Проверки: T28/T36.
  - Приёмка: Чистый install/import воспроизводит конкретный hash; version pins выбраны по реальной среде.

- [ ] **G003 · P0 — Реализовать конфигурацию dimensions/gates/ranges и capability report; early reject unsupported.**
  - Компонент: config/capabilities. Зависимости: G001. Источник/модуль: all.
  - Расхождения: R19/R25/R32. Проверки: T23/T29.
  - Приёмка: Некорректные Ha, m, min/max, τ, η, quant rank/window, levels отклоняются до tracing.

- [ ] **G004 · P0 — Создать общий asymmetric substep ABI с Dk≠Dv, decay once, read after last и explicit heads.**
  - Компонент: cells/base. Зависимости: G003. Источник/модуль: A1.
  - Расхождения: R01/R03/R19. Проверки: T01/T18.
  - Приёмка: Direct matrix recurrence описывает каждое поле; extras/preconditioner также выбираются по Ha.

- [ ] **G005 · P0 — Написать независимый NumPy float64 GDN-2 oracle, fixtures и gradient reference.**
  - Компонент: reference. Зависимости: G004. Источник/модуль: A1.
  - Расхождения: R20/R23/R26. Проверки: T18/T24/T28.
  - Приёмка: Эталон не импортирует production ops; outputs/h_final и все cotangents проверяемы.

- [ ] **G006 · P0 — Импортировать whole baseline checkpoint из kit/atomic через именованный adapter.**
  - Компонент: migration/atomic_import. Зависимости: G005. Источник/модуль: A1.
  - Расхождения: R17/R18/R30. Проверки: T20/T27.
  - Приёмка: Сохранены norms/scale/conv/gates/FFN; shape несовпадение не исправляется скрытым padding.

- [ ] **G007 · P0 — Ввести exact feature-off bypass и gdn2-baseline preset.**
  - Компонент: model/read. Зависимости: G006. Источник/модуль: A1.
  - Расхождения: R03/R10/R17/R19. Проверки: T18/T22/T29.
  - Приёмка: gate=0, extra off, m=1, d=1/η=0 соответствуют объявленному baseline.

- [ ] **G008 · P0 — Разделить математическую функцию, dtype, dot algorithm, WY eps, clipping и surrogate policy.**
  - Компонент: runtime/precision. Зависимости: G004. Источник/модуль: A1.
  - Расхождения: R21–R24. Проверки: T25/T28.
  - Приёмка: highest/bf16x3/default одинаково применяются всеми engines; high не переименован молча.

- [ ] **G009 · P0 — Собирать pre-sanitize NaN/Inf/norm/conditioning/skips и resource events.**
  - Компонент: runtime/health. Зависимости: G008. Источник/модуль: all.
  - Расхождения: R22/R27. Проверки: T26/T31.
  - Приёмка: Численная rescue не превращает health failure в PASS.

- [ ] **G010 · P0 — Исправить atomic decay VJP в flat g=0 и zero-decay substeps.**
  - Компонент: backends/pallas. Зависимости: G005/G008. Источник/модуль: A1.
  - Расхождения: R20. Проверки: T24/T18.
  - Приёмка: Domain one-sided derivative и h0/dh_final contributions сохранены без zero-mask cancellation.

- [ ] **G011 · P0 — Согласовать lean diagonal clipping forward/VJP и border convention.**
  - Компонент: backends/pallas. Зависимости: G008. Источник/модуль: A1.
  - Расхождения: R21. Проверки: T25.
  - Приёмка: Saturated clipped differences имеют derivative выбранной forward функции.

- [ ] **G012 · P0 — Задать WY eps=0 correctness preset и отдельный stabilized eps oracle.**
  - Компонент: runtime/solve. Зависимости: G005/G008. Источник/модуль: A1.
  - Расхождения: R23. Проверки: T28.
  - Приёмка: Formula change отделена от arithmetic error; default не маскирует mismatch.

- [ ] **G013 · P0 — Добавить domain checks прямого kernel API и asymmetric cells.**
  - Компонент: runtime/domain. Зависимости: G009/G012. Источник/модуль: A1/A3/A5.
  - Расхождения: R22/R32. Проверки: T26.
  - Приёмка: g≤0, span и address norms проверены; eigenvalue bound не выдаётся за norm contractivity.

- [ ] **G014 · P0 — Единый runner discovery/exit codes/tiers/counts и machine-readable result schema.**
  - Компонент: tests/runner. Зависимости: G001/G009. Источник/модуль: all.
  - Расхождения: R26/R31. Проверки: T33.
  - Приёмка: Любой failed stage даёт failure; no TPU означает SKIP без certification.

- [ ] **G015 · P0 — Исправить bootstrap package root/pkg discovery, pinned install и log directory.**
  - Компонент: runtime/bootstrap. Зависимости: G002/G014. Источник/модуль: A1.
  - Расхождения: R31. Проверки: T36.
  - Приёмка: Import/install/subprocess failure виден caller; logs создаются в новом output-dir.

## S1 — cell references и корректные адаптации

**Выход этапа:** Каждый основной cell имеет нормативную формулу и independent reference; composition rules явные.

- [ ] **G016 · P1 — Полная KDA gate projection: channel input signal/rate, paper и simple-projection варианты.**
  - Компонент: cells/kda. Зависимости: G004/G008. Источник/модуль: A2.
  - Расхождения: R18. Проверки: T21/T28.
  - Приёмка: Независимое изменение channel gate меняет только объявленные каналы; g≤0 и rate gradients.

- [ ] **G017 · P0 — QED: единый r=b*k+λPperp(b*q), per-head λ и near-zero projector convention.**
  - Компонент: cells/qed. Зависимости: G004/G005. Источник/модуль: A3.
  - Расхождения: R03. Проверки: T03/T18.
  - Приёмка: Vector-b counterexample проходит direct Eq.6; λ=0 bypass; никакого two-erase под paper именем.

- [ ] **G018 · P1 — EDA erase-only e projection, normalized directions и γ range.**
  - Компонент: cells/eda. Зависимости: G004/G005. Источник/модуль: A11.
  - Расхождения: R19. Проверки: T18/T23.
  - Приёмка: Erase перед main, decay once, γ=0 exact bypass, own projection VJP.

- [ ] **G019 · P1 — DeltaProduct m independent substeps с отдельными b/w/extras и scalar-paper preset.**
  - Компонент: cells/delta_product. Зависимости: G004/G005. Источник/модуль: A4.
  - Расхождения: R19. Проверки: T18/T23.
  - Приёмка: m=1 parity; порядок outputs/states/cotangents совпадает; negative m rejected.

- [ ] **G020 · P0 — OSDN paper reference: write-only d, closed hypergradient, box и update timing.**
  - Компонент: cells/osdn. Зависимости: G004/G005. Источник/модуль: A5.
  - Расхождения: R04. Проверки: T04.
  - Приёмка: При фиксированных k/β d trajectory не зависит от v/h; обе box bounds действуют.

- [ ] **G021 · P1 — Вывести generalized OSDN objective/gradient, constraints и signed-address feasibility.**
  - Компонент: cells/osdn/docs. Зависимости: G017/G020. Источник/модуль: A5/A3.
  - Расхождения: R04/R22. Проверки: T04/T41.
  - Приёмка: Формула §5 реализована как project variant; r≈0/residual=0 hold, unsafe composition reject.

- [ ] **G022 · P1 — Единый composition builder EDA→DP slots→QED/OSDN adapter, gates/shapes и off switches.**
  - Компонент: cells/composition. Зависимости: G016/G017/G018/G019/G021. Источник/модуль: A2/A3/A11/A4/A5.
  - Расхождения: R01/R19/R32. Проверки: T18/T29/T41.
  - Приёмка: Нет двойного decay; каждой комбинации назначен validated/research/rejected статус.

- [ ] **G023 · P1 — Авторинг independent forward/FD/VJP fixtures всех core cells и trainable params.**
  - Компонент: reference/tests/cells. Зависимости: G022. Источник/модуль: A2/A3/A11/A4/A5.
  - Расхождения: R03/R04/R18/R19/R26. Проверки: T03/T04/T18/T21.
  - Приёмка: Signal fixtures проверяют не только finite, но и ненулевые continuous derivatives.

## S2 — streaming transaction и банк

**Выход этапа:** Prefill/decode/chunking используют один budget/timing; valid/reset и complete carry работают одинаково.

- [ ] **G024 · P0 — Typed LayerCarry/ModelCarry: все module fields, metadata, ownership и dtype.**
  - Компонент: stream/carry. Зависимости: G004/G003. Источник/модуль: all.
  - Расхождения: R01/R25/R30/R33. Проверки: T01/T27/T30.
  - Приёмка: Нет скрытых runtime caches вне serialized carry; Ha/Dk/Dv/partition axes объявлены.

- [ ] **G025 · P0 — Реализовать token transaction §6 с tentative/committed update и boundary-before-token.**
  - Компонент: stream. Зависимости: G022/G024. Источник/модуль: A1/B1.
  - Расхождения: R05/R07/R25. Проверки: T05/T07/T18.
  - Приёмка: Отвергнутый tentative update не меняет d/quant/HOLA; current clock учитывается один раз.

- [ ] **G026 · P0 — Budget-aware prefill от первого token; chunked и parallel только при same semantics.**
  - Компонент: model/prefill. Зависимости: G025. Источник/модуль: B1/B2.
  - Расхождения: R05. Проверки: T05/T06.
  - Приёмка: Никакого uncapped prefix с последующей cap; multilayer logits/carry эквивалентны.

- [ ] **G027 · P0 — No-op mask, L=0, arbitrary alignment, ragged batches и per-sequence reset.**
  - Компонент: model/stream. Зависимости: G025. Источник/модуль: all.
  - Расхождения: R25/R33. Проверки: T30.
  - Приёмка: Padding сохраняет conv/state/counters/CCQ/windows/external cursor; samples изолированы.

- [ ] **G028 · P1 — Learned bank h_init reset/gradients; cumulative/snap scopes раздельно.**
  - Компонент: memory/bank. Зависимости: G025. Источник/модуль: B5.
  - Расхождения: R18/R12. Проверки: T22/T14.
  - Приёмка: Bank-only default сохраняет cumulative baseline; каждый reset использует init один раз.

- [ ] **G029 · P1 — Frozen и passive-aligned B6 modes с current-token clock и merge time alignment.**
  - Компонент: memory/bank/read. Зависимости: G025/G028. Источник/модуль: B6.
  - Расхождения: R11/R12. Проверки: T13/T14.
  - Приёмка: Decay-only не называется full delta continuation; read clock причинен.

- [ ] **G030 · P1 — Overflow-safe absolute IDs, relative clocks, multiword counters и rebase.**
  - Компонент: stream/clocks. Зависимости: G029. Источник/модуль: B6.
  - Расхождения: R33. Проверки: T13/T37.
  - Приёмка: Near-int32 limits не ломают validity/order; rebase сохраняет differences.

- [ ] **G031 · P1 — gdn2/hybrid/local/snap routing semantics и warm-only-local.**
  - Компонент: model/modes. Зависимости: G026/G028. Источник/модуль: A1/B5.
  - Расхождения: R07/R18/R25. Проверки: T05/T08/T22.
  - Приёмка: Snap overlapping snapshots не суммируются как independent spans; unsupported combination rejected.

## S3 — DLA, hierarchy, router и query cleaning

**Выход этапа:** Live recent/hierarchy bounds и causal coverage существуют в настоящем carry; summary loss явно названа.

- [ ] **G032 · P1 — Независимый DLA additive paper reference и normalized state information Eq.13.**
  - Компонент: reference/dla. Зависимости: G005. Источник/модуль: B1.
  - Расхождения: R07. Проверки: T07/T44.
  - Приёмка: Код не подменяет score key distance; двусмысленные paper обозначения зафиксированы.

- [ ] **G033 · P1 — GDN-2 tentative-state score, per-sample/head aggregation, min/max и info metadata.**
  - Компонент: memory/dla. Зависимости: G025/G032. Источник/модуль: B1.
  - Расхождения: R07. Проверки: T07/T44.
  - Приёмка: Boundary-before-token и n/I sums reproducible; forced/forbidden зоны правильны.

- [ ] **G034 · P1 — Soft-train/hard-inference и optional ST estimator; отдельная mask/surrogate спецификация.**
  - Компонент: memory/dla/train. Зависимости: G033. Источник/модуль: B1.
  - Расхождения: R07. Проверки: T08/T44.
  - Приёмка: θ получает declared gradient; ST не сравнивается с FD hard threshold; train/decode gap логируется.

- [ ] **G035 · P1 — Recent capacity и adjacent min-density merge; age-based promotion independent от count merge.**
  - Компонент: memory/dla/merge. Зависимости: G033/G028. Источник/модуль: B1.
  - Расхождения: R07/R12. Проверки: T08/T14.
  - Приёмка: Least-density соседняя пара освобождает count slot; старые spans всё равно продвигаются.

- [ ] **G036 · P1 — Binary live bins по promotion units, spans/occupancy/IDs и единичное владение.**
  - Компонент: memory/hierarchy. Зависимости: G035. Источник/модуль: B2.
  - Расхождения: R08/R33. Проверки: T09/T45.
  - Приёмка: ≤1 bin/level, нет retained children/all-history node array; coverage без gaps/duplicates.

- [ ] **G037 · P1 — Dynamic level growth с geometric allocation, copy/recompile events и explicit bounded overflow.**
  - Компонент: memory/hierarchy/allocator. Зависимости: G036/G030. Источник/модуль: B2/B12.
  - Расхождения: R09/R33. Проверки: T37/T42.
  - Приёмка: Unbounded mode создаёт новый level; fixed max не скрывает drop/clip.

- [ ] **G038 · P1 — Mean-summary merge и descriptor recompute, initialization/decay semantics.**
  - Компонент: memory/merge. Зависимости: G029/G036. Источник/модуль: B2/B5/B6.
  - Расхождения: R12. Проверки: T14/T10.
  - Приёмка: Count-weighted summary названа lossy; stale child index невозможен.

- [ ] **G039 · P1 — Exact affine (P,U) oracle, nonzero-init correction и cost accounting.**
  - Компонент: reference/affine. Зависимости: G005/G029. Источник/модуль: A1/B2/B5.
  - Расхождения: R12. Проверки: T14.
  - Приёмка: Noncommuting erase fixture воспроизводит matrix composition, не только decay.

- [ ] **G040 · P1 — Paper Log-Linear reference и production GDN-2 stream adapter: reset/bin/transition derivation.**
  - Компонент: memory/loglinear. Зависимости: G004/G039. Источник/модуль: B2.
  - Расхождения: R08. Проверки: T09/T10/T45.
  - Приёмка: Production config/carry/read подключены; separate loglinear-paper vs bank-hierarchy names; no O(T) historical-tree claim.

- [ ] **G041 · P0 — Deterministic index top-k/null, candidate descriptors и shared dense/sparse read.**
  - Компонент: memory/router. Зависимости: G036/G038. Источник/модуль: B2.
  - Расхождения: R02/R10. Проверки: T02/T06.
  - Приёмка: Ties ≤k valid selected, probability mass одинаковая; empty archive returns null.

- [ ] **G042 · P1 — Paper Adaptive λ Eq.4 и отдельный GELU variant; level mapping/init.**
  - Компонент: memory/router/lambda. Зависимости: G040. Источник/модуль: B12.
  - Расхождения: R09. Проверки: T11.
  - Приёмка: Per-head/per-level weights не используются как age penalty; λ≈1 init.

- [ ] **G043 · P1 — Shared per-level MLP для новых levels, feature domain и combined p_j λ read.**
  - Компонент: memory/router/lambda. Зависимости: G037/G041/G042. Источник/модуль: B12.
  - Расхождения: R09/R33. Проверки: T11/T42.
  - Приёмка: Нет fixed trained-bucket clipping; extrapolation quality не объявлена заранее.

- [ ] **G044 · P1 — CCQ Welford centered key statistics, current-key inclusion и long count arithmetic.**
  - Компонент: memory/ccq. Зависимости: G024/G027. Источник/модуль: C1.
  - Расхождения: R14/R33. Проверки: T16/T37.
  - Приёмка: Σ=M2/n по raw main k; identical keys→0; nondecayed stats и count precision объявлены.

- [ ] **G045 · P1 — CCQ paper-read и archive-only adapter, raw/write separation и clean→decay order.**
  - Компонент: memory/read. Зависимости: G044/G029/G041. Источник/модуль: C1/B6.
  - Расхождения: R14/R11. Проверки: T16/T41.
  - Приёмка: QED writes use raw q; λ=0/off preserves scope baseline; covariance conditioning видно.

- [ ] **G046 · P1 — Считать actual live/allocated/peak bytes, visits и merge amortization.**
  - Компонент: memory/accounting. Зависимости: G037/G043/G045. Источник/модуль: B2.
  - Расхождения: R08/R15/R29/R33. Проверки: T09/T35/T42.
  - Приёмка: В byte totals включены float copies, sketches, stats, scratch и capacity slack.

## S4 — expanded writes, priority KV и внешний архив

**Выход этапа:** Дополнительная ёмкость реальна; exact storage, selected memory и answer quality имеют разные метрики.

- [ ] **G047 · P1 — SSE paper sparse-class-write и expanded-partition references.**
  - Компонент: reference/sse. Зависимости: G005. Источник/модуль: B4.
  - Расхождения: R10. Проверки: T12.
  - Приёмка: Независимые N states, top-k selection/soft rows/shared partition/gates/balance описаны.

- [ ] **G048 · P1 — SSE archive-bank adapter: inactive freeze default; optional lazy passive-decay timestamps.**
  - Компонент: memory/sse. Зависимости: G024/G047/G029. Источник/модуль: B4.
  - Расхождения: R10/R33. Проверки: T12/T37.
  - Приёмка: Paper freeze и passive variant различаются; no erase/write inactive, capacity/update costs честные.

- [ ] **G049 · P1 — Вывести SSE normalization/projector/OSDN constraints; оформить composition matrix.**
  - Компонент: memory/sse/composition. Зависимости: G048/G022. Источник/модуль: B4/A3/A5.
  - Расхождения: R10/R32. Проверки: T41/T29.
  - Приёмка: Sparsification не молча нарушает unit-k/QED projection; invalid комбинации early reject.

- [ ] **G050 · P1 — HOLA scalar-paper reference и generalized committed write surprise.**
  - Компонент: reference/hola. Зависимости: G005/G022. Источник/модуль: B8.
  - Расхождения: R13. Проверки: T15.
  - Приёмка: No repeated β on weighted residual; score не включает rejected tentative path.

- [ ] **G051 · P1 — Top-w priority retention, ties/IDs, exact raw payload и bounded causal block.**
  - Компонент: memory/hola. Зависимости: G025/G050. Источник/модуль: B8.
  - Расхождения: R13/R33. Проверки: T15/T37.
  - Приёмка: Old high-surprise переживает >w moderate tokens; duplicate ID не удваивает read.

- [ ] **G052 · P1 — Cache-path learned RMSNorm, causal current-block/null и read integration.**
  - Компонент: memory/hola/read. Зависимости: G051/G041. Источник/модуль: B8.
  - Расхождения: R13. Проверки: T15/T41.
  - Приёмка: Norm отделён от recurrent q/k; bank close не очищает global priority memory.

- [ ] **G053 · P1 — Persistent schema: original spans/hash/version, causal IDs и durable transaction cursor.**
  - Компонент: persistence/archive. Зависимости: G024/G030. Источник/модуль: external.
  - Расхождения: R33. Проверки: T43/T27.
  - Приёмка: Точное хранение не зависит от quant/Oja; append failure не создаёт false durable cursor.

- [ ] **G054 · P1 — Disk index и bounded candidate cache: exact ID, semantic candidate и diagnostic scan modes.**
  - Компонент: persistence/index. Зависимости: G053. Источник/модуль: external.
  - Расхождения: R28/R29. Проверки: T43/T35.
  - Приёмка: Gold ID отсутствует в learned query; disk/index/I/O accounting; no perfect-recall promise.

- [ ] **G055 · P1 — Versioned overwrite/latest-visible, resume reconciliation, tombstones и read-only provenance.**
  - Компонент: persistence/index/archive. Зависимости: G054. Источник/модуль: external.
  - Расхождения: R28/R30. Проверки: T43/T32/T27.
  - Приёмка: Будущая версия факта не видна старому query; replay не дублирует append.

- [ ] **G056 · P1 — Retrieval recall@k, HOLA retention/hits и summary failure attribution.**
  - Компонент: bench/retrieval. Зависимости: G052/G055. Источник/модуль: B8/external.
  - Расхождения: R28. Проверки: T34/T43.
  - Приёмка: Miss разделён на storage/selection/candidate/read/generation; oracle privileged явно подписан.

## S5 — compression и StateX

**Выход этапа:** Сжатие меняет реальный carry/storage; StateX — whole-model migration и post-training контракт.

- [ ] **G057 · P1 — LeapQuant paper window reference, residual subtraction, smoothing и compensator factors.**
  - Компонент: reference/leapquant. Зависимости: G005. Источник/модуль: D1.
  - Расхождения: R15. Проверки: T17.
  - Приёмка: Boundary state+FP buffer воспроизводит reference; simple archive_int8 другое имя.

- [ ] **G058 · P1 — Вывести GDN-2/substep window Γ и factors, OSDN/SSE conditions.**
  - Компонент: compression/leapquant/math. Зависимости: G022/G049/G057. Источник/модуль: D1.
  - Расхождения: R15/R24. Проверки: T17/T41.
  - Приёмка: Преобразованный buffer сравним с unquantized recurrence; unsupported cell явно обозначена.

- [ ] **G059 · P1 — LeapQuant inference integration: arbitrary close flush, merge/refit, no-op, serialize.**
  - Компонент: compression/leapquant. Зависимости: G058/G038/G027. Источник/модуль: D1.
  - Расхождения: R15/R30. Проверки: T17/T27.
  - Приёмка: Partial window resume/close правильны; train quant off; loss/latency измеряемы.

- [ ] **G060 · P1 — Lowbit/factors/window/outliers/scales/temp live/peak accounting и error diagnostics.**
  - Компонент: compression/accounting. Зависимости: G059/G046. Источник/модуль: D1.
  - Расхождения: R15/R29. Проверки: T17/T35.
  - Приёмка: Нет hidden full float state, уничтожающего заявленное compression ratio.

- [ ] **G061 · P1 — OjaKV paper reference: anchors, separate K/V basis, rank/update/orthogonalization.**
  - Компонент: reference/oja_kv. Зависимости: G005. Источник/модуль: D2.
  - Расхождения: R16. Проверки: T19.
  - Приёмка: Rotating-subspace fixture и rank=D fallback; полный payload anchors не изменён.

- [ ] **G062 · P1 — Oja integration в working retrieval KV cache, basis version/remap и old-code consistency.**
  - Компонент: compression/oja_kv. Зависимости: G054/G061. Источник/модуль: D2.
  - Расхождения: R16/R33. Проверки: T19/T27.
  - Приёмка: HOLA/original spans exact; смена basis не переинтерпретирует старые codes.

- [ ] **G063 · P2 — Bank Oja adaptation отдельно: representation/read derivation и quality/memory protocol.**
  - Компонент: compression/oja_banks/docs. Зависимости: G038/G061. Источник/модуль: D2.
  - Расхождения: R16. Проверки: T19/T41.
  - Приёмка: Research variant не выдаётся за paper KV compressor; factorization error измерим.

- [ ] **G064 · P1 — StateX paper-reinit/head-merging reference и GDN-2 gate/norm/scale derivation.**
  - Компонент: migration/statex/math. Зависимости: G006/G022. Источник/модуль: D3.
  - Расхождения: R17. Проверки: T20/T46.
  - Приёмка: Initial parity paper mode не требуется; доступ cross-head blocks и adaptation budget определены.

- [ ] **G065 · P1 — Whole-checkpoint selected-layer migration: projections/conv/output/router/SSE/optimizer/cache/schema.**
  - Компонент: migration/statex. Зависимости: G064/G048/G024. Источник/модуль: D3.
  - Расхождения: R17/R30. Проверки: T20/T27/T46.
  - Приёмка: Все shapes обновлены; partial helper неприемлем; новый state dimension активируется.

- [ ] **G066 · P2 — Optional preserve-compatible migration с доказанным block/norm transformation.**
  - Компонент: migration/statex/compat. Зависимости: G065. Источник/модуль: D3.
  - Расхождения: R17. Проверки: T20/T46.
  - Приёмка: Паритет обещается только для supported transform; unknown checkpoint rejected.

- [ ] **G067 · P1 — StateX post-training preset/schedule/data/accounting и extra-coordinate gradients.**
  - Компонент: configs/statex/bench. Зависимости: G065. Источник/модуль: D3.
  - Расхождения: R17/R28. Проверки: T20/T34/T46.
  - Приёмка: Added state/params/adaptation tokens учтены; quality не заявляется после одного pad.

## S6 — persistence, production backends и авторинг тестов

**Выход этапа:** Архитектура подключена целиком; correctness ещё требует отдельного разрешённого запуска.

- [ ] **G068 · P1 — Versioned inference cache roundtrip всего carry и dynamic allocated levels.**
  - Компонент: persistence/cache. Зависимости: G059/G062/G065/G055. Источник/модуль: all.
  - Расхождения: R30/R33. Проверки: T27/T37/T42.
  - Приёмка: Quant windows/basis/priority/conv/IDs/durable cursor сохраняются и validate-ятся.

- [ ] **G069 · P1 — Training checkpoint RNG/optimizer/scheduler/data cursor/EMA/skips и config hashes.**
  - Компонент: persistence/checkpoint. Зависимости: G068/G009. Источник/модуль: all.
  - Расхождения: R30. Проверки: T27.
  - Приёмка: Resume контракты continuous vs TBPTT объявлены; mismatch не загружается молча.

- [ ] **G070 · P1 — Full-BPTT/remat vs TBPTT training modes и DLA soft/hard state policy.**
  - Компонент: train. Зависимости: G069/G034. Источник/модуль: all.
  - Расхождения: R07/R30/R33. Проверки: T08/T27.
  - Приёмка: Inference bounds не приписываются training activation history; truncation loss обозначена.

- [ ] **G071 · P1 — JAX serial/chunked implementations по ABI; same budget/precision в auto.**
  - Компонент: backends/jax. Зависимости: G026/G043/G049/G052/G059. Источник/модуль: all.
  - Расхождения: R05/R24. Проверки: T05/T28/T29.
  - Приёмка: Unsupported пара не игнорируется; effective backend и fallback reason записаны.

- [ ] **G072 · P1 — Pallas core kernels + custom VJP для KDA/QED/EDA/DP/OSDN adapters.**
  - Компонент: backends/pallas/cells. Зависимости: G010/G011/G013/G023/G071. Источник/модуль: A2/A3/A11/A4/A5.
  - Расхождения: R20/R21/R24. Проверки: T18/T24/T25/T29.
  - Приёмка: Forward/state/VJP contract каждого primitive; аппаратный PASS только позже.

- [ ] **G073 · P1 — Pallas memory/read primitives: router/CCQ/SSE/HOLA/hierarchy updates и λ.**
  - Компонент: backends/pallas/memory. Зависимости: G072/G045/G049/G052. Источник/модуль: B1/B2/B4/B8/B12/C1.
  - Расхождения: R24. Проверки: T02/T09/T12/T15/T16/T29.
  - Приёмка: Kernel или explicit fallback по каждой операции; fusion сохраняет event order.

- [ ] **G074 · P1 — Pallas quant/compensator/Oja primitives и backend checkpoint conversion.**
  - Компонент: backends/pallas/compression. Зависимости: G073/G060/G062. Источник/модуль: D1/D2.
  - Расхождения: R24/R15/R16. Проверки: T17/T19/T29.
  - Приёмка: Cast/STE semantics именованы; fallback не считается hardware certification.

- [ ] **G075 · P1 — Авторинг T01–T37/T41–T48, shape/precision/backend regression fixtures; T38–T40 вместе с deferred references.**
  - Компонент: tests/reference. Зависимости: G074/G070. Источник/модуль: all.
  - Расхождения: R01–R33. Проверки: T01–T48.
  - Приёмка: Independent oracle, exact off, h0/dh_final, nonzero gradient, property checks; не запускать на шаге 3 без команды.

- [ ] **G076 · P1 — Полная capability matrix и документы module status/formula/ABI/source.**
  - Компонент: capabilities/docs. Зависимости: G075/G065. Источник/модуль: all.
  - Расхождения: R24/R32. Проверки: T29/T41/T48.
  - Приёмка: Все 16 core имеют production path/carry/checkpoint; rejected composition сопровождается выводом.

## S7 — будущий запуск, качество и ресурсы

**Выход этапа:** Эти пункты выполняются только на шаге 4 по отдельной команде; presence test files не равно PASS.

- [ ] **G077 · P0 — Запустить independent baseline/cell/atomic correctness tiers и исправить все failures.**
  - Компонент: tests. Зависимости: G075/G014. Источник/модуль: all.
  - Расхождения: R01–R26. Проверки: T01–T30/T41–T42/T44–T46.
  - Приёмка: Actual outputs/states/gradients в объявленных tolerances; raw logs сохранены.

- [ ] **G078 · P0 — Запустить backend/precision tiers на реальном TPU; записать skip отдельно.**
  - Компонент: tests/pallas. Зависимости: G077/G002. Источник/модуль: all.
  - Расхождения: R20–R26. Проверки: T24/T25/T28/T29/T33.
  - Приёмка: No hardware = SKIP; M0 Pallas не закрыт без аппаратных результатов.

- [ ] **G079 · P1 — Stream longevity/growing levels/reset/chunk/resume + external integrity tests.**
  - Компонент: tests/stream. Зависимости: G077/G068. Источник/модуль: all.
  - Расхождения: R05/R25/R30/R33. Проверки: T05/T27/T30/T37/T42/T43.
  - Приёмка: Semantic continuity и byte bound выполняются; clocks/counters near limits проверены.

- [ ] **G080 · P0 — Исправить M1 verdict: paired seeds, baseline health/control, censoring, invalid states.**
  - Компонент: bench/metrics. Зависимости: G009/G014. Источник/модуль: A1/B2.
  - Расхождения: R27. Проверки: T31.
  - Приёмка: Zero baseline/one seed/oracle-only не PASS; capacity P* uncertainty сохранена.

- [ ] **G081 · P1 — Новые MQAR генераторы: distributed facts, boundaries, overwrite, unique/repeated query splits.**
  - Компонент: tasks/mqar. Зависимости: G080. Источник/модуль: A1/B2/B8.
  - Расхождения: R28. Проверки: T32.
  - Приёмка: Gold targets и masks independent; filler-only L growth не изображает увеличение capacity.

- [ ] **G082 · P1 — Разделить causal learned/router oracle-trained/privileged diagnostic paths.**
  - Компонент: evaluate. Зависимости: G081/G056/G026. Источник/модуль: B2/B8/external.
  - Расхождения: R06/R28. Проверки: T06/T32/T47.
  - Приёмка: Encoding/query budgets одинаковы; gold-bank-at-SEP маркирован privileged.

- [ ] **G083 · P1 — Stateful benchmark с full carry, valid filled banks/metadata и device sync.**
  - Компонент: bench/speed. Зависимости: G079/G074. Источник/модуль: all.
  - Расхождения: R29. Проверки: T35.
  - Приёмка: DCE не удаляет update; latency close/merge/quant/resize и compile учтена отдельно.

- [ ] **G084 · P1 — Short controls и paired one-module ablations на том же healthy checkpoint/data.**
  - Компонент: bench/ablation. Зависимости: G078/G080/G082. Источник/модуль: core.
  - Расхождения: R27/R28. Проверки: T31/T34.
  - Приёмка: Полезность модуля отделена от correctness; iso-state/params/runtime accounting.

- [ ] **G085 · P1 — Long-context suite 2K→1M и sustained streaming: density/distance/distractors/overwrite/multi-hop.**
  - Компонент: bench/long_context. Зависимости: G084/G079. Источник/модуль: core.
  - Расхождения: R28/R33. Проверки: T34/T37.
  - Приёмка: Predeclared ε/quality floors и per-position LM metrics; finite lengths честно указаны.

- [ ] **G086 · P1 — Полная composition после individual ablations: stability/quality/speed и interactions.**
  - Компонент: bench/composition. Зависимости: G085/G076. Источник/модуль: all.
  - Расхождения: R22/R32. Проверки: T41/T34.
  - Приёмка: Нет произведения speedups статей; unsafe combination не default.

- [ ] **G087 · P1 — LeapQuant/Oja/StateX tradeoffs на обученном checkpoint с migration/adaptation затратами.**
  - Компонент: bench/compression_statex. Зависимости: G086/G067. Источник/модуль: D1/D2/D3.
  - Расхождения: R15/R16/R17. Проверки: T17/T19/T20/T34/T35.
  - Приёмка: Actual total/peak bytes, error и recall, не helper-only числа.

- [ ] **G088 · P1 — Final reproducible report/spec coverage и обновление STATUS по evidence.**
  - Компонент: docs/status. Зависимости: G087/G082/G083. Источник/модуль: all.
  - Расхождения: R01–R33. Проверки: T48.
  - Приёмка: IMPLEMENTED/VALIDATED/QUALITY различаются; только actual results меняют M0/M1 status.

## S8 — отложенные A6/A7/A8

**Выход этапа:** Формулы и references предусмотрены; production incorporation не требуется для завершения 16 core.

- [ ] **G089 · P2 — FG²-GDN symmetric-paper reference, fine-grained β/decay и dimension restrictions.**
  - Компонент: reference/fg2. Зависимости: G005. Источник/модуль: A6.
  - Расхождения: new attachment. Проверки: T38.
  - Приёмка: Eq.11/βk-βv варианты отдельны от asymmetric GDN-2; off limit описан.

- [ ] **G090 · P2 — Q-Delta Eq.17–20 reference: scalar β, λ range, outer-product write и α timing.**
  - Компонент: reference/qdelta. Зависимости: G005. Источник/модуль: A7.
  - Расхождения: new attachment. Проверки: T39.
  - Приёмка: Missing outer-product в приложении исправлен; QED и Q-Delta не conflated.

- [ ] **G091 · P2 — Preconditioned DeltaNet references: exact SM, diagonal normalized Eq.8 и practical squashed Eq.9.**
  - Компонент: reference/preconditioned. Зависимости: G005. Источник/модуль: A8.
  - Расхождения: new attachment. Проверки: T40.
  - Приёмка: Рекуррентные αP/βP, positive A, μ/x и gradients проверены; exact inverse не conflated с practical paper.

- [ ] **G092 · P2 — Документировать adapters/production opt-in и отдельные experiments для deferred cells.**
  - Компонент: cells/deferred/docs. Зависимости: G089/G090/G091/G076. Источник/модуль: A6/A7/A8.
  - Расхождения: R24/R32. Проверки: T38–T41/T29.
  - Приёмка: Не включать автоматически в default; canonical schema допускает family-specific carry.

## Тестовый план T01–T48

**Существующие T01–T37:** полный набор входов/ожиданий сохранён в [разделе 7 аудита](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_IMPLEMENTATION_AUDIT.md). Он является частью этого TODO, а не уже выполненной проверкой.

### Уточнения контрактов исходных тестов

- T04: independence d от h/v относится к scalar-paper OSDN. Generalized adapter проверяется по собственному objective/hold convention.
- T08: для bank-hierarchy проверять DLA info merge recent queue; binary promotions имеют другой выбор и не обязаны быть minimum-density.
- T09: log bound по promotion count; token span и leaf units различаются. Нет требования сохранения individual facts после lossy merge.
- T10: whole-summary/range contracts явные; merged partial span не называется exact. Stable direct-block oracle без prefix subtraction.
- T12: inactive partition может passive-decay по объявленному SSE contract; erase/write запрещены. N=1 paper не обязательно GDN-2; exact bypass отдельный.
- T14: mean-summary сравнивается со своим reference, affine — с direct union; nonzero init correction только exact mode.
- T15: generalized surprise = сумма норм committed write deltas, а не включая decay/erase-only physical Δ. Raw payload+cache-path norms и duplicated IDs проверяются.
- T16: denominator n, normalized main keys до текущего включительно; no decayed stats.
- T17: windows/low-rank compensators и arbitrary close; simple archive_int8 другой режим.
- **T20 заменён уточнённым T46:** initial parity обязательна только preserve-compatible migration, не paper-reinit StateX.
- T24: derivative g=0 проверяется со стороны разрешённого domain и через g=0 interleaving.
- T34/T37: конечные длины и soak подтверждают объявленный range/quality; не универсальную lossless гарантию.

### Дополнительные сценарии

T38–T40 относятся к deferred cells: обязательны при их opt-in реализации, не блокируют 16 core. Всего 48 сценариев в полном backlog, 45 для основной архитектуры.

| ID | Область | Входы и ожидаемый результат |
|---|---|---|
| T38 | FG²-GDN | Direct symmetric Eq.11, β channel perturbation, βk/βv shape restrictions, zero/one gates и continuous VJP; сравнение с GDN-2 только при выведенном общем limit. |
| T39 | Q-Delta | Dk≠Dv outer-product shape; direct Eq.19 vs ABI, λ=0 scalar-GDN limit, q-feedback gradient, α once; asymmetric chunked reference vs serial. |
| T40 | Preconditioned DeltaNet | Exact SM inverse vs direct SPD inversion; diagonal normalized Eq.8 и squashed Eq.9, positive A/ridge, αP/βP/μ/x gradients и bounds, repeated/collinear keys; paper practical вариант не сравнивается как exact inverse. |
| T41 | Composition derivation | Минимум QED+OSDN, EDA+DP+KDA, SSE+QED, SSE+OSDN, CCQ+aligned, quant+DP, StateX+SSE; direct matrix/carry/VJP fixtures и property bounds. Неподдержанное сочетание early reject с reason; accepted full preset проходит off baseline. |
| T42 | Level allocator | 2^r±1 promotions, создание уровня выше trained max, copy/recompile carry equality, event byte/latency accounting, explicit bounded overflow vs unbounded growth; no level clipping. |
| T43 | External archive | Exact content/hash roundtrip, causal version/overwrite/latest-visible, query without gold ID, retrieval recall@k отдельно от answer, append crash/replay без duplicate/missing IDs; bounded working cache, растущий disk честно учтён. |
| T44 | DLA reference vs adapter | Independent RMS state-variation score; tentative boundary-before-token fixture с rejected updates; information sums/density; soft/hard train gap и masked min/max surrogate derivatives. |
| T45 | Memory coverage | После каждого close/merge/promotion один владелец каждого closed span; disjoint bins, no retained children, variable-length promotion units, active span отдельно. Paper-loglinear transitions и lossy-bank representation тестируются разными oracles. |
| T46 | StateX corrected contract | Paper-reinit не требует initial logits parity; все reshaped tensors/caches, cross-head blocks и nonzero gradients проверены. Optional preserve migration требует exact declared transform и parity; unsupported import rejected; optimizer migration имеет отдельное policy. |
| T47 | Budgeted protocol | Одни encoding/query memory+quant budgets от первого token; fact/overwrite по всей истории; no hidden uncapped prefix; router oracle после causal query-key vs privileged gold-bank oracle; candidate/selection/answer scores разделены. |
| T48 | Spec/evidence coverage | 19 source versions+local B6; 16 core production paths+carry/checkpoint/off/ref/tests, все R01–R33 связаны с G, all allowed capabilities обнаружены runner; absent hardware/result не получает VALIDATED. |

### Общие правила и параметры

1. Independent NumPy float64 прямые формулы, без production imports. Проверять outputs, final state/весь carry и VJP каждого continuous input, h0 и dh_final.
2. B=1/2, H=1/4, Ha=1/2/H, Dk/Dv small и unequal, затем 64/128; lengths 0/1/C−1/C/C+1, boundary/merge/quant/growth events, несколько seeds.
3. FD с несколькими epsilon/directional derivatives и VJP dot-product identity; hard ties/boundaries и ST отдельны. Exact off switches, не большие отрицательные biases.
4. Float64 FD starting atol 1e−8/rtol 1e−5; float32 starting 1e−5…1e−4, до запуска выбрать scale/conditioning budget. bf16x3/default tolerances определяются precision study, не одним широким процентом.
5. Для quant train-free проверять inference, для STE — отдельный surrogate. Domain/clip/eps сравниваются с правильным named reference.
6. Suite authoring = шаг 3; run = шаг 4. PASS/FAIL/SKIP/INVALID/INCONCLUSIVE, backend/device/hashes и raw results обязательны.

## Сводное покрытие расхождений

| Finding | Основные work packages |
|---|---|
| R01 | G004/G024/G025 |
| R02 | G041 |
| R03 | G017/G022 |
| R04 | G020/G021 |
| R05 | G025/G026/G071 |
| R06 | G082 |
| R07 | G032–G035/G031 |
| R08 | G036–G040/G046 |
| R09 | G037/G042/G043 |
| R10 | G047–G049/G041 |
| R11 | G029/G045 |
| R12 | G028/G035/G038/G039 |
| R13 | G050–G052 |
| R14 | G044/G045 |
| R15 | G057–G060/G074 |
| R16 | G061–G063 |
| R17 | G006/G064–G067 |
| R18 | G016/G028 |
| R19 | G018/G019/G022 |
| R20 | G010/G072 |
| R21 | G011/G072 |
| R22 | G009/G013/G021 |
| R23 | G008/G012 |
| R24 | G002/G008/G071–G074 |
| R25 | G027/G031 |
| R26 | G005/G014/G075/G077 |
| R27 | G080/G084 |
| R28 | G056/G081/G082/G085 |
| R29 | G046/G083 |
| R30 | G006/G055/G068–G070 |
| R31 | G002/G014/G015 |
| R32 | G003/G022/G049/G076 |
| R33 | G024/G027/G030/G036/G037/G046/G053/G068/G079 |

## Definition of done будущего проекта

- [ ] D0: baseline/state/gradient correctness подтверждена независимым oracle; zero-decay/clip/eps defects закрыты.
- [ ] D1: все 16 core modules имеют реальные integration/carry/config/checkpoint paths и правильное paper/variant имя.
- [ ] D2: stream chunking/reset/resume/budget/level growth выполняют спецификацию; working-memory accounting совпадает с фактической аллокацией.
- [ ] D3: valid composition и off switches подтверждены; invalid combinations отклоняются явно.
- [ ] D4: CPU/JAX tiers подтверждены; Pallas tier отдельно подтверждён hardware либо честно SKIP.
- [ ] D5: paired healthy baseline и конечные long-context curves, uncertainty/iso-resource controls; compression/migration costs учтены.
- [ ] D6: persistent exact storage не смешано с summary quality; causal retrieval и disk/I/O growth измерены.
- [ ] D7: STATUS публикует конкретные lengths/tasks/hardware и actual evidence, без обещания lossless infinite или неподтверждённых speedups.

**Количество:** 92 work packages + 8 критериев готовности; 48 сценариев тестирования, 33 findings и 20 модульных записей источников. Ничего из implementation/validation не отмечено выполненным на шаге 2.
