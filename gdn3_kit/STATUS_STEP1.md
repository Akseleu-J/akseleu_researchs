# GDN-3 — обновлённый статус

**Дата:** 9 октября 2026. **Метод:** полный статический аудит доступных локальных исходников; новые тесты/обучение/бенчмарки не запускались. Реализация не менялась.

## Документы

- [Полный аудит](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_IMPLEMENTATION_AUDIT.md): 16 требований приложения, 33 точных расхождения/дефекта/риска, ссылки на исходники и первоисточники, 37 сценариев тестирования.
- [Полный TODO](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_TODO.md): чек-лист реализации, зависимости и критерии приёмки.
- [Manifest снимков](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_AUDIT_MANIFEST.json): SHA-256 82 файлов, включая все 72 Python-файла kit/full/atomic, и трёх приложенных входов.
- [Предыдущая рабочая заметка](C:/Users/omirb/Downloads/Downloads/gdn3_kit/STATUS_PREVIOUS_AUDIT.md): сохранена для истории. Исходный Downloads/STATUS.md не изменён.

## Что нормально

- kit содержит GDN-2 + независимые банки, causal read закрытых сегментов, null router, hybrid gate, serial/Pallas adapter и streaming carry.
- Произвольная длина kit/full forward поддерживается top-level pad/crop. Внутренний mixer alignment check корректен; прежнее замечание об отсутствии padding было ошибочным.
- full содержит full per-channel KDA mode, generalized EDA/DP substeps, learned bank init, decay-aware archive/export и exact index top-k router.
- Есть 19 именованных test_* функций kit и 33 full, а также отдельные Fenwick/Pallas main-checks. Наличие сценариев не является результатом запуска.

## Блокеры доверия к текущим результатам

| Приоритет | Проблема | ID аудита |
|---|---|---|
| P0 | Hybrid scan full передаёт H-head substeps в Ha-head archive state; ломается при части архивных голов | R01 |
| P0 | Router kit выбирает >k при ties; sparse decode теряет массу относительно dense/parallel | R02 |
| P0 | QED отличается и от paper Eq.6, и в общем vector-gate случае от additive формулы приложения | R03 |
| P0 | OSDN heuristic и bilateral key scaling не воспроизводят paper hypergradient/write-only update | R04 |
| P0 | Bounded prefill использует uncapped prefix; kit archive-policy scoring фактически оценивает uncapped logits | R05–06 |
| P0 | Atomic VJP использует zero masks при g=0; lean backward не учитывает diagonal clipping | R20–21 |
| P0/P1 | Guards/sanitize, WY epsilon и runner/verdict не обеспечивают объявленную correctness/сертификацию | R22–23, R26–27, R31 |

Эти выводы основаны на call-path, формах и алгебре исходников. Динамическая воспроизводимость отказов и исправлений должна быть подтверждена предложенными тестами.

## Требования, которые не закрыты

- B1: information-based merge отсутствует; threshold gradient snap не обучается через имеющийся surrogate.
- B2: standalone Fenwick хранит O(N) узлов и не подключён к router. Alias policy=fenwick означает leveled compaction.
- B4: SSE реализовано как sparse read, а требовалась sparse write и expanded partitions.
- B8: HOLA использует threshold FIFO вместо top-w surprise; нет paper cache-path RMSNorm.
- B12: MLP управляет age penalty, а не weights интегрированной log-linear иерархии.
- C1: CCQ использует SSᵀ вместо centered running key covariance. Первоисточник: arXiv 2606.01294.
- D1: int8 closed-bank storage + exact entries не является window LeapQuant; actual memory включает float32 sketch/active states.
- D2/D3: Oja/SVD/padding helpers не интегрированы как OjaKV/StateX-модель.
- Новые separate Pallas kernels/custom_vjp, полный FD-suite, inference cache migration/resume и overflow-safe counters не завершены.

## «Бесконечный контекст» и сложность

Текущий parallel archive перебирает все N банков: O(L·N·D²) чтение плюс router; при N≈L/C это O(L²/C). Decode router имеет O(N), даже если D² read касается только top-k. Для фиксированного бюджета стоимость ограничена, но история отбрасывается/сжимается.

Цель нужно формулировать как непрерывный streaming и измеряемое долговременное качество при заданном memory budget. Универсальное точное восстановление произвольной независимой истории при O(log L) finite-precision storage невозможно по числу различимых состояний. При требовании любого старого факта нужен растущий exact archive; bounded HOLA и summaries защищают выбранную информацию.

## Статусы этапов

| Этап | Текущий статус | Условие закрытия |
|---|---|---|
| M0 reference | **Не подтверждён новым запуском** | Independent formula/VJP tests, streaming/budget/head/padding edge cases |
| M0 Pallas | **Не подтверждён для текущего atomic** | g=0/clip/h0/dh_final tests + parity на закреплённой TPU/JAX/kernel версии |
| M1 capacity | **OPEN / NOT VALIDATED** | Healthy paired baseline/learned/oracle diagnostic, ≥3 seeds, adequate P-grid, uncertainty, iso-state/runtime |
| Long streaming hierarchy | **Не реализовано полностью** | Integrated active bins с declared O(log L) live storage, same-engine semantics и resume |
| Paper extensions | **Частично / есть другие варианты** | Матрица 16 пунктов в полном аудите, independent reference и feature-off parity |
| GDN3-Inf | **Не достигнуто** | Sustained stream, length extrapolation, quality/resource contract и sufficient exact storage при arbitrary recall |

## Порядок следующей работы

1. Зафиксировать спецификацию, canonical snapshot и зависимости.
2. Исправить correctness/math/backward/budget/runner блокеры и добавить T01–T06/T18/T24–T33.
3. Довести единый streaming manager, information merge, log-linear bins, clocks/counters/cache resume.
4. Добавить priority exact memory; определить внешний архив, если требуется любой старый факт.
5. Проверять cell/read модификации по одной; затем quant/compression/StateX и Pallas optimization.
6. Закрыть M1 и конечные long-context experiments на честном протоколе.

Полная реализация ожидает отдельной команды пользователя. Сейчас подготовлены только документы аудита и TODO.
