# GDN-3 — статус после шага 3

**Дата: 2026-10-10. Шаг 3 генерации проекта выполнен частично; шаг 4 (запуск тестов, обучение и эксперименты) не начинался.**

Второй этап дал аудит/спецификацию/TODO. По прямой команде пользователя создан отдельный пакет [gdn3_v2](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_v2). Это новый implementation snapshot; исходные kit/full/atomic не изменены.

## Что создано

- Public API: валидируемая конфигурация, parameter init, prefill/decode/full forward, masks, reset, growing carry/cache и health counters.
- Основные cell/memory/compression/backend подсистемы: asymmetric recurrence ABI, QED/EDA/DeltaProduct/OSDN/KDA adapters, hierarchy/loglinear/DLA/router/SSE/HOLA/CCQ/adaptive λ, LeapQuant/OjaKV, SQLite archive, JAX/Pallas dispatch.
- Vendored atomic source с отдельным журналом изменений: [VENDOR_PATCHES.md](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_v2/docs/VENDOR_PATCHES.md).
- Независимый NumPy float64 recurrence oracle, versioned bundle/cache/checkpoint; строгие adapter contracts для atomic import и StateX planning/reinitialization.
- AdamW + gradient clipping + warmup/cosine LR + EMA + RNG/data cursor/skipped count + checkpoint resume scaffold; CLI и JSON config. Генератор данных синтетический, не корпус и не качество.
- Capability ledger, runtime report, tiered test runner, test plan, README и начальные 12 тестовых файлов: [CAPABILITIES.md](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_v2/docs/CAPABILITIES.md), [TEST_PLAN.md](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_v2/docs/TEST_PLAN.md).

## Статическая проверка этого шага

- Python AST parse: **87 Python-файлов, 0 синтаксических ошибок**. Код проекта не импортировался и не исполнялся.
- JSON: файлы пакета разбираются без ошибок.
- Исходный manifest: SHA-256 совпал для **82/82** файлов; доноры не модифицированы.
- Проектные тесты: **не запускались** по границе этапов, назначенной пользователем. Тесты не подтверждены как проходящие.
- Локальная среда ранее показала отсутствие JAX/jaxlib/Optax/pytest; для шага 4 потребуется соответствующий runtime.

## Что ещё не доведено до архитектурного контракта

| Область | Текущее состояние и расхождение | Следующий конкретный шаг |
|---|---|---|
| Полный тестовый пакет | Написаны только начальные 12 test modules; далеко не все T01–T48 покрыты. Ни один не запускался. | Авторить недостающие независимые refs/fixtures и регрессии T01–T48; runner уже создан, но не запускался. Затем — команда шага 4. |
| StateX | Есть план merge и явный reinit callback; нет callback, который перестраивает целую модель/проекции/conv/router/SSE, optimizer schema и cache. | Реализовать checkpoint-level migration, выбранный head map, бумажную reinit и миграционный отчет; не заявлять parity. |
| Atomic import | Есть named adapter, но он ожидает заданную нормализованную mapping-схему, не читает произвольный исходный checkpoint напрямую. | Добавить reader для конкретного реального checkpoint формата и проверить целостность всех полей. |
| Persistent retrieval | Оригиналы и точный disk scan есть; scan O(history). Model API принимает per-layer retrieval tensors; host session требует application-supplied index/key/value projection и ещё не образует end-to-end archive→projection→decode facade во всех режимах. | Закончить typed host orchestrator, causal cursor/overwrite reconciliation, mapping heads/layers и retrieval metrics. |
| OjaKV | Онлайн basis/anchors реализованы helper-ом; полностью в model live retrieval cache и долговременный resume lifecycle не встроен. | Связать KV working cache с session, зафиксировать projection/basis schema и добавить remap/error tests. |
| Training | Реализован full-BPTT синтетический pipeline. TBPTT теперь явно отвергается. | Подключить реальную dataset/data cursor API и затем отдельную TBPTT транзакционную carry semantics; не смешивать с шагом 4 запуска. |
| Численная математика | Есть production helpers, но независимые градиентные/сигнальные тесты не написаны в полном объеме. | Сверить каждый formula/VJP против reference, особенно generalized OSDN/QED, precision и atomic kernel. |
| Pallas/TPU | Источник primitive есть; JAX/Pallas compiler и аппаратный прогон недоступны/не запускались. | Установить фиксированный runtime, выполнить CPU correctness и TPU tier; явно SKIP hardware tier, если TPU нет. |
| Долгий поток/«бесконечный контекст» | Счётчики и геометрический рост позволяют не задавать малый фиксированный предел; summary lossy, точный архив растёт на диске. Ни retrieval accuracy, ни долгосрочное качество не подтверждены. | Longevity/capacity tests, paired MQAR/real-data evaluations, resource/quality accounting; не заявлять универсальный lossless recall. |

## Приоритет дальнейшей доработки

1. Завершить внешнюю session интеграцию и atomic/StateX migration до полного checkpoint/carry contracts.
2. Дописать недостающие independent T01–T48 references/tests и сверить coverage с [TODO](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_TODO.md).
3. Перед шагом 4 зафиксировать supported Python/JAX/jaxlib/Optax версии и install lock, затем пользователь отдельно разрешит запуск.
4. Только после корректностных тестов измерять качество и memory/latency на заявленных 2K→1M наборах.

GDN-3 **не является завершённой/проверенной моделью**. Шаг 3 создал значительный кодовый scaffold, но ряд ключевых требований TODO (StateX целиком, performance-ready external retrieval/Oja integration, полные 48 тестов, runtime validation и measured long-context quality) остаётся открыт. Цель универсального lossless бесконечного контекста конечной памятью математически не обещается; проектный профиль даёт непрерывный stream + lossy summary либо растущий точный внешний архив с retrieval, качество которого предстоит измерить.
