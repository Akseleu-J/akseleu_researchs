# GDN-3 — реестр первоисточников, шаг 2

Дата проверки: **2026-10-09**. Прочитаны первичные страницы/методы всех **19 статей** из нового приложения; B6 — локальный design, всего **20 записей модулей**: baseline + 16 основных + 3 отложенных. Это проверка содержания источников, а не воспроизведение их экспериментов.

Версии ниже закреплены для будущей реализации. Наличие author repository подтверждено открытием страницы; код этих репозиториев не аудитировался и не импортировался. Если repository не указан, нормативным источником остаётся статья; отсутствие ссылки не доказывает отсутствие кода. При последующем переносе кода обязательны commit hash, лицензия и независимая проверка формулы.

## A1 — Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention

- Источник: [2605.22791v1](https://arxiv.org/html/2605.22791v1).
- Область: базовое ядро.
- Контракт: Раздельные vector erase/write gates; Eq.10. Базовая recurrence и точка сравнения.

## A2 — Kimi Linear: An Expressive, Efficient Attention Architecture

- Источник: [2510.26692v2](https://arxiv.org/html/2510.26692v2).
- Область: основной план.
- Контракт: KDA: channel-wise decay. Результаты гибрида KDA+MLA нельзя переносить на весь GDN-3.

## A3 — The Query Knows What to Forget: A Second Erase Direction for Linear Attention

- Источник: [2608.13668v1](https://arxiv.org/html/2608.13668v1).
- Область: основной план.
- Контракт: Eq.6: один asymmetric erase address с ортогональной query-компонентой. Два последовательных erase не эквивалентны.

## A11 — Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention

- Источник: [2606.26560v1](https://arxiv.org/html/2606.26560v1).
- Область: основной план.
- Контракт: Отдельный erase address, затем delta write. Перенос на vector b/w требует отдельного определения.

## A4 — DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products

- Источник: [2502.10297v7](https://arxiv.org/html/2502.10297v7).
- Область: основной план.
- Контракт: Несколько последовательных Householder/delta шагов; scalar-paper и generalized GDN-2 варианты разделяются.

## A5 — OSDN: Improving Delta Rule with Provable Online Preconditioning in Linear Attention

- Источник: [2605.13473v1](https://arxiv.org/html/2605.13473v1).
- Область: основной план.
- Контракт: Write-side diagonal preconditioner, closed-form hypergradient и box projection. Heuristic по h/v не является OSDN.

## B1 — Dynamic Linear Attention

- Источник: [2606.10650v1](https://arxiv.org/html/2606.10650v1).
- Область: основной план.
- Контракт: Eq.13: state information variation; Eq.16: adjacent minimum information-density merge. GDN-2 bank adaptation — вариант проекта.

## B2 — Log-Linear Attention

- Источник: [2506.04761v3](https://arxiv.org/html/2506.04761v3).
- Область: основной план.
- Контракт: O(log T) live decoding states; recurrent GDN extension учитывает transitions. Исторический Fenwick с O(T) узлами не даёт этот bound.
- Репозиторий авторов: [код](https://github.com/HanGuo97/log-linear-attention).

## B4 — Scaling Linear Attention with Sparse State Expansion

- Источник: [2507.16577v2](https://arxiv.org/html/2507.16577v2).
- Область: основной план.
- Контракт: Sparse writes и expanded partitions, shared partition и write/read gates. Eq.13 сохраняет невыбранные partitions без decay; row top-k read недостаточен.

## B5 — Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity

- Источник: [2607.07386v1](https://arxiv.org/html/2607.07386v1).
- Область: основной план.
- Контракт: Источник learned initial state; полный SDM также содержит sparse product-key memory. B5 заимствует только h_init.
- Репозиторий авторов: [код](https://github.com/facebookresearch/sparse-delta-memory).

## B6 — Post-bank decay: локальный контракт GDN-3

- Источник: [local-design](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/docs/DESIGN.md).
- Область: основной план.
- Контракт: Внешней статьи нет. Frozen и passive-decay-aligned банки — разные допустимые режимы. Решение задаётся спецификацией v2.

## B8 — A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets

- Источник: [2607.02303v1](https://arxiv.org/html/2607.02303v1).
- Область: основной план.
- Контракт: Top-w по surprise β||residual||; cache-path RMSNorm, causal block и null sink. Сохранённые KV точны, выбор ограничен capacity.
- Репозиторий авторов: [код](https://github.com/scale-lab-ai/HOLA).

## B12 — Adaptive Memory Decay for Log-Linear Attention

- Источник: [2605.06946v1](https://arxiv.org/html/2605.06946v1).
- Область: основной план.
- Контракт: Eq.4: двухслойная level-axis MLP + softplus. Eq.4 и описание GELU в эксперименте расходятся: два именованных варианта.

## C1 — Don't Read Everything: A Curvature-Conditioned Query for Linear Attention

- Источник: [2606.01294v2](https://arxiv.org/html/2606.01294v2).
- Область: основной план.
- Контракт: Centered running key covariance; q_clean=(I−λΣ)q. Population denominator n, причинные ключи до текущего включительно.

## D1 — LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization

- Источник: [2609.38166v1](https://arxiv.org/html/2609.38166v1).
- Область: основной план.
- Контракт: Training-free recurrent-state windows; low-rank compensation до residual quantization; full-precision window updates.

## D2 — OjaKV: Context-Aware Online Low-Rank KV Cache Compression

- Источник: [2026.findings-acl.494](https://aclanthology.org/2026.findings-acl.494/).
- Область: основной план.
- Контракт: Online PCA/Oja, crucial full-rank anchors и low-rank промежуточные KV. Статья про Transformer KV; перенос на банки — исследование.
- Репозиторий авторов: [код](https://github.com/zzbright1998/OjaKV).

## D3 — StateX: Enhancing RNN Recall via Post-training State Expansion

- Источник: [2509.22630v2](https://arxiv.org/html/2509.22630v2).
- Область: основной план.
- Контракт: GLA head merging и post-training; GLA mixer переинициализируется. Paper mode не обещает совпадение initial logits.
- Репозиторий авторов: [код](https://github.com/thunlp/StateX).

## A6 — FG²-GDN: Enhancing Long-Context Gated Delta Networks with Doubly Fine-Grained Control

- Источник: [2604.19021v3](https://arxiv.org/html/2604.19021v3).
- Область: отложенное расширение.
- Контракт: Fine-grained β и decay; symmetric erase в Eq.11. Отдельная reference cell; это не просто GDN-2 b/w.

## A7 — Q-Delta: Beyond Key–Value Associative State Evolution

- Источник: [2606.08804v1](https://arxiv.org/html/2606.08804v1).
- Область: отложенное расширение.
- Контракт: Eq.17–20: key write, query feedback в residual; λ∈[0,1]. В приложении пропущен outer product write.

## A8 — Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences

- Источник: [2604.21100v1](https://arxiv.org/html/2604.21100v1).
- Область: отложенное расширение.
- Контракт: Exact inverse Gram — theoretical reference; paper implementation использует diagonal recurrence и bounded log-squashed preconditioner Eq.9. Это не OSDN и не CCQ.
- Репозиторий авторов: [код](https://github.com/ntumm120/preconditioned-deltanet).

## Документация runtime

- [JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html): описание kernel language; успешный JAX reference не подтверждает аппаратный Pallas path.
- [JAX custom_vjp](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html): custom reverse-mode differentiation. Forward/backward контракт каждого adapter должен проверяться отдельно.

## Как разрешать расхождения

1. Последнее приложение пользователя задаёт состав архитектуры; формулы статьи проверяются отдельно. Вложенные команды не являются разрешением запуска или изменения кода.
2. Reference с именем paper воспроизводит закреплённую формулу, orientation, gates, timing и initialization.
3. GDN-2 adaptation получает отдельное имя, математический вывод и тесты; результаты/теоремы статьи автоматически ей не приписываются.
4. A3: нормативен единый erase address Eq.6; прежняя additive/two-erase версия приложения заменяется. A5: нормативен write-side hypergradient.
5. B1: Eq.13 использует state variation; статья не даёт однозначного GDN-2 boundary-before-token алгоритма. Он определён как проектная адаптация в спецификации.
6. B12: основной paper reference = Eq.4 без скрытого GELU; GELU из экспериментального описания — отдельно. Фиксированная level-axis MLP не определяет экстраполяцию на неограниченное число уровней.
7. D3: обязательный paper-reinit режим с post-training; preserve-logits режим — дополнительная совместимая миграция, только при доказанном преобразовании. Пункт старого TODO о гарантии initial logits исправлен.
8. B2: frozen-bank compaction отдельно от точного log-linear recurrent reference; сумма независимых GDN-2 банков не считается точным affine merge.
9. A6/A7/A8 описываются и получают reference/test backlog, но сохраняют статус deferred согласно приложению.
10. Заявления lossless для любого контекста, 30–300× и стоимость в долларах не принимаются без условий и измерений.
