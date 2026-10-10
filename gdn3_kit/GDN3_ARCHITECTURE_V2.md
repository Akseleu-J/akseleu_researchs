# GDN-3 v2 — спецификация переделки архитектуры

**Шаг 2. Дата: 2026-10-09. Статус: спецификация подготовлена; реализация, новые тесты и их запуск не выполнялись.**

Документ задаёт будущую реализацию после отдельной команды пользователя. Цель — полный GDN-3 с непрерывным streaming, многоуровневой памятью и проверяемым долговременным recall. Состав соответствует последнему приложению: baseline A1, 16 основных модулей и 3 отложенных. Формулы исходного плана исправлены по первоисточникам. Все решения с пометкой **проектная адаптация** требуют собственной проверки.

## 1. Основание и границы шага

- [Аудит R01–R33 и исходные T01–T37](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_IMPLEMENTATION_AUDIT.md) сохраняет точные места дефектов в kit/full/atomic.
- [Реестр 19 статей, локального B6 и runtime](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_SOURCES.md) содержит закреплённые версии и ссылки.
- [TODO v2](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_TODO.md) разбивает эту спецификацию на зависимые работы и условия приёмки.
- Предыдущий TODO сохранён как [STEP1](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_TODO_STEP1.md). В v2 исправлено прежнее обязательное требование preserve-logits для paper StateX, уточнён сигнал DLA и добавлены A6/A7/A8.

Вложения используются как проектные требования. Команды, обещания и оценки внутри них не разрешают исполнение кода, обучение или публикацию. Пользователь начинает каждый следующий шаг самостоятельно.

## 2. Что означает цель «бесконечный контекст»

**Инженерная цель:** продолжать поток без заранее заданного максимального числа токенов, переносить полный carry между вызовами/процессами, не ломать causal semantics при росте истории и сохранять измеряемое качество в объявленных задачах.

Это не универсальная гарантия lossless. При фиксированном числе бит существует конечное число состояний; число независимых историй растёт экспоненциально с длиной. Поэтому фиксированная или O(log L) память не может различать все произвольные истории и отвечать точно о любом их элементе. Это собственный counting argument, а не результат отдельной статьи.

| Профиль | Что хранится | Контракт | Ограничение |
|---|---|---|---|
| stream-summary | Cumulative state + recent banks + логарифмические summaries + bounded exact HOLA | Непрерывный поток; рабочая память O(log L); качество на заданных распределениях | Старые сводки и отброшенные KV имеют информационные потери |
| stream-persistent | Всё из stream-summary + растущий внешний архив оригинальных token/document spans и индекс | Точное хранение записанных данных; причинный retrieval; отдельно измеряемые retrieval и answer accuracy | Полное хранение растёт O(L); модель не получает автоматической гарантии найти/понять любой факт |
| paper-reference | Закреплённый метод одной статьи или явно выведенная адаптация | Сопоставление формул, outputs, states и gradients | Не заменяет качество обученного GDN-3 |

Заявления 8K/1M «без потерь», 30–300× и цены из вложения становятся гипотезами для эксперимента. Длины 2K, 8K, 32K, 128K, 1M — точки измерения, не доказательство бесконечности. Memory bound относится к inference carry; full BPTT требует отдельной памяти истории.

## 3. Каноническая структура будущего проекта

**Решение:** будущий пакет — отдельный каталог **C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_v2**. Сейчас он не создаётся. kit служит исходной архитектурой baseline; full — донором проверяемых компонентов; atomic — backend, который требует исправления R20–R24. Ни один старый snapshot не принимается целиком как корректный.

Предлагаемые файлы внутри будущего каталога, пока отсутствуют:

| Подсистема | Файлы/каталоги | Ответственность |
|---|---|---|
| Public API | gdn3/config.py, model.py, stream.py, capabilities.py | Validate, prefill/decode, reset, capability report |
| Mathematics | gdn3/cells/{base,kda,qed,eda,delta_product,osdn}.py | Канонические substeps и gates |
| Deferred cells | gdn3/cells/{fg2,qdelta,preconditioned}.py | Именованные paper references, не default |
| Memory | gdn3/memory/{bank,dla,hierarchy,loglinear,merge,router,hola,sse,ccq}.py | Причинная запись/чтение, coverage и live-memory bound |
| Compression | gdn3/compression/{leapquant,oja_kv}.py | Window state и KV basis, accounting |
| Persistence | gdn3/persistence/{cache,checkpoint,archive,index}.py | Atomic save/load, schemas, external retrieval |
| Migration | gdn3/migration/{atomic_import,statex}.py | Whole-model migration, head expansion, validation |
| Runtime | gdn3/backends/{reference,jax,pallas}/ | Общий ABI, precision policy и kernels/VJP |
| Evidence | reference/, tests/, bench/, configs/, docs/ | Независимые эталоны, воспроизводимые проверки и эксперименты |

Модуль считается реализованным только после подключения к model/carry/checkpoint и отсутствия silently ignored config. Наличие helper не закрывает требование.

## 4. Общая математическая ABI

### 4.1 Формы и основная операция

Фиксируем orientation: S имеет **Dk×Dv**, q/k/r/u — Dk, v/z/residual — Dv. Batch/head axes объявляются отдельно. Допускается Dk≠Dv. H — cumulative heads, Ha — архивные головы; выбор Ha реализуется явными indices для **всех** fields и extras, включая preconditioner.

Один substep:

    Sbar = diag(exp(g)) @ S_prev
    residual = z − rᵀ @ Sbar
    S_next = Sbar + outer(u, residual)
    output = scale * q_readᵀ @ S_next

Эквивалентно: P=(I−u rᵀ) diag(exp(g)), U=u zᵀ, S_next=P S_prev+U. Один token может содержать список substeps. Passive decay применяется **один раз**, на первом substep; последующие g=0. Чтение выполняется после последнего substep. Padding/no-op сохраняет всё состояние, включая conv, counters, covariance, quant windows и external events.

### 4.2 Baseline A1

GDN-2: **u=k, r=b⊙k, z=w⊙v**; b по key channels, w по value channels. Scalar beta — специальный вариант, не замена общего b/w. Cumulative h запускается как у импортируемого baseline; learned bank initialization относится к отдельному архиву. Нормы q/k, convolution, output norm/gate/scale, embeddings/FFN и precision сохраняются при baseline import. [GDN-2, Eq.10](https://arxiv.org/html/2605.22791v1).

Baseline-preserving preset: A1, исходный decay, archive/HOLA/CCQ/SSE выключены; дополнительные substeps отсутствуют; quant отключён; WY eps=0; параметры импортированы явно. gate=0 вычисляется как точный bypass архивного чтения. Это требование parity, а не заявление, что текущий код уже его выполняет.

### 4.3 Состав substeps

Проектный compositional preset: сначала отдельный EDA erase, затем m DeltaProduct slots. Каждый slot имеет собственные k/v/b/w; выбранный QED adapter меняет только r соответствующего slot. OSDN изменяет write direction u только в именованной адаптации. HOLA surprise агрегируется по фактическим write deltas этих slots; current output читается после всех slots. Gate strengths/normalization определяются для каждого режима.

Наличие общей ABI обеспечивает размерную совместимость, но не доказывает устойчивость или полезность композиции. Scalar-paper, generalized-GDN2 и compositional presets имеют разные имена и capability entries.

## 5. Контракты cell-модулей

### A2 — KDA / channel-wise decay

Полный input-dependent g имеет B×T×H×Dk. Отдельный learned rate H×Dk при input signal H ещё не является полной KDA-параметризацией. Нужны paper reference gate и GDN-2 gate adapter с указанными low-rank projections/activations; simple full projection — отдельно именованный вариант. g≤0, no-decay g=0, gradients по input gate и rates. Не приписывать скорости Kimi Linear гибридного KDA+MLA этому проекту. [Kimi Linear](https://arxiv.org/html/2510.26692v2).

### A3 — QED

Для normalized k: **r=b⊙k + λ_h P⊥k(b⊙q_raw)**, u=k, z=w⊙v; P⊥k(x)=x−k(kᵀx); λ_h≥0 обучаемый per-head. Для nonunit/near-zero k нужно явно задать norm-aware projector и eps, обозначив вариант. Один rank-one erase. Старые two-erase substeps не используются под именем QED. q_raw для записи не подменяется CCQ q_read. [QED, Eq.6](https://arxiv.org/html/2608.13668v1).

### A11 — EDA

Перед main delta — erase-only substep с собственным normalized e: u=e, r=γe, z=0. Первый substep получает g; main получает g=0. Это проектная запись paper erase-then-delta в общем ABI; generalized b/w main описывается отдельно. γ=0 должен быть точным bypass; extra projection gradients проверяются. [EDA](https://arxiv.org/html/2606.26560v1).

### A4 — DeltaProduct

m=n_h≥1 последовательных delta steps; m=1 даёт baseline. Paper scalar β и его разрешённый диапазон закрепляются отдельно от generalized vector gates. Разделяем независимые learned addresses и физическое переиспользование одинакового токена. Только первый step decays; read после всех; m не может быть отрицательным. [DeltaProduct](https://arxiv.org/html/2502.10297v7).

### A5 — OSDN: paper и GDN-2 adapter

Paper orientation преобразована к нашей: u=β(d⊙k), r=k, z=v. **Residual использует исходный k**. h(d)=((1−β〈d,k²〉)²−1)/(2||k||²); grad_d h=−β(1−β〈d,k²〉)k²/||k||²; d_next=clip(d−η grad,d_min,d_max). Carry содержит d; текущая запись использует d_t, следующий token — d_next. k≈0 имеет явный no-update convention. [OSDN](https://arxiv.org/html/2605.13473v1).

**Проектная адаптация:** u=d⊙k, r=b⊙k (или QED r), z=w⊙v. При фиксированных r,k,z для локального objective f=.5||z−rᵀS||², c=rᵀ(d⊙k), получается нормированный surrogate h=((1−c)²−1)/(2||r||²), grad=−(1−c)(r⊙k)/||r||². Это алгебраический вывод проекта; OSDN paper proof ему автоматически не приписывается. При нулевых r/residual задаётся hold-d. Для r с signed coordinates box projection сама по себе не гарантирует 0<c<2; требуется отдельный feasibility/stability policy либо отказ такой композиции. Спектральные eigenvalue bounds не заменяют контроль operator norm у asymmetric update.

Порядок выпуска: paper OSDN reference → generalized GDN-2 reference → continuous VJP/health → комбинации с QED/EDA/DP. Нельзя подключить старый heuristic и закрыть A5.

## 6. Единый streaming manager и временной порядок

### 6.1 Состояние слоя

Carry содержит: cumulative h[H,Dk,Dv]; active bank[Ha,...] и его init; convolution ring; DLA previous candidate/info/n_tokens; recent closed-bank queue; hierarchy occupancy/spans/levels; router descriptors; passive clock при aligned mode; SSE partition states; HOLA records и causal current-block KV; CCQ count/mean/M2; OSDN d; LeapQuant lowbit state/window/compensators; Oja basis/anchors; versioned sequence ID/absolute position/reset mask; external durable cursor. Все опциональные fields типизированы, имеют shape/dtype/ownership и отражаются в byte accounting.

### 6.2 Token transaction

1. Проверить valid/reset mask. Masked token не изменяет ни один carry field; reset очищает только соответствующую sequence.
2. Получить causal features с корректным convolution carry; построить raw q/k/v/g/b/w и substeps.
3. Обновить cumulative h. Для active bank вычислить **tentative continuation** на его прошлом состоянии и information score из state variation; решение зависит только от прошлого и текущего token.
4. Если hard boundary-before-token разрешена min/max, закрыть предыдущий непустой bank с end=t−1; выполнить insert/merge/promotion; reset active bank learned h_init и committed update текущего token. Tentative continuation, отвергнутая boundary, не попадает в HOLA/d/cov clocks.
5. Иначе committed update остаётся в active bank. OSDN и quant carry меняются по committed path. Passive clock включает текущий valid token ровно один раз.
6. Обновить CCQ статистики текущим raw normalized main k. Admission HOLA — по committed write delta; добавить текущий KV в causal block. В persistent profile append оригинального span идёт с version/cursor.
7. Прочитать cumulative/local и разрешённые closed banks, HOLA и external candidates. current active bank не считается closed archive; current block KV имеет causal mask. Смешать outputs по выбранному mode.
8. Выдать logits + весь new carry + health/counters/events. Любая ошибка resource/validation возвращается явно.

Boundary-before-token — **проектный контракт**, согласованный с банками GDN-2, не дословный перенос DLA additive алгоритма. Prefill, chunked prefill и decode вызывают эту семантику с одинаковым budget от первого token. Parallel kernels допускаются лишь при доказанной эквивалентности; иначе runtime выбирает reference/streaming adapter и сообщает фактический backend. Обрезание уже рассчитанного uncapped prefix не принимается.

### 6.3 Modes

| Mode | Backbone read | Archive | Условие |
|---|---|---|---|
| gdn2 | Cumulative GDN-2 | Выключен | Exact baseline-preserving preset |
| hybrid | Cumulative GDN-2 + gated archive | Независимые сегментные банки | Основной рабочий профиль |
| local | Active bank + gated archive | Независимые банки | warm-start применяется только здесь и отдельно тестируется |
| snap | Cumulative snapshots | Выбранный snapshot или выведенные disjoint deltas | Нельзя складывать overlapping snapshots как непересекающиеся банки |

Default цель — hybrid stream-summary, frozen archive, exact top-k router, shared-level λ, bounded HOLA; дополнительные cells/compression подключаются по validated presets. До обучения не заявляется выигрыш. Для gate=0 archive bypass сохраняет cumulative computation.

## 7. B1/B2 — сегменты, сводки и log-linear память

### 7.1 B1: information boundaries и capacity

Paper reference: RMS-normalized state variation Eq.13, threshold, soft train/hard inference; information-density merge соседней пары с минимумом (I_left+I_right)/(n_left+n_right). Это не key-distance и не mean β. [DLA, Eq.13–17](https://arxiv.org/html/2606.10650v1).

Проектный bank score: I_t=||RMS(S_candidate)−RMS(S_prev)||F/(||RMS(S_prev)||F+eps). S_candidate получен tentative continuation до reset. Score агрегируется по явно выбранным Ha heads (по умолчанию RMS по head scores); boundary единая на sample/layer для общих spans. Information accumulation хранит raw sum I и count. min_tokens≤max_tokens; max force-close, min forbids except explicit flush/end. Fixed threshold baseline, затем soft boundary train и hard inference; ST как отдельный estimator с собственными gradient tests, не finite-difference hard decision.

Soft-train и hard-inference могут расходиться: это измеряемый train/decode gap. Training full carry и memory budget объявляются; ST/soft не подписываются как exact gradients hard segmentation.

### 7.2 B2: два честно названных пути

**loglinear-paper:** независимый reference по статье, disjoint prefix bins и structured recurrent transitions; затем интегрированный production GDN-2 asymmetric stream adapter с отдельным выводом reset/transition/read rules и независимым oracle. O(log L) live states допустим только для реального streaming algorithm. Полный Fenwick node per bank сохраняется лишь как offline diagnostic. [Log-Linear Attention, §3](https://arxiv.org/html/2506.04761v3), [author implementation](https://github.com/HanGuo97/log-linear-attention).

**bank-hierarchy-v2:** выбранная проектная память GDN-3. Это логарифмическая организация независимых frozen summaries; не заявление о точной реализации recurrent Log-Linear Attention. Она сохраняет масштабные spans, но merge теряет информацию.

Порядок bank-hierarchy-v2:

- Recent queue ограничена K_recent закрытыми банками и token-age horizon W_recent. На переполнение по count сливается adjacent minimum-density pair. Это освобождает место, сохраняя chronological coverage.
- Банки старше horizon продвигаются из начала recent queue в hierarchy; promotion и recent membership взаимно исключаются.
- Один продвинутый bank — одна **promotion unit**, даже если он объединяет разные числа token segments. Binary carry: свободный level получает bank; занятый level объединяет два chronological neighbouring blocks и переносит результат вверх. Level ℓ покрывает 2^ℓ promotion units; token span хранится отдельно, не считается равным степени двойки.
- Живых hierarchy bins ≤1+floor(log2(1+N_promoted)); recent ≤K_recent. При продвижении нет полного history array, archived parent/children не остаются скрытой live памятью.
- Default merge независимых S: count-weighted mean на общей coordinate/clock, с n/I/span metadata. Это проектная lossy summary; не delta recurrence union и не paper additive merge. Mean ограничивает искусственное увеличение constant h_init; при сложной init dynamics всё равно нет exact union.
- Descriptors/router keys пересчитываются из нового представления; удалённые child IDs не доступны по старым indices. Частичное чтение внутри merged span не выдаётся как exact — это whole-summary read либо external span retrieval.
- Новый level создаётся при binary overflow. Host capacity растёт геометрически, copying/JIT recompilation фиксируются как события. max_levels используется только в bounded profile с явной overflow policy. Неограниченный profile не clipping-ит level к последнему trained bucket.

Coverage invariant: каждый закрытый token span присутствует ровно в одной live recent/hierarchy summary; ни gap, ни duplicate. Active bank отдельно. HOLA содержит копии selected tokens намеренно и является другим read branch.

### 7.3 Exact affine merge и passive decay

Для двух chronological blocks с S_out=P S_in+U: **P_ab=P_b P_a, U_ab=P_b U_a+U_b**. Это точная композиция в арифметике reference. С одинаковым h_init: S_ab=P_b(S_a−h_init)+S_b. Поэтому simple sum S_a+S_b и decay-only transport не воспроизводят union при erase/nonzero initialization.

Exact affine diagnostic сохраняет P[Dk,Dk] и U[Dk,Dv]; dense composition стоит O(Dk³+Dk²Dv), low-rank products накапливают rank. Он пригоден для oracle и исследования, не получает цену O(DkDv) автоматически. Эффективный loglinear adapter должен выводить structured transitions отдельно. После сжатия P/U exact guarantee снимается.

### 7.4 B6: clock contract

Frozen default: закрытый bank не подвергается последующему passive decay; age penalty router — отдельный параметр. Aligned optional: read_t(q,S_end)=(q_clean⊙exp(G_t−G_end))ᵀ S_end; включает current token g, исключает future erases. Merge переносит operands к общему end clock до summary operation. Это passive alignment, не точное продолжение delta dynamics. Причина выбора frozen default — сохранение дальних summaries без обязательного exponential fading; качество обоих modes измеряется.

Clocks локально rebase-ятся с сохранением differences; growing абсолютные ID хранятся host arbitrary integer/многословным counter и version. int32 totals не допускаются как lifetime clock. Decode JIT использует bounded relative ages/multiword counters. Относительные counters тоже имеют объявленную overflow/rebase policy. Float saturation/underflow диагностируются до sanitize.

## 8. Router, B12 и C1

### 8.1 Exact selection

Candidate IDs — только live recent/hierarchy spans. Score вычисляется из описанного descriptor с отдельной age term. top-k задаётся **indices**; tie-break по стабильному bank ID, null кандидат всегда доступен. Для n<k выбираются n валидных; n=0 даёт null output. Dense/reference и sparse production имеют одинаковые normalization и selected mass. Score ALL history не допускается в log-memory профиле.

Default гибридное чтение: router probability p_j over selected candidates + null; archive output = Σ_j p_j λ_level(j) q_bank,jᵀ S_j. Recent λ=1 либо отдельная learned local λ. λ не подменяет p_j и не является age penalty; степень влияния каждого branch логируется. Этот combined read — проектная адаптация, paper λ-only reference идёт отдельно.

### 8.2 B12: Adaptive λ

Paper Eq.4: d_t baseline projection H×L_levels; h=d_t W1; λ=softplus(h W2+b), W2=0, b=.54. Фиксированный axis L требует конечного maximum. В экспериментальном описании упомянут GELU, которого нет Eq.4: сохраняем **paper-eq4** и **paper-gelu** отдельно. [Adaptive Memory Decay](https://arxiv.org/html/2605.06946v1).

Проектный unbounded-level adapter: shared MLP по (x_t, head embedding, level/span features) → softplus λ_hℓ, одинаковые weights для любых ℓ. Level features — описанные bounded/log-periodic encodings и logspan; не learned table с silent clipping. Новая level не создаёт новые model parameters. Zero last layer/bias .54 — baseline-init λ≈1. Экстраполяция качества остаётся экспериментальной. Уточнить numeric domain очень больших logspan и feature rebase policy.

### 8.3 C1: CCQ

Paper: μ=sum k/n, C=sum kkᵀ/n, Σ=C−μμᵀ, q_clean=(I−λΣ)q, λ=sigmoid(Wλ q+bλ). Практическая версия — Welford count/mean/M2; Σ=M2/n, не n−1 и не decayed covariance. Current main raw normalized k включается причинно; DP extras не считаются отдельными tokens по умолчанию. SSᵀ не является key covariance. [CCQ, Eq.8–9](https://arxiv.org/html/2606.01294v2).

Paper reference меняет recurrent read. Default GDN-3 adaptation применяет CCQ только к archive branch, сохраняя gate=0 baseline; это отдельный scope. Порядок — clean raw q, затем bank passive transport, затем read. QED/EDA/OSDN write использует raw q. HOLA/external exact branch имеет собственный q norm; CCQ туда не распространяется автоматически. Численная PSD/conditioning диагностика обязательна; clipping eigenvalues помечается как стабилизированный вариант.

## 9. B4/B5/B8 — дополнительная ёмкость и важные токены

### B4 — SSE

Paper references разделяют class sparse write и partition expansion. Реальная ёмкость — N independent partition states, а не reshape старого S. Общее q/k/v projection; hard partition selection, soft row weights внутри выбранной partition, shared always-on partition, write/read gates и balancing/bias. [SSE](https://arxiv.org/html/2507.16577v2).

Проектный adapter: SSE расширяет **active archive bank**, baseline cumulative остаётся неизменным; банк/summary содержит partition axis. Выбранные partitions записываются через основной cell ABI с soft class key weights; В paper Eq.13 невыбранные partitions остаются полностью неизменными, включая отсутствие decay. Проектный default inactive=freeze сохраняет это правило; optional inactive=passive_lazy применяет только passive decay без erase/write и имеет отдельную проверку. Lazy decay использует timestamp на partition и считается в carry/accounting. SSE+QED/OSDN требуют вывода норм/addresses: row sparsity может нарушить unit-k/QED projector. До этого совместный preset запрещён. N=1 в paper не обязан совпадать с GDN-2; feature-off bypass обязан. N и top-k фиксированны для complexity statement; рост N учитывается как реальный рост памяти.

### B5 — learned initial state

h_init[Ha,N,Dk,Dv] только у independent bank по default. Оно применяется на каждом bank reset; cumulative zero/imported initialization не меняется. В paper SDM этот элемент часть большей sparse архитектуры; GDN-3 не объявляется реализацией полного SDM. [Sparse Delta Memory](https://arxiv.org/html/2607.07386v1).

Snap scope и whole-cumulative learned init задаются отдельными variants. Gradients суммируются по resets; serialization содержит init version. Mean merge даёт собственную summary semantics, не точный union. Для exact affine reference нужен initialization correction §7.3.

### B8 — HOLA

Paper: priority top-w наиболее surprising tokens **среди всех seen tokens**, по β||residual|| при unit k; raw exact KV + отдельные learned RMSNorm γ для cache q/k, causal current-block и null sink. [HOLA](https://arxiv.org/html/2607.02303v1).

GDN-3 adapter: score=Σ_slots ||outer(u_i,residual_i)||F=Σ ||u_i||₂||residual_i||₂ для committed write steps; erase-only шаги исключаются. Это declared generalized-write score, не полный physical ΔS с decay. После b/w residual не умножается ещё раз на mean β. Stored payload по default raw projected k/v + token ID/span/score, не повторно weighted v. Выбор vector-weighted values допустим отдельным вариантом.

При заполнении admission заменяет минимальный score только если новый score выше; tie по stable token ID. Current-block cache имеет ограниченный размер, causal mask и flush policy, не хранит весь active DLA block без bound. При выходе токен может остаться в priority set. Dup IDs между current/priority не удваивают softmax. HOLA сохраняется при ordinary bank close и очищается только per-sequence reset. Ограниченный w не гарантирует любой старый факт.

## 10. D1/D2/D3 — сжатие и расширение

### D1 — LeapQuant

Training-free inference path: на window boundary выделить low-rank compensator, вычесть его **до** quantization residual, применить smoothing/scales; lowbit boundary state + full-precision token update buffer представляют recurrent state внутри window. На новой boundary refit/quantize, старые compensators заменяются. Simple archive_int8 остаётся отдельным baseline compression, не называется LeapQuant. [LeapQuant, §3](https://arxiv.org/html/2609.38166v1).

Для asymmetric/substep GDN-2 нужен вывод transformed update factors/window Γ и independent recurrence comparison. Arbitrary DLA close посреди quant window — forced flush с явным error/latency. Frozen closed bank может хранить quantized residual+compensator; merged operands dequantize, merge, refit, requantize. Считаются все lowbit/factors/scales/buffers/temp arrays и peak bytes. No-op не сдвигает window. Default training quant off; STE вариант имеет собственный контракт и не получает training-free утверждений статьи.

### D2 — OjaKV

Paper reference относится к Transformer KV: crucial tokens сохраняются full precision, другие — online low-rank basis; update/orthogonalization и координаты старых vectors согласованы. [OjaKV](https://aclanthology.org/2026.findings-acl.494/).

В GDN-3 primary integration — optional **working KV retrieval cache** persistent profile; original external spans и HOLA exact payload не сжимаются. Basis carries separate key/value normalization/statistics, rank/anchors и basis version. Rotation требует remap старых codes или block-specific basis; запрет silent reuse новых coordinates для старых codes. Bank Oja compression — separate research variant, потому что paper не задаёт delta-state representation. rank=D с disable loss проверяет fallback; fixed-anchor capacity/accounting честные. Нет общей гарантии 4–8× для всей модели.

### D3 — StateX

Paper GLA расширяет state слиянием голов; q/k/v total widths могут сохраняться, cross-head blocks увеличивают capacity. Mixer переинициализируется, FFN/embeddings сохраняются, затем post-training. Это не zero-pad migration без обучения. [StateX, §4](https://arxiv.org/html/2509.22630v2), [ACL версия](https://aclanthology.org/2026.findings-acl.1073/).

Обязательные режимы будущего проекта:

1. **paper-reinit reference** для поддержанного backbone и GDN-2 head-merge adaptation. Whole-model schema пересчитывает H/Ha/Dk/Dv, norms/scales/projections/conv/output/router/caches/optimizer. GDN-2 vector b/w и KDA block-diagonal/merged gates получают отдельный вывод. Initial logits не обязаны совпадать. Доступ к cross-head state и gradients после initialization проверяется, adaptation budget учитывается.
2. **preserve-compatible optional migration**: заявляет parity только для доказанного shape/norm/output transformation; например block-diagonal heads при сохранении per-subhead norms. Дополнительные directions активируются ненулевой асимметричной initialization/gating по явно описанному schedule. Если перенос не доказан, migration отклоняется, а не обещает logits preservation.

Zero-pad одной W/h не закрывает D3. Unsupported checkpoints не модифицируются частично. Expand selected layers; config/checkpoint/optimizer/cache version обязательны. Обучение расширенной модели — отдельный будущий эксперимент, не действие шага 2.

## 11. A6/A7/A8 — отложенные расширения

Сохранить отдельные reference/test tasks, не смешивать с mandatory default:

- **A6 FG²-GDN:** Eq.11 symmetric erase с k_tilde=sqrt(β)⊙k; отдельные β_k/β_v варианты по статье. Ориентация и Dk/Dv restrictions фиксируются. Это другая cell family, не молчаливая замена asymmetric GDN-2. [FG²-GDN](https://arxiv.org/html/2604.19021v3).
- **A7 Q-Delta:** в нашей orientation scalar paper recurrence S=α(I−β k(k+λq)ᵀ)S_prev+β k vᵀ, λ∈[0,1]. ABI u=βk, r=k+λq, z=v. Write обязателен как outer product. Это query-feedback method, отличается от ортогонального QED. Generalized b/w adapter требует отдельного вывода. [Q-Delta, Eq.17–20](https://arxiv.org/html/2606.08804v1).
- **A8 Preconditioned DeltaNet:** exact-Gram theoretical reference: tilde_k=P_prev k/(1+kᵀP_prev k), write outer(tilde_k,v−S_prevᵀk), Sherman–Morrison P. Paper practical reference Eq.9: A_t=αP_t A_prev+βP_t(k⊙k); ell=log(A_t)−μ, s=ell/(1+|ell|), B_t=exp(−log(x)s), tilde_k=B_t⊙k; scalar write β outer(tilde_k,v−Sbarᵀk). Positive A initialization, μ, αP/βP, x≤2 paper domain и decay timing фиксируются. Это не OSDN; exact inverse, diagonal normalized Eq.8 и squashed Eq.9 проверяются отдельно. [Preconditioned DeltaNet, §3](https://arxiv.org/html/2604.21100v1).

SWA/MLA/Liger/NHA/MSGLA/SSA/SubQ остаются вне шага основной архитектуры; Triadic/MoM/Raven не возвращаются. Отложенные модули не должны блокировать критерий готовности 16 core, но их формулы и точки расширения предусмотрены.

## 12. Persistent profile: точность хранения и retrieval

Проектное дополнение для arbitrary-history goals; не модуль, приписываемый какой-либо статье.

- Durable append-only original token/document chunks со stable sequence ID, absolute span, content hash, tokenizer/schema version. Индекс versioned и обновляется только после commit; transaction cursor включён в inference cache.
- Сохранение повторного факта/overwrite связывает версии и causal timestamp. Для «последнего значения» retrieval учитывает latest visible version, не только similarity. Delete имеет tombstone и отдельную privacy/storage policy без удаления оригинала по неявной команде.
- Causal query содержит только доступные model features/явный query text. Gold fact ID используется лишь privileged oracle diagnostic, не ordinary path.
- Tiered index на disk; bounded candidate KV cache. Exact ID lookup, approximate semantic search и exhaustive diagnostic — разные режимы с разной гарантией. Retrieval recall@k измеряется до answer generation; при candidate miss log содержит причину.
- Copy/hash integrity означает exact storage, но не perfect model recall. Raw text остаётся доступен для exact replay/inspection; сжатые Oja embeddings не заменяют оригинал.
- I/O и disk bytes, index working set/build/update cost, fetch latencies и failures учитываются отдельно. При невозможности durable append транзакция останавливается/возвращает ошибку, а не продолжает с ложной exact гарантией.

## 13. Complexity, backend и precision

Пусть B batch, H cumulative heads, Ha archive heads, Np fixed SSE partitions, M=max substeps, Dk/Dv widths, K recent slots, W HOLA tokens, Cq quant window, r quant/Oja rank, J=O(log(1+N_promoted)).

| Путь | Рабочая память inference | Стоимость на token / событие |
|---|---|---|
| Cumulative + active bank | O(B(H+Ha Np)DkDv), substep scratch отдельно | O(BM(H+Ha Np_selected)DkDv); lazy decay costs объявляются |
| bank hierarchy | O(BHa Np(K+J)DkDv) + descriptors/metadata | Router O(K+J) descriptors; selected read O(k Ha Np DkDv); merge worst-case O(J Ha Np DkDv), amortized binary promotions O(1) merges, age/info work отдельно |
| loglinear-paper adapter | O(log L) live structured states при корректном алгоритме | O(L log L) общий attention work; actual constants/D costs подтверждаются реализацией |
| CCQ | O(BHa Dk²) stats | O(BHa Dk²) stat update + cleaning |
| HOLA | O(BHa(W+C_block)(Dk+Dv)) | O(W+C_block) bounded selection/read, GPU priority implementation отдельно |
| LeapQuant | Lowbit + rank factors + window buffers + temps | Flush/refit и DLA-close latency отдельно; общий model compression не равен state compression |
| Exact affine oracle | Дополнительные P[Dk²] per block | O(Dk³+Dk²Dv) dense merge |
| External history | O(L) disk + индекс; working set отдельно | Lookup/search/fetch; сложность зависит от выбранного индекса |

Иерархический режим не имеет строго линейного общего T времени при чтении O(log T) bins. Ключевой контракт — bounded/log working memory и измеряемые token/merge latencies. Независимые module speedups не перемножаются. Total live/peak counts включают descriptors, float copies, convolution, counters, fullprec retained tensors и allocated level capacity.

**Backend tiers:** independent NumPy float64 oracle → pure JAX serial → chunked/parallel JAX → Pallas TPU. Runtime поддерживает один mathematical ABI. Force unsupported combination = ранняя ошибка; auto fallback выдаёт reason/effective backend. Missing TPU → SKIP, не PASS.

PrecisionPolicy отдельно содержит storage dtype, accumulation dtype, dot algorithm highest/bf16x3/default, WY eps, clipping mode и domain checks. Router high не считается bf16x3 автоматически. g=0 derivative корректна; clamp forward требует соответствующего VJP либо explicit surrogate. NaN/overflow health фиксируется **до** sanitize; numerical rescue не сертификат здоровья. asymmetric kernel не пользуется доказательством symmetric stability без вывода. [Pallas](https://docs.jax.dev/en/latest/pallas/index.html), [custom_vjp](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html).

Каждый новый primitive получает production kernel/VJP либо явный JAX fallback. Composite fusion допустима после reference, но requirement «Pallas реализован» закрывается только при реальном аппаратном запуске и сохранённых результатах.

## 14. Совместимость, training и доказательства

### 14.1 Минимальные пресеты

| Preset | Включение | Статус до реализации |
|---|---|---|
| gdn2-baseline | A1 и исходный gate; всё extra off | Обязателен exact parity |
| gdn3-stream | A1 + B1 + bank hierarchy + B5 + frozen B6 + B8 + shared B12 | Основной проектный профиль; качество неизвестно |
| cell-ablation | Baseline + один A2/A3/A11/A4/A5 | Формула и полезность проверяются отдельно |
| archive-ablation | gdn3-stream + B4 или C1 | Объявленный scope и off parity |
| compress-ablation | Валидный FP stream + D1; persistent KV + D2 | Ошибка/ресурсы измеряются отдельно |
| statex-adapt | Whole-model D3 migration + отдельное post-training | Нельзя заявить готовность по helper |
| full-composition | Все 16 модулей в совместимых adapters | Только после composition derivation и T41; не default |

Core implementation scope включает все 16; каждый имеет независимый reference, адаптер, конфиг, интеграцию и тесты. Полная архитектура означает завершённые modules и совместимые presets. Она не означает включение всех механизмов без проверки. QED+OSDN, SSE+QED, SSE+OSDN, CCQ+aligned, quant+multi-step, StateX+expanded partitions — обязательные ячейки capability matrix с условиями/выводом, а не silent exclusions. Если derivation обнаружит принципиальную несовместимость, записать точный конфликт и отдельные valid presets.

### 14.2 Training/checkpoints

Full BPTT/remat и truncated BPTT имеют разные gradient semantics. В checkpoint входят params/optimizer/scheduler/step/RNG/dataloader cursor/EMA/skips/health/config/source hashes. Inference cache имеет schema version, complete carry и durable external cursor. Resume обязан воспроизводить declared continuity. Неизвестная config/model shape — ранняя ошибка или именованная migration.

Обучаемые gates/theta/λ/router/SSE imbalance regularization/h_init и module learning-rate policies явно перечисляются. paper StateX включает adaptation tokens/compute, а не «бесплатное расширение». Kernel/bootstrap dependencies закрепляются после выбора совместимого hardware/runtime; шаг 2 не выдумывает неподтверждённые version pins.

### 14.3 Уровни готовности

1. SPECIFIED — настоящий текущий статус v2.
2. IMPLEMENTED — production path и carry подключены, config не игнорируется.
3. REFERENCE-VALIDATED — independent formula + state/gradient/timing checks прошли.
4. BACKEND-VALIDATED — аппаратный backend/precision подтверждён отдельно.
5. STREAM-VALIDATED — произвольные chunking/reset/resume/level growth без semantic drift.
6. QUALITY-VALIDATED — paired healthy baseline, finite long lengths, uncertainty и iso-resource controls.

M0/M1 из старых документов не объявляются закрытыми. Universal infinite lossless сертификат не вводится.

### 14.4 Будущие тесты и эксперименты

Подробные T01–T37 остаются в аудите, с исправленным D3 contract. [TODO v2](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_TODO.md) дополняет T38–T48. Reference не импортирует production cells. Проверяются outputs/final carry/gradients всех continuous inputs/h0/dh_final; hard decisions и ST surrogate — отдельно. При eps/clipping/quant сравнивается соответствующая именованная функция, не разные recurrences.

Experiment acceptance: short positive control; long retention и distributed facts/overwrites; causal learned router отдельно от oracle; минимум 3 paired seeds для capacity, censoring P-grid и uncertainty; same data/tokens/schedule; iso-state/iso-params/iso-runtime. Speed — stateful chain с full new carry, filled valid cache и synchronization, close/merge/quant/level-resize events включены. Все результаты содержат hardware/runtime/config/checkpoint/source hashes, live/peak memory, p50/p95/p99/event latency, health до sanitize и PASS/FAIL/SKIP/INVALID.

## 15. Порядок следующих шагов пользователя

**Шаг 2 завершён документами:** источники, архитектура, полный TODO и статус. Код не создавался и не исправлялся.

**Шаг 3, только после команды пользователя:** генерация/переделка package по этому backlog, исправление baseline/atomic, затем memory/modules/persistence/migration, добавление независимых references и тестовых сценариев. Наличие файлов не закрывает validation.

**Шаг 4, только после команды пользователя:** запуск тестов сначала correctness/streaming, затем Pallas на совместимом устройстве, обучение и long-context/ресурсные experiments по согласованному доступному бюджету. Условия среды и missing hardware явно отражаются в результатах.
