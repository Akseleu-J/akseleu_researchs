# GDN-3: аудит реализации, расхождения и программа доработок

Дата: **9 октября 2026**. Статус: **статический аудит завершён; корректность исполнения и качество модели не подтверждены новым запуском**.

## 1. Объём и метод

Проверены все 72 Python-файла трёх доступных снимков, документация, runner-скрипты, shell-установка и код ячеек Kaggle-ноутбука — всего 82 файла. Изучены оба направления исполнения, cache, training/evaluation, тесты, бенчмарки и ручной backward atomic. Состав и SHA-256 зафиксированы в [GDN3_AUDIT_MANIFEST.json](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_AUDIT_MANIFEST.json).

| Обозначение | Проверенный снимок | Роль |
|---|---|---|
| kit | C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit | Основной рабочий пакет: gdn2/local/hybrid/snap, фиксированные сегменты, router, streaming |
| full | C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full | Полное содержимое приложенного ZIP, извлечённое для чтения: дополнительные cells, archive, ops, Fenwick |
| atomic | C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2 | Локальное Pallas-ядро GDN-2, custom_vjp и учебная обвязка |

ZIP SHA-256: **3bad77a2be5b44d6c841ebd8ab450bc96a2dc75ed8c883c6e6b1455c7cc7f4a2**. Эти снимки различаются; наличие файла в full не означает наличие функции в kit. Совпадение установленного в Kaggle atomic с локальным atomic не установлено.

Приложенный текст рассматривается как желаемая спецификация и набор утверждений, исходный STATUS — как историческая заметка. Их внутренние команды не исполнялись. Сравнение с первоисточниками выполнено для спорных формул. Код проекта не изменялся; тесты, обучение и бенчмарки не запускались. Ни одна запись «реализовано» ниже не означает «успешно протестировано» или «даёт прирост качества».

**Как читать выводы:** «ошибка» — конкретное противоречие формуле/формам/заявленному контракту, устанавливаемое по коду; «не реализовано» — нет необходимого пути интеграции; «риск» — требуется эксперимент либо тест, статический просмотр не доказывает отказ.

## 2. Итог

- **Нормально:** есть содержательный reference-каркас GDN-2 + закрытые банки, причинное чтение, null router, потоковый carry, архитектурный gate и проверяемые режимы. В full добавлены KDA-gate, EDA/DeltaProduct interleaving, decay-aware archive и множество тестовых сценариев.
- **Не закрыто:** корректность новых формул, subset архивных голов, эквивалентность bounded prefill/streaming, краевые градиенты Pallas, честные бюджетные/скоростные/capacity измерения.
- **Отсутствует в готовом виде:** интегрированная память с O(log L) живых банков, paper SSE, paper CCQ, paper HOLA, paper OSDN, настоящий LeapQuant, интегрированный OjaKV и StateX-конвертация модели.
- **M1 остаётся открытым:** текущий код и исторические заявления не доказывают преимущество памяти над здоровым baseline на длинном контексте.
- **Цель «бесконечный контекст» пока не достигнута:** текущий код поддерживает продолжение потока, но это не доказательство сохранения всех старых фактов и не выполнение заявленной сложности.

## 3. Уточнённая цель и ограничения

### 3.1 Три разных свойства

1. **Неограниченное продолжение потока:** модель принимает новые токены без заданного максимального L, carry сохраняется и восстанавливается, счётчики не переполняются, память/время соответствуют выбранной политике.
2. **Долговременная полезная память:** на заранее описанных задачах качество остаётся в пределах ε от опорного уровня на всё больших конечных длинах. Это эмпирическая цель, зависит от плотности фактов и объёма памяти.
3. **Точное сохранение произвольной истории:** любой независимый факт можно восстановить после произвольного числа новых фактов. Для этого нужна растущая ёмкость хранения; небольшой bounded cache и lossy merge общей гарантии не дают.

При фиксированной точности пусть история содержит M независимых битов, а сохраняемое состояние — b битов. Для точного восстановления любой истории необходимо различать 2^M вариантов; при b<M минимум две истории дают одинаковое состояние. Поэтому универсальное exact recall произвольных данных при памяти O(log L) бит невозможно. Это вывод подсчёта состояний, а не результат бенчмарка. Бесконечно точные действительные числа не описывают JAX f32/int8 реализацию.

**Предлагаемый целевой контракт GDN3-Inf:** непрерывный streaming + ограниченный рабочий recurrent state + O(log L) живых summaries + отдельный exact/archive storage при требовании произвольного recall. Если внешний архив исключён, задача — измеряемое приближённое сохранение важной информации; обещание «без потерь для любых данных при L→∞» нужно снять.

### 3.2 Стоимость текущего кода

Для N банков, ширины скетча m, размера головы D, top-k=k (без множителей batch/heads/layers):

| Путь | Фактическая работа | Основание |
|---|---|---|
| kit parallel archive | O(L·N·D·m) router + O(L·N·D²) чтение; dense scores O(L·N) | [kit/gdn3/read.py:15](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/read.py:15), [kit/gdn3/read.py:35](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/read.py:35) |
| full parallel archive | Та же зависимость от всех банков, несмотря на top-k веса | [full/gdn3/mixer.py:36](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:36) |
| kit/full decode retrieval | O(N·D·m + k·D²), при exact index O(N·D²) | [kit/gdn3/cache.py:114](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/cache.py:114), [full/gdn3/archive.py:116](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/archive.py:116) |
| full decode metadata | При bank_decay обновляет acc всех N банков | [full/gdn3/mixer.py:191](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:191) |
| archive storage | O(N·D²), плюс T и метаданные | [full/gdn3/archive.py:20](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/archive.py:20) |
| standalone Fenwick | O(log N) посещений на запрос, **O(N) хранимых узлов** | [full/gdn3/fenwick.py:26](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/fenwick.py:26), [full/gdn3/fenwick.py:43](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/fenwick.py:43) |

При фиксированном C и N≈L/C parallel-archive становится O(L²/C), а не O(L) или O(L log L). Нулевой softmax-вес не отменяет уже вычисленный matmul. Постоянный max_banks даёт ограниченную по L стоимость ценой отбрасывания/сжатия фактов. Если живых банков O(log L), стоимость их полного обхода — O(log L), а не O(1). Размер D также нельзя опускать из практической оценки.

### 3.3 Что в приложенном плане нужно исправить

- QED сформулирован неверно относительно статьи: это изменение erase address единственного rank-one обновления, а не добавочное erase q⊥q⊥ᵀ. См. R03.
- B6 противоречит самой приведённой frozen-bank формуле qS и текущему DESIGN kit: отсутствие post-bank decay — смена выбранной семантики, не автоматически ошибка арифметики. См. R11.
- Prefix table/Fenwick с сохранением каждого исторического узла не даёт O(log L) хранения. Нужна политика ограниченного числа активных bins. См. R08.
- OSDN использует write-side scaling при неизменном residual; текущая привязка обеих сторон к одному k этого не обеспечивает. См. R04.
- Порог D/3, «физический предел 128–256», «learned==oracle ⇒ идеальный router», «гейт гарантирует не хуже», «30–300× экономия», «достаточно N_train=64 для N_infer=1000» не являются универсальными следствиями архитектуры. Нужны отдельные эксперименты.
- γ=0 даёт включение соответствующего baseline в класс моделей при одинаковых остальных параметрах. Оптимизация может изменить общий backbone; качество после обучения не гарантировано. Битовая идентичность требует отдельной проверки backend/dtype, а не allclose.
- HOLA с конечным top-w защищает выбранные важные KV; не устраняет информационную потерю всех остальных фактов.
- Статья об улучшении на конкретном backbone не доказывает тот же прирост на GDN-2 или совместимость всех 16 изменений. Оценки «5 строк», «низкая сложность» не учитывают backward, миграцию checkpoint, cache semantics и тесты.

## 4. Матрица требований из приложения

Обозначения: **есть** — основной механизм присутствует; **частично** — часть контракта/другая параметризация; **другой метод** — название совпадает, операция отличается; **нет** — интегрированного пути нет.

| ID | Требование | kit | full | Что реально надо закрыть |
|---|---|---|---|---|
| A2 | KDA per-channel gate | Частично: rates H×D, сигнал wd только H | Есть при decay_mode=kda | Независимый gate по каждому каналу; dh/da тест; guard |
| A3 | QED | Нет | Другой метод: два erase/delta substeps | Исправить спецификацию и erase address по статье; independent ref |
| A11 | EDA | Нет | Есть generalized GDN-2 interleaving | Зафиксировать b/w вариант, γ=0, независимые forward/backward |
| A4 | DeltaProduct | Нет | Есть generalized дополнительные шаги | n_h≥1, gates/decay/read timing, n_h=1 parity |
| A5 | OSDN | Нет | Другой метод: heuristic, обе стороны k*d | Точный hypergradient, write-only scaling, box projection |
| B1 | DLA + information merge | Нет | Частично: key-diff границы, oldest/level merge | Источник information score, обучение θ, low-info adjacent merge |
| B2 | Log-linear иерархия | Нет | Standalone Fenwick; leveled archive отдельно | O(log L) живых bins, router integration, dynamic streaming |
| B4 | SSE sparse write / partitions | Нет | Другой метод: top-k чтение строк | Классификатор записи, независимые состояния, read/write gates |
| B5 | Learned h_init | Есть для независимых банков | Есть для независимых банков | Scope H vs Ha; cumulative/snap не использует параметр |
| B6 | Post-bank decay | Нет: frozen archive | Есть в обоих engines | Согласовать с frozen design, точность merge и long-range underflow |
| B8 | HOLA | Нет | Частично: threshold FIFO ring | Top-w surprise, committed write score, RMSNorm cache read |
| B12 | Adaptive λ hierarchy | Нет | MLP есть, применяется к age penalty | Иерархические веса после B2; уровень ≠ age bucket |
| C1 | CCQ | Нет | Другой метод: SSᵀ/Frobenius | Running centered key covariance; causal stats |
| D1 | LeapQuant | Нет | Другой метод: int8 банков + exact entries | Window update buffer, low-rank compensator tokens, реальная память |
| D2 | OjaKV | Нет | Только Oja/SVD helper | Crucial anchors, online basis, low-rank cache integration |
| D3 | StateX | Нет | Только pad h/W helper | Преобразование всей модели/checkpoint, subset layers, parity |

**Инженерный контракт приложения также частичный.** atomic имеет Pallas + custom_vjp; новые full операции в основном pure JAX, не отдельные Pallas-kernel/custom_vjp. Layout на kernel boundary head-major соблюдается; внутри модели B,L,H,D допустим при явной конвертации. Полноценный finite-difference набор для новых модулей отсутствует. Router precision=high не является явным bf16x3 алгоритмом make_dot. Требования инвариантов нельзя считать закрытыми наличием domain.py.

## 5. Точные расхождения и дефекты

Приоритеты: **P0** — блокирует доверие к математике/выводам/обязательному режиму; **P1** — необходим для целевой архитектуры; **P2** — расширение/производительность после correctness. Это приоритет работ исследовательского проекта.

### R01 · P0 · full: subset архивных голов ломает hybrid scan

**Место:** [full/gdn3/mixer.py:164](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:164), [full/gdn3/mixer.py:199](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:199); размер A задаётся в [full/gdn3/mixer.py:143](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:143).

substeps построены для H голов; A имеет Ha голов. В token_update(A, subs) передаются несрезанные H-головные k/v/b/g/extras. При H=4, Ha=2 операции не broadcast-совместимы. При Ha=1 broadcasting может расширить результат до H, что нарушает форму carry lax.scan. config разрешает эти значения.

**Доработка:** выделить archive-substeps со срезом всех головных полей (включая d, gam/e и dp extras) и сохранить shape A=(B,Ha,D,D). **Тесты:** T01.

### R02 · P0 · kit: top-k по порогу выбирает больше k банков

**Место:** [kit/gdn3/routing.py:56](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/routing.py:56); повторное ограничение в [kit/gdn3/cache.py:134](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/cache.py:134).

sel=(masked>=kth) выбирает все совпадения. Для трёх равных scores и k=1 softmax распределяет массу по трём банкам. parallel/dense суммирует все три; sparse decode берёт только один и теряет часть массы. Нулевые и одинаковые банки делают ties естественными, а не экзотическими.

**Доработка:** выбирать ровно индексы top-k с детерминированной политикой ties и единым null. В full bank_gate это уже сделано ([full/gdn3/routing.py:37](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/routing.py:37)), но read-row masking остаётся threshold-based. **Тесты:** T02.

### R03 · P0 · A3/QED: код и приложенная формула не воспроизводят paper

**Место:** [full/gdn3/layer.py:142](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:142), [full/gdn3/ops.py:65](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:65).

Первичный QED сохраняет один rank-one оператор (I−k aᵀ); a=b⊙k+λ_h P⊥_k(b⊙q), с обучаемым неотрицательным λ_h. Это другое обновление, чем отдельное стирание γq⊥q⊥ᵀ. [QED, Eq. 6](https://arxiv.org/html/2608.13668v1#S3.E6).

Даже если принять additive-формулу приложения как отдельный эксперимент, код применяет произведение (I−k(b⊙k)ᵀ)(I−γq⊥q⊥ᵀ). В нём возникает дополнительный член **γ k[(b⊙k)ᵀq⊥]q⊥ᵀ**. kᵀq⊥=0 его не зануляет при векторном b. Например, k=(1,1)/√2, q⊥=(1,−1)/√2, b=(0.2,0.8), γ=0.5 дают ненулевой коэффициент −0.15 перед kq⊥ᵀ. Это аналитический контрпример, не запуск проекта.

**Доработка:** зафиксировать правильный QED-address reference; расширить интерфейс write/erase адресов либо доказать корректное преобразование в существующий kernel. Не подменять paper двумя erase substeps. Старый вариант можно сохранить как отдельно названную абляцию. **Тесты:** T03, T18.

### R04 · P0 · A5/OSDN: другой residual и heuristic вместо hypergradient

**Место:** [full/gdn3/layer.py:139](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:139), [full/gdn3/ops.py:98](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:98), [full/gdn3/mixer.py:202](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:202).

k*d используется и для outer-product записи, и для чтения/стирания. Paper меняет только write-side key, оставляя residual по исходному k. Там ∇_d h=−β[1−β⟨d,k²⟩]k²/||k||² и d_next=clip(d−η∇h,d_min,d_max); текущий сигнал −k·(h_new v_new) зависит от состояния и значения и ограничен лишь снизу. [OSDN, §4.1–4.2](https://arxiv.org/html/2605.13473v1#S4.SS2).

**Доработка:** independent scalar-paper reference; затем отдельно вывести адаптацию к GDN-2 с channel erase/write gates. Развязать storage key и erase/residual address. Ограничить d и η; повторно вывести stability guard. **Тесты:** T04, T18. η→0 при d=1 проверяет выключение эвристики, а не правильность OSDN.

### R05 · P0 · bounded prefill имеет другую семантику, чем streaming

**Место:** [kit/gdn3/cache.py:237](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/cache.py:237), [kit/gdn3/cache.py:100](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/cache.py:100); [full/gdn3/model.py:77](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/model.py:77), [full/gdn3/mixer.py:126](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:126).

Сначала parallel forward читает весь доступный архив, затем экспорт сокращает его до N. Ограничение не действовало во время prefix. При нескольких слоях это меняет не только старые logits, но и features/банки следующих слоёв. Заявление «оба engine идентичны» должно исключать budget/quant режимы либо их необходимо согласовать.

**Доработка:** одинаковая политика во время обработки каждого токена/блока; явный выбор budget-aware engine; auto не должен молча выбирать uncapped путь. **Тесты:** T05, T06.

### R06 · P0 · kit archive_policy не измеряет заявленный бюджет

**Место:** [kit/gdn3/evaluate.py:48](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/evaluate.py:48), [kit/bench/archive_policy.py:55](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/bench/archive_policy.py:55).

eval_mqar_stream делает prefill всей последовательности и оценивает его logits. При L%align=0 все logits получены uncapped forward до применения N/policy: результат не измеряет бюджет, хотя таблица подписана как capped. В full запросы вынесены в scan ([full/gdn3/evaluate.py:34](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/evaluate.py:34)), поэтому это конкретное улучшение, но prefix остаётся uncapped по R05.

**Доработка:** обработать всю историю с выбранным budget либо честно назвать протокол «uncapped encoding → capped query read»; выдавать оба результата отдельно. **Тесты:** T06: специально построенный факт должен исчезать при FIFO малом N.

### R07 · P1 · B1/DLA: неполная information-based память и неработающий θ в snap

**Место:** [full/gdn3/mixer.py:169](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:169), [full/gdn3/mixer.py:175](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:175), [full/gdn3/mixer.py:185](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:185), [full/gdn3/archive.py:91](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/archive.py:91).

«residual» — ||k_t−k_prev||, а не ||Δx||/state innovation; «beta» — среднее b, а не указанная норма. Архив не хранит information score, слияние выбирает oldest/level, не минимальную information density соседей. Information-guided segmentation/merging — механизм [DLA](https://arxiv.org/html/2606.10650v1).

В snap soft θ влияет только на hard condition с stop_gradient; surrogate bnd используется исключительно для hybrid-reset A. Поэтому градиент loss по θ для snap отсутствует/нулевой. В hybrid surrogate reset не маскируется admissibility min/max: ложный gradient возможен при запрещённой границе. Это требует явной политики surrogate, а не утверждения о derivative hard segmentation.

**Доработка:** выбрать information score; хранить per-bank information/density; adjacent merge; согласовать boundary-before-token; обучаемый surrogate с допустимой областью или признать θ фиксированным. **Тесты:** T07, T08.

### R08 · P1 · B2: самостоятельное Fenwick не решает O(log L) памяти

**Место:** [full/gdn3/fenwick.py:26](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/fenwick.py:26), [full/gdn3/fenwick.py:43](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/fenwick.py:43), [full/gdn3/fenwick.py:13](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/fenwick.py:13); alias [full/gdn3/config.py:7](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/config.py:7).

Один node добавляется на каждый bank; O(N) state. Host-index дерево не используется mixer/archive/router. policy=fenwick — alias leveled, а ops.fenwick_build — просто полная prefix table ([full/gdn3/ops.py:146](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:146)). Это три разные конструкции.

**Доработка:** выбрать log-linear active-bin hierarchy с ограниченным числом representatives на уровень и интегрировать чтение/запись/carry; вывести live-byte bound и число операций. Полное Fenwick допустимо как external historical index, с честным O(N) storage. [Log-Linear Attention](https://arxiv.org/html/2506.04761v3).

range=P_b−exp(Δspan)P_a ([full/gdn3/fenwick.py:53](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/fenwick.py:53)) может терять малый хвост из-за вычитания больших близких значений. exp≤1 не гарантирует numerical stability. Необходим прямой range traversal либо доказанный допустимый error budget. **Тесты:** T09, T10.

### R09 · P1 · B12: MLP возрастного штрафа вместо весов иерархии

**Место:** [full/gdn3/layer.py:66](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:66), [full/gdn3/layer.py:122](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:122), [full/gdn3/mixer.py:56](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:56), [full/gdn3/archive.py:126](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/archive.py:126).

Положительный MLP реализован, но его значения выбираются по floor(log2(1+age)) и входят в −λ log(1+age) router-score. Уровней живой иерархии нет. n_levels=0 тоже имеет конечный cap, основанный на max_banks либо 4096 ([full/gdn3/config.py:109](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/config.py:109)).

**Доработка:** определить λ как веса/decay реально выбранных hierarchical blocks, способ роста levels без изменения shapes модели, conditioning input и causal timing. Текущий age-penalty оставить отдельной абляцией. [Adaptive Memory Decay](https://arxiv.org/html/2605.06946v1). **Тесты:** T11.

### R10 · P1 · B4/SSE: разрежено чтение, записи и новых partitions нет

**Место:** [full/gdn3/routing.py:47](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/routing.py:47), [full/gdn3/ops.py:172](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:172), [full/gdn3/ops.py:179](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:179).

read_core зануляет query-координаты по |q_i|·||S_i||; запись состояния остаётся плотной. sse_partitions лишь reshape существующих columns: число elements не растёт. Нет class-based sparse write, bias/gates для независимых partition states. Helper сам не подключён к модели. Это не механизм [SSE](https://arxiv.org/html/2507.16577v2).

**Доработка:** сначала paper-reference с selector, write gate, read gate и partition storage; затем определить GDN-2-вариант и выключающий bypass. N=1 само по себе не доказывает GDN-2-equivalence для отличающегося backbone. Пороговые ties также оставляют >k строк. **Тесты:** T12.

### R11 · P1 · B6: decay реализован в full, но его смысл нужно выбрать

**Место:** [kit/gdn3/read.py:73](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/read.py:73), [kit/gdn3/cache.py:130](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/cache.py:130); [full/gdn3/mixer.py:72](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:72), [full/gdn3/mixer.py:191](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:191), [full/gdn3/archive.py:134](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/archive.py:134).

kit имеет recency в router, но не exp(G_t−G_end) в read. full применяет этот множитель по каналам; scan добавляет decay текущего токена перед чтением, что соответствует выбранной формуле. Это реализованная часть full.

Frozen independent bank может намеренно хранить устойчивый исторический след. Post-bank decay, наоборот, экспоненциально подавляет старый bank при постоянно отрицательном g. Ни одна из этих формул не равна точному переносу через все последующие delta erasures. Возвращать decay как «универсальное исправление памяти» нельзя.

**Доработка:** закрепить frozen / decay-aligned semantics и сравнить отдельно; для сохранения старых фактов не давать decay единственному exact/archive пути. Проверить cumulative float32 clock и подводку на очень больших расстояниях; rebase/relative metadata. **Тесты:** T13.

### R12 · P1 · merge не является точным объединением recurrent сегментов

**Место:** [full/gdn3/archive.py:56](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/archive.py:56); kit политики [kit/gdn3/cache.py:54](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/cache.py:54).

full складывает exp(span_new)S_old+S_new, учитывая только passive decay. Future erase operators не сохранены. Для independent банков это допустимая lossy superposition; для реконструкции исходной recurrence — не exact. При h_init≠0 банк S_new уже содержит начальное состояние, а перенос S_old тоже несёт его след; сравнение с единственным recurrent union требует удаления двойного initial contribution. Комментарий «exact when beta=0» нуждается в h_init=0/оговорённом affine-offset условии.

**Доработка:** описать точную семантику объединения; хранить span/start/end/init-offset и, если нужно, полный affine transition или признать сжатие lossy. Для leveled доказать invariant adjacency: код выбирает overloaded level, но объединяет i и i+1 без отдельной проверки их level ([full/gdn3/archive.py:100](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/archive.py:100)). Это пока риск инварианта, не установленный отдельный отказ.

**Тесты:** T08, T14 — nonzero h_init, vector decay, noncommuting erases, mixed levels, measured recall loss.

### R13 · P1 · B8/HOLA: threshold FIFO вместо top-w surprise

**Место:** [full/gdn3/mixer.py:208](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:208), [full/gdn3/ops.py:205](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:205), [full/gdn3/ops.py:231](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:231), [full/gdn3/layer.py:74](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:74).

Сейчас admitted tokens записываются в FIFO ring при sur>threshold. Старый наиболее важный token будет вытеснен очередным умеренным admitted token. Paper хранит top-w по committed write magnitude и exact KV, а cache q/k проходят отдельный RMSNorm с learned gamma. [HOLA, §3.3–3.4](https://arxiv.org/html/2607.02303v1#S3.SS3).

GDN-2 v_new=w⊙v−(b⊙k)ᵀh уже является committed innovation; его дополнительное умножение на mean(b) нельзя без вывода считать ||ΔS||. Для generalized step естественная величина ||k||·||v_new||; при EDA/DP требуется определить, какие substeps учитываются. Cached w*v вместо исходного v — также вариант, который нужно назвать и обосновать. Scalar exp(lgamma) при unit q/k не воспроизводит learnable channel RMSNorm. Current-token visibility/null sink отличается от paper block bookkeeping.

**Доработка:** priority storage со scores, exact-key/value контракт, committed innovation, cache-path norm, одинаковая causal/block семантика. **Тесты:** T15.

### R14 · P1 · C1/CCQ: covariance состояния отличается от covariance ключей

**Место:** [full/gdn3/routing.py:47](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/routing.py:47), [full/gdn3/layer.py:124](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:124), [full/gdn3/ops.py:248](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:248).

Вычитается λ·qSSᵀ/||S||². Это state-derived query contraction; статистики keys не собираются. Корректное равенство pre/post-read для этой же матрицы не подтверждает paper covariance.

Первоисточник найден: **arXiv 2606.01294**, а не placeholder 2608.xxxx. Он использует running **центрированную невзвешенную** covariance past keys. Поэтому даже комментарий routing.py про decay*C+kkᵀ не описывает выбранную в paper статистику. [CCQ, Appendix I](https://arxiv.org/html/2606.01294v1#A9).

**Доработка:** causal count/mean/centered covariance (например, Welford); определить global/per-bank scope и момент включения текущего k; sigmoid strength; сохранить state-derived вариант как отдельную абляцию. Undecayed covariance и decay-aligned variant — разные режимы. **Тесты:** T16.

### R15 · P1 · D1/LeapQuant: реализована простая квантизация закрытого банка

**Место:** [full/gdn3/ops.py:259](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:259), [full/gdn3/archive.py:42](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/archive.py:42), [full/gdn3/archive.py:20](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/archive.py:20).

Per-bank maxabs→int8 + exact top-k scalar entries не являются window-buffered recurrent LeapQuant с low-rank компенсаторами. Нет high-precision update buffer внутри окна и compensator token factors. Именно эти механизмы описывает [LeapQuant](https://arxiv.org/html/2609.38166v1).

Outliers остаются в maxabs scale даже после восстановления: остальные элементы теряют precision. Active h/A остаются float32. JL sketch T тоже float32 и при m=D/2 занимает 2D² байт на bank, против D² байт Sq; метаданные/точные выбросы дополнительно увеличивают память. Итоговый выигрыш не равен «4× весь cache».

В auto/parallel quant не действует на training reads, только export; в scan int8 cast даёт почти везде нулевую производную storage path (остаются другие пути через T/Sf/exact entries). Training-free режим должен быть явным; STE нельзя вводить молча как exact gradient.

**Доработка:** либо переименовать в archive_int8_outliers и измерить его честно, либо реализовать настоящий window/reference LeapQuant. Учесть все buffers и temporary dequantization в byte budget. **Тесты:** T17, T06.

### R16 · P2 · D2/OjaKV: helper не подключён к cache

**Место:** [full/gdn3/ops.py:287](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:287), [full/gdn3/ops.py:293](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:293).

Есть отдельное правило Ойи и offline SVD матрицы. Нет crucial full-fidelity anchors, адаптивного online basis, low-rank storage/read или compression policy в archive. Multi-column правило без ортогонализации не гарантирует устойчивый независимый basis.

**Доработка:** определить перенос token-KV метода на matrices/banks как отдельное исследование; anchor selection, online subspace, rank/error budget, reorthogonalization, basis rotation существующего cache. [OjaKV, ACL Findings](https://aclanthology.org/2026.findings-acl.494/). **Тесты:** T19.

### R17 · P2 · D3/StateX: padding helper не является расширением модели

**Место:** [full/gdn3/ops.py:304](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:304), [full/gdn3/ops.py:311](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:311).

Расширяются только h и одна W. Не меняются связанные q/k/v/b/w/g projections, conv buffers, output projection, norm, router sketch и config/checkpoint. При изменении D меняются scale=1/√D и нормы; одной zero-pad недостаточно для сохранения logits. Новые q/k координаты, одновременно нулевые, могут не включиться в динамику без отдельной инициализации/обучения.

**Доработка:** end-to-end migration выбранного subset layers с явным параметрическим overhead и точным preserve-output контрактом; warm adaptation при необходимости. [StateX, ACL Findings](https://aclanthology.org/2026.findings-acl.1073/). Наличие обученной atomic модели не означает совместимость checkpoint: atomic LM и kit/full отличаются параметризацией норм/проекций. **Тесты:** T20, T27.

### R18 · P1 · A2/KDA и B5/h_init: реализованы с ограничениями

**Место:** [kit/gdn3/model.py:16](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/model.py:16), [kit/gdn3/model.py:36](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/model.py:36); [full/gdn3/layer.py:42](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:42), [full/gdn3/layer.py:105](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:105); [atomic/layer.py:19](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/layer.py:19).

kit имеет per-channel learned rate, но input-dependent gate общий на голову. full decay_mode=kda даёт H·D projection и нужную форму отрицательного g; default=channel по-прежнему shared-head signal. В atomic локально g уже канален по projection, поэтому предположение «везде скалярный head decay» неверно. [Kimi Linear](https://arxiv.org/abs/2510.26692).

h_init есть для Ha независимых banks; cumulative h и snap начинают с zero. Это нормально при bank-only контракте, но не полная замена всех нулевых states параметром H×D×D. SDM-публикация сама по себе не доказывает «бесплатный выигрыш» от этого одного параметра. [Sparse Delta Memory](https://arxiv.org/abs/2607.07386).

**Доработка:** выбрать scope h_init; унифицировать gate-mode naming; подтвердить канал-независимость, градиент decay_a, reset/init propagation и checkpoint conversion. **Тесты:** T21, T22.

### R19 · P1 · A11/EDA и A4/DeltaProduct: generalized варианты требуют независимой проверки

**Место:** [full/gdn3/layer.py:146](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:146), [full/gdn3/layer.py:150](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/layer.py:150), [full/gdn3/config.py:30](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/config.py:30).

EDA имеет отдельную e-проекцию, erase перед main delta, decay один раз. DP имеет собственные дополнительные k/v/b, последовательные updates, shared w и read после последнего шага. Это полезная реализованная часть. Но vector b и отдельный w делают их GDN-2-адаптациями scalar-beta формул приложения; параметризацию надо описать. [EDA](https://arxiv.org/html/2606.26560v1), [DeltaProduct](https://arxiv.org/abs/2502.10297).

dp_extra<0 не отклоняется config: n_h=0 ломает interleaving/alignment. Для γ=0 нужно принудительное выключение, а sigmoid(logit) не даёт точный zero при конечном bias. cfg.cell выбирает ровно одну из qed/eda/dp/osdn: совместное включение усилителей пока отсутствует, хотя общий план подразумевает композицию.

**Доработка:** независимый matrix reference; выключающий bypass; валидатор n_h; явная таблица совместимости и порядка operators. Не обязательно включать все cells одновременно для M1. **Тесты:** T18, T23.

### R20 · P0 · atomic backward: запрещённая маска снова используется при g=0

**Место:** [atomic/precision.py:41](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/precision.py:41), [atomic/bwd/mega.py:24](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/bwd/mega.py:24), [atomic/bwd/mega.py:46](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/bwd/mega.py:46).

exp_nonpos возвращает mask=(x<0) и прямо предупреждает не занулять dgc этим mask при flat gc. mega всё ещё умножает вклад dgc3 и dgc_last на mgc/mkg/mdec. При x=0 mask=0, хотя допустимая левая производная exp(g) в g=0 равна 1. Простая recurrence с k=v=0 и nonzero h0/q сохраняет эту зависимость выхода от decay; full no-decay interleaving использует g=0.

На kink min(x,0) отдельно нужно определить принятую производную (JAX tie convention не обязана быть равна единице). Это не устраняет ошибку границы допустимого g≤0 и противоречие комментарию самого ядра.

**Доработка:** сверить весь ручной VJP с smooth/domain reference; исправление должно следовать объявленной convention и сохранению cancellation между legs. **Тесты:** T24, T18. Особое внимание dh_final и ненулевому h0; обычные случайные отрицательные g этого случая не покрывают.

### R21 · P0 для lean · atomic backward не учитывает clipping в diagonal lean

**Место:** [atomic/fwd/scores.py:31](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/fwd/scores.py:31), [atomic/bwd/b4.py:20](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/bwd/b4.py:20).

Forward использует exp(clip(diff,−diff_clip,diff_clip)), backward даёт dgc как для unclipped exponent. В области diff<−diff_clip derivative по diff должна быть zero, но возвращается ненулевой вклад. Такое diff возможно при валидном g≤0 и half_span<88 (например, diff=−30 при clip=20). b4_exact_f64_np имеет отдельный clip mask, значит reference и production формулы различаются.

**Доработка:** согласовать clipped lean forward/VJP и его роль reference. BTL не должен оцениваться как ошибочный только из-за отличия от специально clipped surrogate. **Тесты:** T25.

### R22 · P0/P1 · guards и sanitize не обеспечивают domain-of-correctness

**Место:** [atomic/domain.py:46](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/domain.py:46), [atomic/precision.py:51](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/precision.py:51), [full/gdn3/config.py:74](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/config.py:74); atomic pipeline использует kernels без обязательного вызова check_preconditions.

check_preconditions считает half-span; не проверяет фактические g≤0 и b·k². Автоматическое ограничение layer не защищает прямой kernel API и новые preconditioned cells. sanitize clip/nan_to_num может превращать нечисло в конечный zero: isfinite(loss/gn) после этого не гарантирует здоровое вычисление. VJP также не является точной производной всех активных forward clamps/sanitize.

**Доработка:** pre-sanitize health counters, debug strict checks, guard g/b/k/shape и derivation для asymmetric адресов. I−k(b⊙k)ᵀ имеет oblique geometry: eigenvalue bound сам по себе не гарантирует Euclidean contraction. Проверять norms на длинных потоках. Явно описать, где gradient exact, где surrogate, где reject domain. **Тесты:** T26, T24–25.

### R23 · P1 · WY epsilon меняет математическую recurrence

**Место:** [atomic/fwd/solve.py:1](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/fwd/solve.py:1), [atomic/fwd/solve.py:25](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/fwd/solve.py:25), [atomic/config.py:40](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/config.py:40).

С default wy_eps=1e−3 решается (I+(1−eps)Akk)⁻¹. Это отличная от обычной unregularized delta recurrence функция, даже в highest precision. Нужно отделять эффект stabilizer от rounding precision и kernel defects.

**Доработка:** correctness preset eps=0; production stabilized preset со своим reference и допустимой погрешностью. Не требовать bit-equality serial/Pallas, если формулы отличаются. **Тесты:** T24, T28.

### R24 · P1 · отдельные Pallas kernels / custom_vjp / precision не реализованы для новых модулей

**Место:** [atomic/pipeline.py:28](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/pipeline.py:28), [atomic/precision.py:27](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/precision.py:27), [full/gdn3/ops.py:1](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:1), [full/gdn3/routing.py:8](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/routing.py:8).

atomic custom_vjp есть; full extensions идут через pure JAX или interleaving общего ядра. Autodiff pure JAX допустим и полезен для reference, но это не выполнение требования «каждый новый модуль — отдельный Pallas kernel с backward». Router/read поддерживают default/high/highest; explicit bf16x3 не проходит через общий make_dot. Scan обычно жёстко highest, parallel использует config.precision: precision-конфигурация не является общей для engines.

**Доработка:** сначала независимые reference и VJP-контракт; затем kernels для измеренных bottlenecks, поддержка requested precision и явные tolerances. Отдельный kernel для каждого маленького математического блока не обязан быть наиболее быстрым: после reference возможна проверяемая fusion. **Тесты:** T28, T29.

### R25 · P1 · no-op padding обещан, но код padding ID — обычный токен

**Место:** [kit/gdn3/model.py:69](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/model.py:69), [full/gdn3/model.py:47](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/model.py:47).

Оба top-level forward действительно поддерживают произвольный L через pad/crop. Предыдущий STATUS ошибочно противопоставлял этому mixer alignment check; эту оценку исправляю. Проверка внутри mixer корректна для его входного контракта.

Однако embedding pad_id, conv и gates продолжают обновление: это не k=b=w=g=0 no-op из приложения. Причинность защищает logits реального prefix; want_cache для неполного alignment запрещён, а prefill идёт через настоящий tail scan. Поэтому pad-ID сам по себе не доказывает ошибку реальных logits, но обещанный no-op state контракт не реализован. Пустой prefill L=0 вызывает concatenate([]) в обоих пакетах.

**Доработка:** valid-token mask с явной политикой обновления recurrence/conv/position; либо документировать ограниченный pad/crop контракт и отдельный unpadded cache. Empty input, ragged batch и per-sequence reset должны иметь спецификацию. **Тесты:** T23, T30.

### R26 · P0 для сертификации · suite покрывает иные, часто совпадающие неправильные формулы

**Место:** [full/tests/test_gdn3.py:356](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/tests/test_gdn3.py:356), [full/tests/test_gdn3.py:432](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/tests/test_gdn3.py:432), [full/tests/run.py:5](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/tests/run.py:5), [full/tests/test_pallas.py:13](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/tests/test_pallas.py:13).

В kit 19 именованных test_* функций, в full test_gdn3 — 33. Это наличие сценариев, не PASS. В full Fenwick/Pallas оформлены main, отдельный run.py их не запускает; Kaggle runner вызывает отдельно. Обычный pytest-discovery не превращает main в тест. На CPU test_pallas возвращает 0 с SKIP: success процесса не равен TPU-сертификации.

Substeps-vs-ops сравнивает две реализации одной и той же неверной QED схемы. DLA assert isfinite(theta_grad) допускает нулевой gradient. Нет полноценных независимых finite differences, subset-Ha edge cases, budgeted multilayer prefill equivalence, flat-g atomic checks или actual complexity accounting. Выключение всех новых механизмов требует собственной матрицы тестов.

**Доработка:** общий обнаруживаемый runner с count/pass/fail/skip/backend/hash; независимые refs; усилить fixtures и краевые случаи. **Тесты:** T01–T30, T33.

### R27 · P0 для M1 · m1_verdict может принять нездоровый baseline

**Место:** [full/bench/_common.py:26](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/bench/_common.py:26), [full/bench/mqar_capacity.py:78](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/bench/mqar_capacity.py:78); P* [full/gdn3/evaluate.py:28](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/evaluate.py:28).

При base=[0,0,0], hybrid=[1,1,1] функция возвращает ratio=∞ и passed=True. Минимум seeds не проверяется; one-seed/zero-variance допускает «3σ» без оценённой дисперсии. В health-гейте нет positive control, baseline quality floor, skip/NaN budget или censoring по конечной P-grid. Oracle может попасть в такую же таблицу «M1 passed», хотя использует дополнительную истинную информацию. Независимая SE не использует парные seeds; вывод при дискретном P* требует осторожности.

**Доработка:** INVALID/INCONCLUSIVE/PASS/FAIL отдельно; healthy baseline first; paired protocol; ≥3 seeds как минимум, при слабой оценке CI больше; censored P*; learned-only criterion. Неудача всех ветвей не диагностирует автоматически bank-capacity limitation. **Тесты:** T31.

### R28 · P1 · MQAR/oracle не полностью проверяют долговременную ёмкость

**Место:** [kit/gdn3/tasks.py:23](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/tasks.py:23), [kit/gdn3/tasks.py:57](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/tasks.py:57), [full/gdn3/tasks.py:19](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/tasks.py:19), [full/gdn3/tasks.py:42](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/tasks.py:42).

Диапазоны key/value/filler разделены — это сделано правильно. Но все P фактов идут в начале. При C=128 и P≤42 они помещаются в один bank: рост L добавляет filler, а не независимую фактологическую нагрузку на каждый последующий bank. Это проверка retention отдельных ранних фактов, не всей ёмкости многобанковой системы. Запросы с replacement повторяются; teacher-forced предыдущий ответ попадает в causal history и может облегчить повторный вопрос.

kit oracle указывает bank первого токена факта; triple может пересечь boundary. full делит вес на оба банка, что не доказывает oracle upper bound для состояния, в котором association формируется после нескольких токенов. Оба включают bank истинного факта уже на SEP, до получения ключа текущего вопроса: это privileged oracle, не обычный causal content router. Маска j<seg не устраняет раннее раскрытие скрытой query identity. Для upper bound допустима привилегия, но протокол должен явно это назвать и исключить перенос oracle-trained качества в learned model.

**Доработка:** facts по всей истории/разной плотности, независимые query distances и distractor types, без повтора либо separate repeated-query split; факты на boundaries; oracle по effective committed representation после доступности query key или отдельный privileged ceiling. **Тесты:** T32, T34.

### R29 · P1 · speed измеряет неполный или несогласованный decode

**Место:** [kit/bench/speed.py:78](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/bench/speed.py:78), [full/bench/speed.py:49](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/bench/speed.py:49).

kit заменяет S/seg, но не пересчитывает T/Sf и count; router index относится к другим банкам. full передаёт пустой cache на каждый повтор; не проходит close/merge/quant событий. Оба timed wrapper возвращают только logits, игнорируя new cache: компилятор вправе убрать updates, не влияющие на текущий output. Повторы с одинаковым входным cache не измеряют непрерывный stateful decode.

**Доработка:** real populated consistent cache, возвращать/synchronize все updated leaves, stateful stream; regular-token/close/overflow/merge/quant latency отдельно, p50/p95/p99 + amortized; реальные live/peak bytes и scaling по N/L. Первая компиляция в kit включает первый вызов, её название стоит уточнить. **Тесты:** T35.

### R30 · P1 · resume не воспроизводит непрерывную тренировку

**Место:** [kit/gdn3/train.py:55](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/train.py:55), [kit/gdn3/train.py:78](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/train.py:78), [kit/gdn3/train.py:108](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/train.py:108); [full/gdn3/train.py:49](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/train.py:49), [full/gdn3/train.py:69](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/train.py:69).

Сохраняются params/optimizer/step/config, но не RNG state, gn_ema и skipped counters. Resume reseed(seed*7919+start) не продолжает исходную генерацию, EMA сбрасывается. Метаданные config не проверяются перед загрузкой. Изменение steps/warmup/lr между вызовами может также изменить schedule.

**Доработка:** versioned checkpoint schema с data RNG, optimizer/scheduler/health state, fingerprints/config; migration compatibility. Для inference отдельно сохранять carry, counters, conv, hierarchy/exact KV state. **Тесты:** T27.

### R31 · P1 · bootstrap/logging/exit codes могут создавать ложный успех

**Место:** [kit/gdn3/env.py:11](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/env.py:11), [full/gdn3/env.py:15](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/env.py:15), [kit/kaggle/run_all.py:54](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/kaggle/run_all.py:54), [kit/gdn3/train.py:81](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/train.py:81).

Автопоиск/clone ищет root/atomic_gdn2, а локальный package лежит в atomic_ops/pkg/atomic_gdn2; root/pkg не перебирается. Явный ATOMIC_GDN2_PATH к pkg работает, но свежий clone может иметь несовпадающую структуру. Clone без закреплённого commit и широкие зависимости не повторяют проверенный снимок.

kit runner добавляет rc лишь для Pallas-tests; serial/env/bench failures могут закончиться exit=0. Standalone mqar/lm передают log_path в ещё не созданную out-директорию, а train.open не создаёт parent; dump mkdir вызывается позднее. Kaggle заранее создаёт out — это объясняет различие standalone поведения.

**Доработка:** pin atomic/JAX/optax/runtime, record real module path/hash; учесть pkg layout; все subprocess rc агрегировать, обязательный correctness stage закрывать до capacity; mkdir перед log open. **Тесты:** T33, T36.

### R32 · P1 · validation config не закрывает допустимые формы и диапазоны

**Место:** [full/gdn3/config.py:74](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/config.py:74), [kit/gdn3/config.py:1](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/config.py:1), [atomic/config.py:1](C:/Users/omirb/Downloads/Downloads/atomic_ops/pkg/atomic_gdn2/config.py:1).

В full недостаточно проверяются positive dimensions/layers/d_conv, dp_extra, max_banks<0, max_levels, quant_k, min≤max для DLA, положительные half-lives/η/τ и допустимое precision. Необходимо также связывать actual gate bounds с bs2/kernel preset. N=0, L=0 и forced engine с неподдерживаемыми опциями должны давать документированный результат/ошибку, а не поздний reshape/KeyError. Unsupported combinations должны отклоняться до tracing.

**Доработка:** централизованный validated config + capability matrix, explicit runtime guards. **Тесты:** T23, T26, T29.

### R33 · P1 · длительность carry и память требуют отдельной инженерии

**Место:** [kit/gdn3/cache.py:95](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/cache.py:95), [kit/gdn3/cache.py:234](C:/Users/omirb/Downloads/Downloads/gdn3_kit/gdn3_kit/gdn3/cache.py:234), [full/gdn3/mixer.py:143](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/mixer.py:143), [full/gdn3/ops.py:235](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/ops.py:235), [full/gdn3/config.py:123](C:/Users/omirb/Downloads/Downloads/gdn3_kit/_zip_inspect/gdn3_full/gdn3/config.py:123).

position/nseg/HOLA cnt имеют int32: при очень длинном потоке overflow влияет на ages/modulo/valid mask. default full slots растёт линейно с L; конечные max_levels=12 и n_levels также не обеспечивают неограниченную обученную иерархию. Full acc обновляется для всех слотов; kit при закрытии банка пересчитывает bank index на полном массиве. Сохранение/восстановление inference carry, per-sequence reset и ragged batch не имеют завершённого публичного контракта.

**Доработка:** epoch+offset/rebasing или безопасный широкий счётчик с учётом JAX x64; относительные log-clocks; bounded-shape allocator/hierarchy; inference cache schema; isolate sequences; account append/compaction/sketch cost. **Тесты:** T09, T10, T13, T27, T30, T35, T37.

## 6. Что уже сделано нормально и надо сохранить

- Общая GDN-2 serial recurrence и kernel adapter; head-major boundary. Исправлять VJP/guards, сохраняя independent baseline.
- Hybrid additive archive gate и causal закрытые банки; null option. Проверять gate=0 на тех же weights, не обещать автоматический quality improvement.
- В kit настоящий model.forward pad/crop и streaming tail; не удалять поддержку arbitrary L из-за внутреннего mixer check.
- В full exact top-k индексы bank router, несколько engines, conv carry, learned bank init, канальный KDA mode.
- EDA/DP substep reuse как кандидат на эффективную generalized реализацию после независимого reference.
- Full decay-aware archive/export; честно отделить superposition от полной recurrent transition.
- Existing tests causality, basic streaming parity, gradient flow, data disjointness, quick train; расширить их, а не заменять одними benchmark scripts.
- Full capped-query evaluator лучше kit all-sequence prefill scoring; добавить второй протокол budget-from-start.
- Инкрементальное сохранение full M1 результатов и explicit SKIP в Pallas helper; довести до машинно проверяемого статуса.

## 7. Полный список тестов, которые надо добавить/усилить

**Это план будущих тестов. В этом аудите они не добавлялись и не запускались.** Имена файлов ниже — предлагаемые места в выбранном каноническом package, а не существующие артефакты.

### 7.1 Общие правила correctness

- Независимый NumPy float64 reference пишет формулу напрямую. Он не вызывает production ops/substeps и не копирует decomposition kernel, иначе одинаковая ошибка останется незамеченной.
- Проверять outputs, final state и carry каждого слоя; для kernel VJP — все q,k,v,w,b,g,h0 и cotangent dh_final, не только loss от o.
- Для smooth reference: central/directional finite differences с несколькими ε (например 10⁻³…10⁻⁶), сравнение analytic/JAX VJP и dot-product identity. В NumPy float64 начинать с atol≈10⁻⁸, rtol≈10⁻⁵ для FD; float32 serial/engines — заранее выбранная шкала около 10⁻⁵…10⁻⁴. Это стартовые tolerances, уточняемые по conditioning, а не подтверждённый error budget.
- bf16x3/default допуски закреплять по отдельному precision study, absolute+relative error и gradient cosine/norm. Общий допуск 6% способен скрыть дефект малых производных. Относительную ошибку около zero дополнить абсолютной.
- Для top-k проводить FD вдали от ties и проверять ties отдельными discrete tests. Для hard DLA boundary численная производная почти везде zero; **ST surrogate не может совпадать с FD hard decision**. Проверять выбранный surrogate отдельно и causal forward отдельно.
- Для quant cast не требовать обычной ненулевой производной int8. Training-free inference и возможный STE-training контракт проверять отдельно.
- Off-switch задавать точно (γ=0, λ=0, n_h=1, d=1+η=0, disabled hierarchy/cache/quant), не только большим отрицательным bias.
- Каждую фикстуру параметризовать по B=1/2, H=1/4, Ha=1/2/H, D small/64/128, L у boundaries и нескольким seeds. Новые paper-модули проверять отдельно и попарно; полный product всех опций может быть слишком дорогим.

### 7.2 Функциональные и математические тесты

| ID | Предлагаемое место / сценарий | Ожидаемый результат | Закрывает |
|---|---|---|---|
| T01 | test_head_subset.py: H=4, Ha=1/2/4; all cells; hybrid scan, prefill/decode и градиенты; B=2 | A всегда B×Ha×D×D; logits/carry совпадают с independent per-head run, shape не расширяется | R01 |
| T02 | test_router_ties.py: 3 одинаковых nonzero banks, k=1/2; zero q/S; invalid slots; all scores equal | ≤k выбранных valid banks; одинаковые indices/weights dense/sparse/parallel; null mass сохранена; empty archive→0 | R02, R10 |
| T03 | test_qed_reference.py: paper a=b*k+λP⊥(b*q), D=2 counterexample b=(.2,.8); λ=0; k≈0, q||k | State/output равны прямой rank-one formula; gradients q/k/b/λ проходят independent FD; additive/interleaved вариант не выдаётся за paper | R03 |
| T04 | test_osdn_reference.py: scalar-paper residual и closed hypergradient; менять v/h при фиксированных k/β; d near bounds | d trajectory не зависит от v/h; update меняет только write-side address; d∈box; d=1,η=0→baseline; outer gradient через d соответствует выбранной спецификации | R04 |
| T05 | test_budgeted_equivalence.py: ≥2 layers, N=2, ≥8 banks, fifo/merge/leveled; nonzero gate/h_init; разбиения перед/после границы | Budget-from-start one-shot scan = scan кусками = token decode = budget-aware prefill; logits и все carries allclose | R05 |
| T06 | test_engine_capabilities.py + test_archive_eval.py: один ранний orthogonal fact, FIFO маленький N; quant/memory policy engine=auto | Нельзя получить uncapped score под capped label; auto выбирает поддерживаемую семантику либо reject; result manifest содержит encoding/query policy | R05–06, R15 |
| T07 | test_dla_boundaries.py: constant stream, скачок information, min/max windows, B×H неодинаковые границы | Boundary-before-token единообразна, min/max/chronology выполняются; max forces closure, forbidden boundaries отсутствуют | R07 |
| T08 | test_dla_surrogate_merge.py: hybrid/snap θ-gradient на контролируемом smooth surrogate; банки с заданной density | θ действительно получает declared gradient в допустимой зоне; запретная зона не даёт unintended reset gradient; merge выбирает соседнюю low-info пару | R07, R12 |
| T09 | test_hierarchy_live_state.py: 2^r±1 bank appends, r=1…20 (малый D), active bins/tree transitions | Live slots/bytes ≤a+b log₂(number_banks); span coverage не теряется/не перекрывается сверх контракта; visits и amortized append соответствуют bound | R08, R33 |
| T10 | test_range_numerics.py: huge prefix 10^8 + small tail 1 в float32, g=0/weak decay, mixed signs; direct block reference | Range tail не исчезает из-за prefix subtraction; documented error bound; associativity с error tolerance; empty/single/all range | R08 |
| T11 | test_adaptive_levels.py: ручные λ, один active block на уровень, content-change и lengths beyond trained levels | Правильные blocks/weights и nonzero MLP gradients; новый уровень не молча clipped в последний; disabled adaptive→fixed reference | R09 |
| T12 | test_sse_write.py: N=1/2/4 partitions, известный class selector, k=1/2 и ties | Меняются только selected write rows/partitions; capacity реально N states; read/shared partition/gates correct; bypass→baseline; FD на continuous branches | R10 |
| T13 | test_bank_decay.py: per-channel analytic exp, first token after close, old bank under extreme distance, artificial large clock offset | Scan/parallel reading clock совпадает; no future decay; rebase-invariance; finite output; underflow loss явно измеряется; frozen mode неизменен | R11, R33 |
| T14 | test_merge_semantics.py: h_init≠0, β=0, vector decay; noncommuting erases; mixed-level histories | Exact claims проходят только в заявленном режиме; init offset корректен; lossy case не сравнивается как exact; associative summary if promised | R12 |
| T15 | test_hola_priority.py: один very-high surprise followed by >w moderate; equal ties; write score known; cache norm | Top-w сохраняет старый high-score item; score=committed ||ΔS||; causality/current-token visibility/null стабильны; cached KV и RMSNorm независимы от recurrent q/k | R13 |
| T16 | test_ccq_covariance.py: все k одинаковы и ненулевые; rotated keys; известная sample covariance | При одинаковых keys centered covariance=0; read identity; mean/count/cov match independent reference без future keys; λ=0→baseline; FD q/strength | R14 |
| T17 | test_leapquant_window.py: W=1/8/32, rank/outlier toy matrix, window-crossing decode, buffers/checkpoint | Fixed lowbit state+buffer reference = streaming semantics; compensator factors capture selected residual; error vs full float; no unexpected quant during train; actual live/peak bytes | R15 |
| T18 | test_cell_vjp.py: QED/EDA/DP interleaved vs direct matrix reference, nonzero h0, g=0 substeps, dh_final≠0 | Forward и VJP всех входов/новых projections совпадают с независимым smooth reference; γ=0/n_h=1→baseline; token read после правильного substep | R03, R19–20 |
| T19 | test_online_pca_cache.py: rotating low-rank subspace, preserved anchors, rank=D, long stream | Online basis tracks subspace и остаётся conditioned/orthonormal; basis change обновляет storage; anchors exact; rank=D→full path; bounded bytes | R16 |
| T20 | test_statex_migration.py: checkpoint selected layers, zero expansion, positive expansion, warm adaptation | Before adaptation logits соответствуют declared preserve contract; все tensors/carries согласованы; новые coords могут получать gradients; model-param delta измерен | R17 |
| T21 | test_kda_channel_gate.py: perturb one output channel wd и один decay_a; extreme logits/half-lives | Independent channel response при kda; g≤0, halfspan safe; derivative decay_a matches FD; head/channel/kda намеренно различаются | R18 |
| T22 | test_h_init_scope.py: nonzero bank initial state, reset каждой границы, Ha<H; cumulative/snap declared policy | Banks получают одно declared init; grad h_init nonzero на crafted loss; off h_init=0→старый путь; merged init учтён | R12, R18 |
| T23 | test_config_validation.py: dp_extra=−1/0/1, γ exact0, invalid dims/τ/η/levels, unsupported cell combinations | Ранние понятные errors; n_h=1 reference; допустимые shapes; cell composition capability matrix совпадает с поведением | R19, R25, R32 |
| T24 | test_atomic_decay_vjp.py: g=0 и −ε, flat sections, h0/q nonzero, k=v=0; partial zero channels; output+final-state cotangent | One-sided domain derivative/convention consistent; finite nonzero decay contribution там, где должен быть; no mask cancellation loss | R20, R22–23 |
| T25 | test_atomic_lean_clip.py: diff below/inside/above clip bound, −30/−10/0, valid halfspan | Forward clipped function и manual VJP совпадают; saturated diff gradient zero; BTL/lean сравниваются лишь в общем exact domain | R21 |
| T26 | test_domain_guards.py: g>0, huge span, b*k²>2, OSDN scaled key, NaN/Inf до sanitize | Invalid input reject/flag по контракту; health counters фиксируют pre-sanitize failure; allowed border не отвергается произвольно | R22, R32 |
| T27 | test_checkpoint_roundtrip.py: uninterrupted K steps vs K1+save/resume, inference carry save/resume, mismatch config | RNG/optimizer/schedule/EMA/carry roundtrip совпадают; mismatched schema/config отклонён или migrated; atomic→GDN-3 warmstart parity объявлен | R17, R30, R33 |
| T28 | test_precision_presets.py: eps=0 highest reference, eps≠0 stabilized reference, bf16x3/default long sequence | Ошибка decomposed на formula vs arithmetic; все gradients/outputs в заранее заданных bounds; backend/engine precision реально применяется | R23–24 |
| T29 | test_backend_matrix.py: all supported cells/engines/Ha/precision/layout + unsupported combos | Reference/JAX/Pallas parity; explicit skip/reject вместо false pass; zero-feature composition baseline; no silent option ignored | R24, R32 |
| T30 | test_padding_batch_reset.py: L=0/1/C−1/C/C+1, ragged B, pad tails, reset one sequence | Real logits и unpadded cache invariant; no-op mode не меняет recurrence/conv/position согласно контракту; sample isolation | R25, R33 |

### 7.3 Тесты обвязки, метрик и устойчивости

| ID | Предлагаемое место / сценарий | Ожидаемый результат | Закрывает |
|---|---|---|---|
| T31 | test_m1_verdict.py: zero baseline, all fail, empty seeds, one seed, zero variance, censored P*, oracle-only, health fail | INVALID/INCONCLUSIVE вместо PASS; learned-only criterion; strict schema/seed pairing; raw curves и uncertainty сохранены | R27 |
| T32 | test_mqar_protocol.py: fact across C, arbitrary fact positions/qstart, unique queries, repeated-query split, overwrite key | Targets/answer masks verified; no answer leakage/future dependencies; causal oracle starts после известного query key либо privileged label | R28 |
| T33 | test_runner_discovery.py: inject failing serial/Fenwick/Pallas; TPU unavailable | Все обязательные tests обнаружены; итог fail при fail; SKIP не засчитывается certification; counts/hash/backend записаны; all-stage rc корректен | R26, R31 |
| T34 | long_context_recall suite: 2K/8K/32K/128K/1M; facts uniformly/interleaved; adversarial distractors; multi-hop/passkey/MQAR overwrite | Paired accuracy-distance-density curves; predefined ε и min quality; trained/inference lengths отличаются; errors разделены router/bank/exact selection | R28, цель Inf |
| T35 | test_speed_protocol.py + benchmark: consistent filled cache, cache leaves outputs, stateful regular/close/merge/quant events | Timed code действительно обновляет state; no stale sketch; latency/bytes vs N/L и p95/p99; compile отделена от steady execution | R29, R33 |
| T36 | test_bootstrap_logging.py: clean tmp output, mock package in root/pkg, pinned dependency, install/import failure | Out-dir создан до log; package path/hash зафиксирован; любой subprocess error передан caller; no false success | R31 |
| T37 | test_stream_longevity.py: inject pos/nseg/cnt near 2^31−1, clock rebase, много append/delete и cache resume | No overflow corrupting validity/order; relative ages remain correct; allocator не растёт сверх declared bound; finite norms; stream restart эквивалентен | R33, цель Inf |

### 7.4 Дополнительные обязательные property checks

Каждый новый memory-policy должен проверять chronology, причинность, coverage, отсутствие cross-batch/head corruption, finite states, metadata/sketch consistency после append/merge/quant, сохранение count/levels при export и resume. Для каждого trainable parameter проверить не просто finite gradient, а ненулевой на специально построенном signal и независимый derivative там, где он существует.

Для composition проверить минимум KDA+EDA, KDA+DP, hierarchy+adaptive λ, hierarchy+HOLA, CCQ+decay, quant+merge и Ha<H. Сочетания QED/OSDN с другими cells запускать только после определения общего asymmetric-address algebra. Quant/Oja/StateX нужно проверять также при загрузке ранее обученного checkpoint, а не только от init.

## 8. План доработок: зависимости и критерии приёмки

Полный чек-лист находится в [GDN3_TODO.md](C:/Users/omirb/Downloads/Downloads/gdn3_kit/GDN3_TODO.md). Ниже — порядок и условия закрытия этапов. Реализация всех paper-идей не является предварительным доказательством «Inf»; каждый механизм должен отдельно улучшать проверяемый показатель.

| Этап | Работа | Условие перехода дальше |
|---|---|---|
| S0 · Спецификация | Выбрать canonical snapshot; утвердить GDN-2 recurrence, memory semantics, цель/ε/resources; paper vs variant names | Один versioned spec + hash manifest + capability matrix; нет противоречий O(log L)/O(1)/lossless |
| S1 · Correctness baseline | R01–06, R20–23, R25–27, R31–32; reference, guards, VJP, runner, budget engine | Independent refs и краевые tests проходят на serial; Pallas проверен отдельно на TPU; отсутствие тестов/TPU не закрывает stage |
| S2 · Streaming memory | B1/B2/B5/B6/B12; cache schema/counters; sparse read/иерархия | One-shot/chunk/decode/resume equivalence; live-bin bound доказан и измерен; lossiness и state costs объявлены |
| S3 · Recall protection | Paper HOLA и, если нужно arbitrary exact recall, растущий external archive/index | Old important facts survive declared policy; quota/eviction/I/O semantics; quality-length curve и failure attribution |
| S4 · Cell/read ablations | KDA, corrected QED, EDA, DP, OSDN, SSE, CCQ по одной | Independent formula/VJP, exact off-switch, paired MQAR improvement; отсутствие improvement допустимо, но статус «не принят» |
| S5 · Экономика | Настоящий LeapQuant; Oja adaptation; StateX migration; profiled Pallas fusion | Actual live+peak memory, error-quality curve, real stateful speed; checkpoint compatibility; нет скрытого dense O(LN) пути |
| S6 · Длинный контекст | Retention/capacity/reasoning/LM через streaming, extrapolation, soak, resume | Протокол и конечные L явно названы; quality within ε при healthy runs; bounded/log memory и latency measurements; theoretical claim ограничен контрактом |

### 8.1 Минимальная архитектура, которую стоит довести первой

1. **Стабильный GDN-2/KDA local backbone**, независимый от archive implementation, с проверенным kernel VJP.
2. **Один causal memory manager:** current bank + information metadata + active exponential bins; одинаковые операции для обучения, prefill и decode. Bounded policies допускаются отдельными режимами.
3. **Hierarchical candidate retrieval:** ограничить число оцениваемых summary candidates, выполнять D² read только выбранных; не создавать плотную L×N матрицу. Что происходит со старым контентом, записано в lossiness contract.
4. **Exact important-token cache:** top-w по committed surprise, отдельная sharp read-нормализация. Это защита выбранных KV, а не универсальная память истории.
5. **Optional external persistent archive:** нужен, если требуется восстановление любого независимого старого факта; хранение растёт с фактами. Отдельно измерять RAM/диск/индекс/I/O и качество поиска. Exact storage без корректного retrieval ещё не обеспечивает exact answer.
6. **Versioned cache и health state:** restore, reset, per-sample mask, rebasing, reproducible config.

Сначала эта система должна пройти memory advantage эксперимент. Дополнительные cell refinements не исправят сами по себе O(N) router или утрату фактов при merge.

### 8.2 Протокол M1 / retention / extrapolation

- **Positive control:** короткая задача, где baseline обучается. Проверять learning curves/skip rate/norms, а не только итоговый accuracy.
- **Retention:** один/несколько старых фактов + контролируемое расстояние, отдельно pure filler и фактологические distractors.
- **Capacity:** новые независимые факты распределены по всей истории; queries охватывают все возрастные уровни, не только первый bank; varied fact density и boundary placement.
- **Extrapolation:** train на 8K–16K после успешных small controls; eval на 32K/128K/1M по streaming. Это предлагаемые ступени, не утверждение о доступной TPU-памяти/времени.
- **Сравнение:** GDN-2c baseline, GDN-3 learned, диагностический privileged oracle, HOLA-only, hierarchy-only; затем accepted components. Одинаковые supervised tokens/optimizer schedule/data seeds; отдельные iso-params, iso-state и iso-FLOP/runtime controls.
- **Диагностика:** router candidate recall / correct-bank top-k / null mass / score margins / per-level hits; bank decoding error при true candidate; HOLA admission/retention/hit-rate; quant error и state norms; gradients/skip/health.
- **M1:** P* по заранее заданному accuracy threshold (например 0.9), adequate P-grid без потолка/провала, минимум три парных seeds и uncertainty для gap. Критерий 2× и 3σ сохранять лишь при валидной оценке; base=0, один seed, censoring, baseline health failure→INVALID/INCONCLUSIVE.
- **Oracle:** oracle-training ceiling отдельно от test-time oracle на learned checkpoint. Разница может зависеть от обучения; равенство низких/насыщенных accuracies не доказывает идеальный routing.
- **Reasoning/LM:** MQAR не доказывает long-context language understanding. Добавить passkey/needle, overwrite/update, multi-hop facts, selective copy и text perplexity by position. Для externally referenced public suite version/dataset/license зафиксировать перед запуском.

### 8.3 Что писать в итоговый экспериментальный статус

Для каждого результата сохранять: snapshot/source hashes; atomic commit и импортированный path; Python/JAX/jaxlib/optax; hardware/device count; dtype/precision/kernel/eps/bs2; model/config; seed/data protocol; target mask; length/distance/density/budget; train steps/schedule/skips/health; checkpoint hash; outputs/metric aggregation; test pass/fail/skip counts; memory live+peak; compile/steady timing и event latencies.

Уровни готовности разделить:

- **IMPLEMENTED:** нужный путь есть в модели, а не только helper.
- **REFERENCE-VALIDATED:** независимый forward/VJP и выключение механизма проверены.
- **BACKEND-VALIDATED:** проверена конкретная JAX/Pallas/hardware версия.
- **QUALITY-VALIDATED:** paired controls подтверждают полезность на объявленном протоколе.
- **LONG-STREAM-VALIDATED:** подтверждены finite lengths/continuous stream, memory/latency и resume.

Ни один из этих статусов не означает математическое доказательство бесконечной exact памяти произвольной истории.

## 9. Исправления относительно предыдущего STATUS

1. Arbitrary-length forward kit **есть**: pad/crop model.forward. Alignment check mixer не является его опровержением.
2. Теперь прочитан весь доступный локальный Python-код и полностью извлечён ZIP. Предыдущий выборочный обзор не был полным code audit.
3. Локальный atomic проверен по коду; возможные дефекты backward теперь имеют точные ссылки R20–23. Соответствие историческому TPU-снимку по-прежнему не установлено.
4. Источники SSE/CCQ/QED/HOLA доступны; paper mapping конкретизирован. CCQ placeholder заменён первоисточником, covariance уточнена.
5. Нельзя считать capped benchmark kit валидным измерением budget; R06 имеет конкретный call-path.
6. **M0/M1 не помечаются PASS:** новых запусков здесь не было. Исторические короткие результаты не перенесены на текущий code snapshot.

Обновлённый [STATUS.md](C:/Users/omirb/Downloads/Downloads/gdn3_kit/STATUS.md) даёт краткий статус. Старую рабочую заметку сохранено отдельно как STATUS_PREVIOUS_AUDIT.md; исходный файл Downloads/STATUS.md сохранён без изменения.

## 10. Заключение

Текущий GDN-3 — исследовательский каркас memory augmentation, а full — набор частично интегрированных расширений. Наиболее срочно исправить subset-head scan, router ties, budget-prefill/evaluation, QED/OSDN math и atomic corner VJP; затем строить реальную streaming hierarchy и проверять memory advantage. Откладывать correctness до реализации всех 16 модулей будет ошибкой: некоторые нынешние тесты проверяют одинаковые неправильные формулы, а некоторые бенчмарки измеряют другой режим.

Для цели «бесконечные контексты» первым принимаемым результатом должен стать воспроизводимый долгий поток с объявленными гарантиями хранения, возрастающей длиной экспериментов и измеренной стоимостью. Universal lossless history при sublinear finite-precision state не является достижимым контрактом. Exact сохранение всех фактов требует отдельного растущего хранилища либо ограничения класса задач.
