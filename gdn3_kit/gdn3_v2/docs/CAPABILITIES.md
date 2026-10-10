# Реестр возможностей GDN-3 v2

Статус относится к наличию и интеграции кода, не к корректности/качеству. «Реализован код» требует тестового подтверждения на шаге 4.

| Возможность | Состояние шага 3 | Место | Критический долг |
|---|---|---|---|
| Config/API/prefill/decode | Код есть; runtime не запускался | gdn3/config.py, model.py | JAX absent; carry invariants unverified |
| GDN-2 asymmetric ABI | Код + независимый NumPy recurrence | cells/base.py, reference/gdn2.py | full oracle/gradient fixtures не закрыты |
| QED/EDA/DP/OSDN/KDA/PDN | helpers/stream integration partial | cells/, stream.py | formula and VJP tests pending; deferred cells not paper certified |
| Dense/hierarchical memory/router/DLA | code paths present | memory/, stream.py | coverage/overflow and merge semantics need tests |
| Log-linear | adapter path present | memory/loglinear.py | reference alignment and long-run tests pending |
| SSE/HOLA/CCQ/adaptive λ | integrated branches present | memory/, stream.py | extensive paper/parity tests pending |
| LeapQuant/OjaKV | state code + selected integration | compression/, stream.py | low-bit error, dynamic flush, model KV session wiring incomplete |
| Persistent exact archive | SQLite append/search/version API | persistence/archive.py | exact scan O(history); new PersistentSession is host adapter and caller supplies projection |
| StateX | plan/merge contract and fail-closed callback | migration/statex.py | whole-model callback, optimizer/schema migration and parity not supplied/verified |
| Atomic import | explicit shape-checking adapter + initializer | migration/atomic_import.py, parameters.py | external source checkpoint format adapter may need mapping |
| Training | AdamW, cosine, clip, EMA, RNG/data cursor, save/resume | training/ | TBPTT currently declared but not implemented; training dataset synthetic only |
| Checkpoint | versioned NumPy bundle + schema checks | persistence/ | bfloat16 typed-array edge case unresolved |
| JAX/Pallas | dispatch and custom primitive source | backends/ | no JAX/Pallas compiler installed here; hardware correctness not evaluated |
| Tests | 11 starter modules + tier runner | tests/, scripts/run_checks.py | T01–T48 not all covered or run |

## Honest goal status

The project can host an unbounded token counter and geometric growth of binary levels, while the summary profile remains lossy and grows approximately O(log L). The exact persistent archive grows O(L) on disk; its semantic index is an exact disk scan, not a practical ANN service. Neither mechanism proves successful recall, answer quality, or universally lossless infinite context.
