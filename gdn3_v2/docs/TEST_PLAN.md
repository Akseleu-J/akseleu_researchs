# Step 4 test plan (not executed)

Run with `python scripts/run_checks.py --tier core --output-dir artifacts/test-results` in an environment with pinned JAX/jaxlib/Optax/pytest; the hardware tier uses `--tier hardware` and reports SKIP when no TPU is visible. A passing CPU suite cannot validate Pallas compilation or TPU throughput.

| IDs | Test focus | Existing or intended test file |
|---|---|---|
| T01–T04 | asymmetric ABI; QED; paper OSDN; cell signal/gradient | tests/test_reference.py, tests/test_cells.py |
| T05–T08 | budget-aware prefill, chunk parity, DLA boundary, surrogate | tests/test_stream.py, tests/test_dla.py |
| T09–T14 | binary bins, capacity growth, deterministic top-k, affine merge, aligned clock | tests/test_hierarchy.py, tests/test_router.py |
| T15–T18 | adaptive λ, CCQ, HOLA, end-to-end baseline recurrence | tests/test_adapters.py, tests/test_model.py |
| T19–T24 | SSE, learned init, atomic import/VJP and precision | tests/test_sse.py, tests/test_migration.py, tests/test_backend.py |
| T25–T32 | clipping/domain guards, no-op/reset/ragged masks, checkpoint/cache | tests/test_health.py, tests/test_stream.py, tests/test_persistence.py |
| T33–T37 | runner results, install, source hashes, longevity/counters | tests/test_packaging.py, tests/test_longevity.py |
| T38–T40 | deferred FG2/QDelta/PDN paper references | deferred reference suite; not a core release gate |
| T41–T48 | generalized OSDN, summaries, persistence causal retrieval, compression, StateX, MQAR | tests/test_compositions.py, tests/test_archive.py, tests/test_compression.py, tests/test_migration.py, tests/test_bench_data.py |

The listed suite is larger than the twelve starter modules currently present. Step 4 starts by running the committed tests, then adds missing coverage against the matrix. No test result is implied by this plan.
