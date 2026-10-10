# GDN-3 v2

A research implementation scaffold for causal recurrent language modeling with bounded hierarchical summaries and an optional exact external archive. «Infinite context» means the stream has no configured terminal position and carry can grow; it does **not** promise lossless retention or correct answers about every past token.

## Install

Use Python >=3.11. Create a clean environment and install a JAX build matching the selected CPU/CUDA/TPU target first, then install this package with training extras as required. The repository declares JAX, NumPy and Optax; pytest is optional. Do not assume Pallas support from the Python version alone. Record jax/jaxlib/backend/device versions with every result.

`pip install -e .`

## API

`GDN3Config` validates dimensions and incompatible modes before tracing.

`params = init_params(jax.random.PRNGKey(0), cfg)`

`logits, carry = prefill(params, token_ids, cfg)`

`next_logits, carry = decode_step(params, carry, next_token, cfg)`

Persist a carry with `save_cache/load_cache`; save optimizer, RNG and EMA together with `save_checkpoint/load_checkpoint`. Carried cache schema is versioned and its config fingerprint is checked. For persistent retrieval, `Archive` preserves original spans and `PersistentSession` performs causal retrieval. Application code remains responsible for durable append at committed boundaries and supplying a key/value projection.

## CLI

`gdn3 train --config configs/default.json --output run.gdn3`

`gdn3 smoke --config configs/default.json --tokens token_ids.json`

Training currently uses generated random-token batches as a pipeline smoke source, not a meaningful language corpus. MQAR helper is a diagnostic generator, not a completed benchmark. TBPTT is not implemented despite being reserved in config.

## Evidence and status

Read [capability matrix](docs/CAPABILITIES.md), [test plan](docs/TEST_PLAN.md), [source patch record](docs/VENDOR_PATCHES.md) and root [status](../STATUS.md). Tests, training, benchmarks, import/package smoke, Pallas compilation and TPU execution have not been run in this step.
