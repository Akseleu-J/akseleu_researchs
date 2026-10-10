"""Machine-readable feature availability; code presence is not validation."""
from __future__ import annotations
import importlib.metadata
from dataclasses import asdict

def _version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None

def report(cfg):
    jax_version=_version("jax");jaxlib_version=_version("jaxlib");optax_version=_version("optax")
    modules={
      "asymmetric_recurrence":"implemented_unvalidated","hierarchy":"implemented_unvalidated",
      "loglinear":"implemented_unvalidated","router":"implemented_unvalidated",
      "dla":"implemented_unvalidated","sse":"implemented_unvalidated",
      "hola":"implemented_unvalidated","ccq":"implemented_unvalidated",
      "leapquant":"implemented_unvalidated","oja_kv":"helper_only_unvalidated",
      "external_archive":"implemented_exact_scan_unvalidated",
      "statex":"migration_contract_partial","atomic_import":"normalized_schema_only",
      "training":"full_bptt_synthetic_only","tbptt":"unsupported",
      "pallas":"source_present_compiler_unvalidated"}
    unsupported=[]
    if cfg.quant!="none" and cfg.backbone=="atomic":unsupported.append("atomic-backbone plus quant integration is not certified")
    if cfg.mode=="snap" and cfg.memory!="fifo":unsupported.append("snap requires FIFO")
    if cfg.lambda_mode.startswith("paper") and not cfg.max_levels:unsupported.append("paper lambda requires bounded max_levels")
    return dict(schema=1,config_hash=cfg.fingerprint(),requested=dict(mode=cfg.mode,memory=cfg.memory,backbone=cfg.backbone,cell=cfg.cell,quant=cfg.quant,backend=cfg.backend,training_modes=["full_bptt"]),modules=modules,environment=dict(python_platform=__import__("platform").python_version(),jax=jax_version,jaxlib=jaxlib_version,optax=optax_version),runtime_ready=bool(jax_version and jaxlib_version),unsupported=unsupported,validated=False,production_ready=False,context_claim="unbounded stream position; summary is lossy; exact archive grows O(L); universal lossless recall is not promised")
