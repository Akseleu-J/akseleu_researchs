import jax
from ..cells.base import apply_step

def select_backend(cfg):
    hardware=jax.default_backend()
    if cfg.backend=="pallas":
        if hardware!="tpu" and not cfg.pallas_interpret:
            raise RuntimeError("forced Pallas requires TPU; pallas_interpret is a separate diagnostic mode")
        return "pallas", "explicit primitive backend; memory control/MLPs use JAX"
    if cfg.backend=="auto" and hardware=="tpu":return "pallas","TPU primitive selected; other operations use explicit JAX fallback"
    return "jax","JAX recurrence and memory operations"

def primitive(cfg):
    name,_=select_backend(cfg)
    if name=="jax":return apply_step
    from .pallas import step
    return lambda s,x,precision:step(s,x,interpret=cfg.pallas_interpret)
