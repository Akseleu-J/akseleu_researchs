"""Legacy BLR diagnostic adapter; v2 model defaults to the new asymmetric primitive.

Legacy sanitize is a surrogate outside its unclipped healthy domain. The host
entry validates inputs and exposes this limitation rather than certifying a
rescued output. Traced callers must use checked inputs from their data path.
"""
import numpy as np
from .._vendor.atomic_gdn2.config import BLRConfig
from .._vendor.atomic_gdn2.pipeline import make_blr_trainable
from .._vendor.atomic_gdn2.domain import check_preconditions

def checked_call(q,k,v,w,b,g,h0=None,config=None,scale=None):
    config=config or BLRConfig(wy_eps=0.)
    for name,a in zip(("q","k","v","w","b","g"),(q,k,v,w,b,g)):
        if not np.isfinite(np.asarray(a)).all():raise ValueError(name+" contains nonfinite values")
    gg=np.asarray(g);kk=np.asarray(k);bb=np.asarray(b)
    if np.any(gg>0):raise ValueError("g must be nonpositive")
    alignment=np.sum(bb*kk*kk,axis=-1)
    if np.any(alignment<0) or np.any(alignment>2):raise ValueError("erase alignment outside [0,2]")
    if g.shape[-2]:check_preconditions(g,config,strict=True)
    fn=make_blr_trainable(config,scale if scale is not None else k.shape[-1]**-.5)
    return fn(q,k,v,w,b,g,h0)
