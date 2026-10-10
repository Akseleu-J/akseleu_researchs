"""Named, fail-closed import adapter for the vendored atomic GDN-2 checkpoint."""
from __future__ import annotations
from typing import Mapping
import numpy as np

def _array(x,name):
    a=np.asarray(x)
    if not np.isfinite(a).all(): raise ValueError(f"nonfinite tensor: {name}")
    return a.copy()

def import_atomic_checkpoint(checkpoint: Mapping, source_config: Mapping, target_config, *, selected_layers=None):
    """Map dense atomic layers into explicit GDN3 atomic_core schema.

    Input keys: token_embed, layers[{q,k,v,b,w,g,o,up,down}], final_norm
    and optional per-layer convolution carry. Shape mismatches fail; no silent pad.
    """
    required={"token_embed","layers"}
    if not required.issubset(checkpoint): raise ValueError("checkpoint requires token_embed and layers")
    if source_config.get("d_model")!=target_config.d_model: raise ValueError("d_model mismatch")
    if source_config.get("n_heads")!=target_config.n_heads: raise ValueError("n_heads mismatch")
    if source_config.get("d_head")!=target_config.d_key or target_config.d_key!=target_config.d_value: raise ValueError("atomic import requires matching square head dimensions")
    if len(checkpoint["layers"])!=target_config.n_layers: raise ValueError("layer count mismatch")
    if selected_layers is None: selected_layers=range(target_config.n_layers)
    selected=set(int(i) for i in selected_layers)
    if any(i<0 or i>=target_config.n_layers for i in selected): raise ValueError("selected layer out of range")
    d=target_config.d_model; h=target_config.n_heads; k=target_config.d_key; v=target_config.d_value
    if h*k!=h*v: raise ValueError("atomic output must be square")
    embed=_array(checkpoint["token_embed"],"token_embed")
    if embed.shape!=(target_config.vocab,d): raise ValueError("token embedding shape mismatch")
    result={"embed":embed,"layers":[],"final_norm":_array(checkpoint.get("final_norm",np.ones(d)),"final_norm"),"head":checkpoint.get("head")}
    for i,src in enumerate(checkpoint["layers"]):
        if i not in selected: raise ValueError("partial import is not a whole-model migration; supply every layer")
        core={}
        for name,out in (("q",h*k),("k",h*k),("v",h*v),("b",h*k),("w",h*v),("g",h*k),("o",d),("up",int(source_config.get("expansion",2))*d),("down",d)):
            if name not in src: raise ValueError(f"missing layer {i}.{name}")
            block={key:_array(val,f"layers.{i}.{name}.{key}") for key,val in src[name].items()}
            if block.get("w",np.empty(0)).shape!=(d,out) or block.get("b",np.empty(0)).shape!=(out,): raise ValueError(f"shape mismatch: layers.{i}.{name}")
            core[name]=block
        result["layers"].append({"atomic_core":core,"conv_q":np.zeros((target_config.d_conv-1,h*k),np.float32),"conv_k":np.zeros((target_config.d_conv-1,h*k),np.float32),"conv_v":np.zeros((target_config.d_conv-1,h*v),np.float32)})
    if result["final_norm"].shape!=(d,): raise ValueError("final norm shape mismatch")
    if result["head"] is not None:
        head={key:_array(val,"head."+key) for key,val in result["head"].items()}
        if head.get("w",np.empty(0)).shape!=(d,target_config.vocab) or head.get("b",np.empty(0)).shape!=(target_config.vocab,): raise ValueError("output head shape mismatch")
        result["head"]=head
    return result,dict(adapter="atomic-gdn2-dense-v1",layers=list(range(target_config.n_layers)),optimizer="reinitialize_required",cache="discard_and_rebuild",parity="requires_validation")
