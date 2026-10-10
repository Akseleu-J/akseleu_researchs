"""Explicit StateX paper reinitialization and selected-layer head migration.

StateX is model surgery, not an inference-time option. This module builds a
reviewable plan, requires a caller supplied paper reinitializer for the new
coordinates, and refuses to silently reshape dependent tensors."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Mapping, Any
import hashlib
import json
import numpy as np

@dataclass(frozen=True)
class StateXPlan:
    schema: int
    selected_layers: tuple[int, ...]
    merge_groups: tuple[tuple[tuple[int, ...], ...], ...]
    old_heads: int
    new_heads: int
    key_dim: int
    value_dim: int
    policy: str = "paper_reinit"
    seed: int = 0

    def to_dict(self):
        return asdict(self)
    def fingerprint(self):
        raw=json.dumps(self.to_dict(),sort_keys=True,separators=(",",":")).encode()
        return hashlib.sha256(raw).hexdigest()

def build_statex_plan(selected_layers, merge_groups, old_heads, key_dim, value_dim, seed=0):
    layers=tuple(sorted(set(int(x) for x in selected_layers)))
    if not layers or layers[0]<0: raise ValueError("select at least one nonnegative layer")
    groups=tuple(tuple(tuple(int(h) for h in g) for g in layer) for layer in merge_groups)
    if len(groups)!=len(layers): raise ValueError("one group list per selected layer")
    new_counts=[]
    for layer_groups in groups:
        flat=[h for g in layer_groups for h in g]
        if any(not g for g in layer_groups) or sorted(flat)!=list(range(old_heads)):
            raise ValueError("each merge plan must partition all source heads exactly once")
        new_counts.append(len(layer_groups))
    if len(set(new_counts))!=1: raise ValueError("selected layers must have a shared new head count")
    if new_counts[0]<1 or new_counts[0]>=old_heads: raise ValueError("StateX merge must reduce head count")
    if key_dim<1 or value_dim<1: raise ValueError("head dimensions must be positive")
    return StateXPlan(1,layers,groups,int(old_heads),new_counts[0],int(key_dim),int(value_dim),seed=int(seed))

def merge_heads(array, groups, head_axis=0, weights=None):
    """Weighted merge of complete head slices; caller handles paper reinit."""
    x=np.asarray(array)
    axis=head_axis % x.ndim
    if x.shape[axis] != sum(len(g) for g in groups): raise ValueError("head axis does not match groups")
    if any(not g for g in groups): raise ValueError("empty merge group")
    outputs=[]
    for gi,g in enumerate(groups):
        if len(set(g))!=len(g) or any(h<0 or h>=x.shape[axis] for h in g): raise ValueError("invalid head index")
        part=np.take(x,g,axis=axis)
        if weights is None: w=np.full((len(g),),1/len(g),np.float64)
        else:
            w=np.asarray(weights[gi],np.float64)
            if w.shape!=(len(g),) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0: raise ValueError("invalid merge weights")
            w=w/w.sum()
        shape=[1]*part.ndim;shape[axis]=len(g)
        outputs.append(np.sum(part*w.reshape(shape),axis=axis))
    return np.stack(outputs,axis=axis)

def migrate_model(params: Mapping[str,Any], plan: StateXPlan, reinitialize: Callable[[dict,StateXPlan],dict], merge_weights=None):
    """Perform explicit selected-layer merge then delegated paper reinitialization.

    The callback must rebuild every head-dependent projection, convolution,
    router, SSE and output tensor and document its initialization policy. The
    callback result is validated for layer/head counts; optimizer and carry are
    deliberately not accepted here and must be migrated/reinitialized by the
    checkpoint-level caller.
    """
    if plan.policy != "paper_reinit": raise ValueError("only paper_reinit is supported; preserve parity is not proven")
    if "layers" not in params: raise ValueError("model parameters lack layers")
    result=dict(params); layers=list(params["layers"])
    if max(plan.selected_layers)>=len(layers): raise ValueError("selected layer out of range")
    audit=[]
    for pos,index in enumerate(plan.selected_layers):
        source=layers[index]; groups=plan.merge_groups[pos]
        merged={}
        # Merge only fields explicitly marked by the callback. Unknown schemas fail closed.
        head_fields=source.get("statex_head_fields")
        if not head_fields: raise ValueError("layer must declare statex_head_fields before migration")
        for path,axis in head_fields.items():
            cursor=source; parts=path.split(".")
            for key in parts[:-1]: cursor=cursor[key]
            key=parts[-1];node=dict(cursor)
            node[key]=merge_heads(cursor[key],groups,int(axis),None if merge_weights is None else merge_weights[pos])
            merged[path]=node[key]
        candidate=reinitialize(dict(source),plan)
        if not isinstance(candidate,dict): raise TypeError("reinitializer must return a layer mapping")
        if candidate.get("head_count")!=plan.new_heads: raise ValueError("reinitializer must record the new head_count")
        layers[index]=candidate
        audit.append(dict(layer=index,merged_paths=sorted(merged),policy=plan.policy,seed=plan.seed))
    result["layers"]=layers
    return result,dict(plan=plan.to_dict(),plan_hash=plan.fingerprint(),layers=audit,optimizer="reinitialize_required",cache="discard_and_rebuild")
