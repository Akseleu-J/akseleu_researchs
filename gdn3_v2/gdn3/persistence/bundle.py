"""Versioned, atomic JSON + NumPy-array bundles. No pickle or executable metadata."""
from pathlib import Path
import hashlib
import io
import json
import os
import tempfile
import zipfile
import numpy as np
SCHEMA=1

def _encode(value,arrays):
    if isinstance(value,dict):return {"kind":"dict","items":[[k,_encode(v,arrays)] for k,v in value.items()]}
    if isinstance(value,(list,tuple)):return {"kind":"list" if isinstance(value,list) else "tuple","items":[_encode(v,arrays) for v in value]}
    if hasattr(value,"dtype") and hasattr(value,"shape"):
        name=f"a{len(arrays)}";a=np.asarray(value)
        if a.dtype.hasobject:raise TypeError("object arrays are not supported")
        logical=a.dtype.name
        if logical=="bfloat16":a=a.view(np.uint16)
        arrays[name]=a
        return {"kind":"array","name":name,"shape":list(a.shape),"dtype":logical,"storage_dtype":a.dtype.str}
    if value is None or isinstance(value,(str,bool,int,float)):return {"kind":"scalar","value":value}
    raise TypeError(f"unsupported bundle value: {type(value)}; flatten optimizer state against a fresh template")

def _decode(spec,arrays,as_jax):
    kind=spec["kind"]
    if kind=="dict":return {k:_decode(v,arrays,as_jax) for k,v in spec["items"]}
    if kind in {"list","tuple"}:
        out=[_decode(v,arrays,as_jax) for v in spec["items"]]
        return tuple(out) if kind=="tuple" else out
    if kind=="scalar":return spec["value"]
    if kind!="array":raise ValueError("unknown bundle node")
    a=arrays[spec["name"]]
    if list(a.shape)!=spec["shape"] or a.dtype.str!=spec.get("storage_dtype",spec["dtype"]):raise ValueError("array schema mismatch")
    if spec["dtype"]=="bfloat16":
        if as_jax:
            import jax.numpy as jnp
            return jnp.asarray(a).view(jnp.bfloat16)
        return a.view(np.dtype("V2"))
    if as_jax:
        import jax.numpy as jnp
        return jnp.asarray(a)
    return a.copy()

def save_bundle(path,tree,metadata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={};structure=_encode(tree,arrays)
    data=io.BytesIO();np.savez(data,**arrays);payload=data.getvalue()
    manifest=dict(schema=SCHEMA,structure=structure,metadata=metadata or {},arrays_sha256=hashlib.sha256(payload).hexdigest())
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w+b") as raw:
            with zipfile.ZipFile(raw,"w",compression=zipfile.ZIP_STORED) as z:
                z.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,allow_nan=False))
                z.writestr("arrays.npz",payload)
            raw.flush();os.fsync(raw.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return manifest

def load_bundle(path,as_jax=True):
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read("manifest.json"));payload=z.read("arrays.npz")
    if manifest["schema"]!=SCHEMA:raise ValueError("unsupported bundle schema")
    if hashlib.sha256(payload).hexdigest()!=manifest["arrays_sha256"]:raise ValueError("bundle checksum mismatch")
    with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:tree=_decode(manifest["structure"],arrays,as_jax)
    return tree,manifest["metadata"]
