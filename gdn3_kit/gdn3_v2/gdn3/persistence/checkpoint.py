import importlib.metadata
import jax
from .bundle import save_bundle,load_bundle

def versions():
    out={}
    for name in ("jax","jaxlib","numpy","optax"):
        try:out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:out[name]=None
    return out

def save_checkpoint(path,state,cfg,training_config,source_manifest=None):
    leaves,_=jax.tree_util.tree_flatten(state["optimizer"])
    tree={k:v for k,v in state.items() if k!="optimizer"};tree["optimizer_leaves"]=leaves
    return save_bundle(path,tree,dict(kind="training_checkpoint",config=cfg.to_dict(),config_hash=cfg.fingerprint(),training_config=training_config,versions=versions(),source_manifest=source_manifest))

def load_checkpoint(path,cfg,tx,training_config,allow_version_change=False):
    tree,meta=load_bundle(path)
    if meta.get("kind")!="training_checkpoint" or meta["config_hash"]!=cfg.fingerprint():raise ValueError("checkpoint configuration mismatch")
    if meta["training_config"]!=training_config:raise ValueError("optimizer/schedule/data configuration mismatch")
    if not allow_version_change and meta["versions"]!=versions():raise ValueError("runtime versions changed; explicitly accept migration or restore the recorded environment")
    template=tx.init(tree["params"]);expected,structure=jax.tree_util.tree_flatten(template)
    saved=tree.pop("optimizer_leaves")
    if len(saved)!=len(expected) or any(a.shape!=b.shape or a.dtype!=b.dtype for a,b in zip(saved,expected)):raise ValueError("optimizer schema mismatch")
    tree["optimizer"]=jax.tree_util.tree_unflatten(structure,saved)
    return tree,meta
