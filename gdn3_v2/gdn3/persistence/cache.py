from .bundle import save_bundle,load_bundle

def save_cache(path,cache,cfg,external_cursor=None):
    return save_bundle(path,cache,dict(kind="inference_cache",config=cfg.to_dict(),config_hash=cfg.fingerprint(),external_cursor=external_cursor))

def load_cache(path,cfg):
    cache,meta=load_bundle(path)
    if meta.get("kind")!="inference_cache" or meta.get("config_hash")!=cfg.fingerprint():raise ValueError("cache/model configuration mismatch; use explicit migration")
    return cache,meta.get("external_cursor")
