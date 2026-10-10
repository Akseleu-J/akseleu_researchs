"""GDN3 command-line entry point."""
import argparse
import json
from pathlib import Path
import jax
from .config import GDN3Config
from .training import TrainConfig,train
from .parameters import init_params
from .model import prefill,decode_step
from .persistence import save_bundle,load_bundle

def main(argv=None):
    parser=argparse.ArgumentParser(prog="gdn3")
    sub=parser.add_subparsers(dest="cmd",required=True)
    tr=sub.add_parser("train");tr.add_argument("--config",required=True);tr.add_argument("--output",required=True);tr.add_argument("--resume")
    sm=sub.add_parser("smoke");sm.add_argument("--config",required=True);sm.add_argument("--tokens",required=True);sm.add_argument("--output")
    ns=parser.parse_args(argv)
    raw=json.loads(Path(ns.config).read_text(encoding="utf-8"));mc=GDN3Config.from_dict(raw["model"])
    if ns.cmd=="train":
        tc=TrainConfig(**raw["training"]);train(mc,tc,ns.output,resume=ns.resume);return 0
    params=init_params(jax.random.PRNGKey(0),mc);ids=jax.numpy.asarray(json.loads(Path(ns.tokens).read_text(encoding="utf-8")),dtype=jax.numpy.int32)
    logits,cache=prefill(params,ids[None,:],mc)
    logits2,cache=decode_step(params,cache,ids[-1:],mc)
    if ns.output:save_bundle(ns.output,dict(params=params,cache=cache),dict(config_hash=mc.fingerprint()))
    print(json.dumps(dict(prefill_shape=list(logits.shape),next_logits_shape=list(logits2.shape),position=jax.device_get(cache["position"]).tolist()),default=str));return 0

if __name__=="__main__":raise SystemExit(main())
