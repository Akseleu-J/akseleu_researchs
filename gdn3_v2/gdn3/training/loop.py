"""Optax AdamW training, cosine schedule, EMA and resumable data/RNG counters."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import optax
from .config import TrainConfig
from .data import random_lm_batch
from ..model import forward,masked_ce
from ..parameters import init_params
from ..persistence.checkpoint import save_checkpoint,load_checkpoint
from .. import counters as ct

def optimizer(cfg):
    schedule=lambda step: cfg.min_learning_rate+.5*(cfg.learning_rate-cfg.min_learning_rate)*(1+jnp.cos(jnp.pi*jnp.clip((step-cfg.warmup_steps)/max(1,cfg.steps-cfg.warmup_steps),0,1)))
    if cfg.warmup_steps:schedule=lambda step:jnp.where(step<cfg.warmup_steps,cfg.learning_rate*(step+1)/cfg.warmup_steps,cfg.min_learning_rate+.5*(cfg.learning_rate-cfg.min_learning_rate)*(1+jnp.cos(jnp.pi*jnp.clip((step-cfg.warmup_steps)/max(1,cfg.steps-cfg.warmup_steps),0,1))))
    return optax.chain(optax.clip_by_global_norm(cfg.clip_norm),optax.adamw(schedule,weight_decay=cfg.weight_decay))

def init_training(model_cfg,train_cfg):
    key=jax.random.PRNGKey(train_cfg.seed);param_key,data_key=jax.random.split(key)
    params=init_params(param_key,model_cfg);tx=optimizer(train_cfg)
    return dict(params=params,optimizer=tx.init(params),ema=params,step=jnp.int32(0),rng=data_key,data_cursor=ct.zero((),2),skipped=jnp.int32(0)),tx

def make_step(model_cfg,train_cfg,tx):
    @jax.jit
    def update(state):
        key,next_key=jax.random.split(state["rng"])
        ids=random_lm_batch(key,train_cfg.batch_size,train_cfg.sequence_length,model_cfg.vocab)
        def objective(params):
            logits,_,stats=forward(params,ids,model_cfg,training=True)
            loss,acc=masked_ce(logits,ids)
            return loss,(acc,stats)
        (loss,(acc,stats)),grads=jax.value_and_grad(objective,has_aux=True)(state["params"])
        updates,opt=tx.update(grads,state["optimizer"],state["params"])
        candidate=optax.apply_updates(state["params"],updates)
        good=jnp.isfinite(loss)&jax.tree_util.tree_reduce(lambda a,x:a&jnp.all(jnp.isfinite(x)),grads,initializer=jnp.array(True))
        params=jax.tree_util.tree_map(lambda n,o:jnp.where(good,n,o),candidate,state["params"])
        opt=jax.tree_util.tree_map(lambda n,o:jnp.where(good,n,o),opt,state["optimizer"])
        ema=jax.tree_util.tree_map(lambda e,p:train_cfg.ema_decay*e+(1-train_cfg.ema_decay)*p,state["ema"],params)
        return {**state,"params":params,"optimizer":opt,"ema":ema,"step":state["step"]+1,"rng":next_key,"data_cursor":ct.add(state["data_cursor"],train_cfg.batch_size*train_cfg.sequence_length),"skipped":state["skipped"]+(~good).astype(jnp.int32)},dict(loss=loss,accuracy=acc,finite=good,stats=stats)
    return update

def train(model_cfg,train_cfg,out,*,resume=None,source_manifest=None):
    state,tx=init_training(model_cfg,train_cfg)
    if resume:state,_=load_checkpoint(resume,model_cfg,tx,train_cfg.to_dict())
    update=make_step(model_cfg,train_cfg,tx)
    history=[]
    while int(state["step"])<train_cfg.steps:
        state,metrics=update(state);history.append({k:jax.device_get(v) for k,v in metrics.items() if k!="stats"})
        if int(state["step"])%train_cfg.save_every==0:save_checkpoint(out,state,model_cfg,train_cfg.to_dict(),source_manifest)
    save_checkpoint(out,state,model_cfg,train_cfg.to_dict(),source_manifest)
    return state,history
