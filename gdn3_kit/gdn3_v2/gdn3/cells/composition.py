"""Addresses are composed once; memory branches select every field by head index."""
import jax
import jax.numpy as jnp
from .base import Step
from .qed import address
from .osdn import paper_gradient, generalized_gradient, feasible

def build_steps(f, d, cfg):
    steps=[]
    first_g=f["g"]
    if cfg.eda:
        e=f["erase"]
        steps.append(Step(e*f.get("u_scale",jnp.ones(e.shape[:-1]))[...,None],f["gamma"][...,None]*e,jnp.zeros_like(f["v"][...,0,:]),first_g,False))
        first_g=jnp.zeros_like(first_g)
    for i in range(cfg.delta_steps):
        k,v,b,w=f["k"][...,i,:],f["v"][...,i,:],f["b"][...,i,:],f["w"][...,i,:]
        r=b*k
        if cfg.qed: r=address(k,f["q"],b,f["qed_strength"])
        if cfg.osdn=="paper":
            beta=jnp.mean(b,-1)
            u=beta[...,None]*d[...,i,:]*k
            r,z=k,v
        else:
            u=k if cfg.osdn=="off" else d[...,i,:]*k
            z=w*v
        if cfg.cell=="fg2":
            u=jnp.sqrt(b)*k
            r=u
            z=jnp.sqrt(w)*v
        elif cfg.cell=="qdelta":
            beta=jnp.mean(b,-1)
            u=beta[...,None]*k
            r=k+f["feedback"][...,None]*f["q"]
            z=v
        elif cfg.cell=="pdn":
            u=f["pdn_scale"]*k*jnp.mean(b,-1)[...,None]
            r,z=k,v
        u=u*f.get("u_scale",jnp.ones(k.shape[:-1]))[...,None]
        steps.append(Step(u,r,z,first_g if i==0 else jnp.zeros_like(first_g)))
    return tuple(steps)

def update_preconditioner(f,d,state,steps,cfg):
    if cfg.osdn=="off": return d
    out=[]
    # d is per slot; generalized zero-residual hold refers to that slot's pre-update state.
    from .base import apply_step
    s=state
    off=int(cfg.eda)
    if off: s,_=apply_step(s,steps[0],cfg.precision.dot)
    for i in range(cfg.delta_steps):
        k=f["k"][...,i,:]; b=f["b"][...,i,:]; step=steps[i+off]
        ns,res=apply_step(s,step,cfg.precision.dot)
        if cfg.osdn=="paper":
            grad=paper_gradient(k,jnp.mean(b,-1),d[...,i,:])
            nxt=jnp.clip(d[...,i,:]-cfg.osdn_lr*grad,cfg.osdn_min,cfg.osdn_max)
        else:
            base_k=k*f.get("u_scale",jnp.ones(k.shape[:-1]))[...,None]
            grad=generalized_gradient(base_k,step.r,d[...,i,:])
            nxt=feasible(d[...,i,:]-cfg.osdn_lr*grad,base_k,step.r,cfg.osdn_min,cfg.osdn_max,cfg.osdn_margin)
            hold=(jnp.sum(step.r*step.r,-1)<1e-12)|(jnp.sum(res*res,-1)<1e-24)
            nxt=jnp.where(hold[...,None],d[...,i,:],nxt)
        out.append(nxt); s=ns
    return jnp.stack(out,-2)
