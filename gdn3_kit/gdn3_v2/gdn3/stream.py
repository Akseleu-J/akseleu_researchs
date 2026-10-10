"""One causal token transaction, shared by training, prefill and decoding."""
import jax
import jax.numpy as jnp
from . import counters as ct
from .parameters import features
from .precision import einsum,linear,l2norm,rmsnorm,layernorm
from .cells.base import Step,apply_steps
from .cells.composition import build_steps,update_preconditioner
from .cells.preconditioned import update_moments
from .memory import bank,hierarchy,loglinear,dla,router,adaptive,ccq,sse,hola
from .compression import leapquant
from .health import observe
from .backends import primitive

STATS=("nonfinite","max_abs","overflow","merges","promotions","closed","boundary_soft","sse_balance","router_null","hola_null")
def empty_stats():return {k:jnp.float32(0) for k in STATS}

def state_value(value,cfg,which):
    enabled=cfg.quant=="leap" and (which=="active" or cfg.quant_scope=="all")
    return leapquant.materialize(value,cfg) if enabled else value

def _warm_initial(c,initial,cfg):
    state=initial
    if cfg.warm:
        start=(c["warm_cursor"]-c["warm_count"])%cfg.warm
        for j in range(cfg.warm):
            idx=(start+j)%cfg.warm
            steps=tuple(Step(*(c["warm_"+key][idx,i] for key in ("u","r","z","g")),write=i>=int(cfg.eda)) for i in range(cfg.substeps))
            state=jax.lax.cond(j<c["warm_count"],lambda s:apply_steps(s,steps,cfg.precision.dot)[0],lambda s:s,state)
    return state

def _save_warm(c,steps,cfg):
    if not cfg.warm:return c
    out=dict(c)
    for key in ("u","r","z","g"):
        values=jnp.stack([getattr(s,key) for s in steps])
        out["warm_"+key]=out["warm_"+key].at[c["warm_cursor"]].set(values)
    out["warm_cursor"]=(c["warm_cursor"]+1)%cfg.warm
    out["warm_count"]=jnp.minimum(c["warm_count"]+1,cfg.warm)
    return out

def _external_read(q,records,precision):
    if records is None or records["k"].shape[1]==0:return jnp.zeros(q.shape[:-1]+(records["v"].shape[-1] if records is not None else q.shape[-1],))
    logits=einsum("hd,hnd->hn",q,records["k"],precision)/(q.shape[-1]**.5)
    logits=jnp.where(records["valid"],logits,-jnp.inf)
    weights=jax.nn.softmax(jnp.concatenate((logits,jnp.zeros(logits.shape[:-1]+(1,))),-1),-1)
    return einsum("hn,hnv->hv",weights[:,:-1],records["v"],precision)

def layer_token(p,c,x,position,cfg,training=False,retrieved=None):
    mode=cfg.precision.dot
    prim=primitive(cfg)
    h,f,conv=features(p,c,x,cfg)
    out={**c,**conv};stats=empty_stats()
    if cfg.cell=="pdn":
        A,scale=update_moments(c["pdn_A"],f["k"][:,0],jax.nn.sigmoid(linear(h,p["pdn_alpha"],mode)),jax.nn.sigmoid(linear(h,p["pdn_beta"],mode)),jax.nn.softplus(p["pdn_mu"]),cfg.pdn_x)
        out["pdn_A"]=A;f["pdn_scale"]=scale
    hc=state_value(c["hc"],cfg,"hc")
    steps=build_steps(f,c["dc"],cfg)
    if cfg.quant=="leap" and cfg.quant_scope=="all":
        out["hc"],hc_next,surprise=leapquant.append(c["hc"],steps,cfg)
    else:
        hc_next,surprise=apply_steps(hc,steps,mode,prim)
        out["hc"]=hc_next
    out["dc"]=update_preconditioner(f,c["dc"],hc,steps,cfg)
    qread=f["q"]
    if cfg.ccq!="off":
        out["ccq"]=ccq.update(c["ccq"],f["k"][:,0])
        strength=jax.nn.sigmoid(jnp.sum(f["q"]*p["ccq_w"],axis=-1)+p["ccq_b"])
        cleaned=ccq.clean(f["q"],out["ccq"],strength,mode)
        if cfg.ccq=="all":qread=cleaned
    else:cleaned=f["q"]
    output=einsum("hd,hdv->hv",qread,hc_next,mode)/(float(cfg.norm_group_key or cfg.d_key)**.5)
    A=cfg.ha
    if A:
        heads=jnp.asarray(cfg.heads,jnp.int32)
        af={name:jnp.take(v,heads,axis=0) for name,v in f.items()}
        aq=cleaned[heads] if cfg.ccq!="off" else af["q"]
        if cfg.sse_partitions>1:
            gate_logits=linear(h,p["sse_w"],mode).reshape(A,cfg.sse_partitions)+p["sse_bias"]
            partition_weights,partition_mask=sse.select(gate_logits,cfg.sse_top_k,cfg.sse_shared)
            stats["sse_balance"]=sse.balance(partition_weights,partition_mask)
        else:
            partition_weights=jnp.ones((A,1));partition_mask=jnp.ones((A,1),bool)
        af=sse.expand(af,partition_weights,partition_mask,cfg)
        read_partition=partition_weights*partition_mask
        if cfg.memory=="loglinear":
            archive_steps=tuple(Step(s.u[heads,None],s.r[heads,None],s.z[heads,None],s.g[heads,None],s.write) for s in steps)
            out["levels"]=loglinear.update(c["levels"],position,archive_steps,p["router_sketch"],mode)
            bins=out["levels"]
            levels=jnp.arange(bins["valid"].shape[0])
            spans=ct.log_value(ct.span_count(bins["start"],bins["end"]))
            lambdas=adaptive.weights(h,p,levels,spans,cfg)
            memory_out=einsum("hd,nhdv->hnv",aq,bins["state"][:,:,0],mode)
            memory_out=jnp.sum(memory_out*lambdas[...,None]*bins["valid"][None,:,None],axis=1)/(cfg.d_key**.5)
            # Interpolate baseline -> lifted recurrence. λ=1 gives baseline, not a duplicated branch.
            output=output.at[heads].add(jnp.tanh(p["arch_gate"])[...,None]*(memory_out-output[heads]))
        else:
            current=state_value(c["active"],cfg,"active")
            arch_steps=build_steps(af,c["da"],cfg)
            tentative,_=apply_steps(current,arch_steps,mode,prim)
            info=dla.information(current,tentative)
            close,mix,soft=dla.boundary(info,c["count"],jnp.maximum(p["dla_threshold"],0),cfg,training)
            stats["boundary_soft"]=soft
            raw_init=p["h_init"].astype(jnp.float32) if cfg.learned_init else jnp.zeros_like(current)
            initial=_warm_initial(c,raw_init,cfg)
            reset_d=jnp.ones_like(c["da"])
            fresh_steps=build_steps(af,reset_d,cfg)
            fresh,_=apply_steps(initial,fresh_steps,mode,prim)
            bits=cfg.quant_bits if cfg.quant!="none" else 0
            rank=cfg.quant_rank if bits else 0
            def insert(args):
                recent,levels=args
                closing=hc[heads,None] if cfg.mode=="snap" else current
                start=ct.zero((),position.shape[-1]) if cfg.mode=="snap" else c["start"]
                value=bank.item(closing,start,ct.sub(position,ct.from_int(1,position.shape[-1])),c["density"],p["router_sketch"],bits,rank)
                return hierarchy.append(recent,levels,value,position,cfg,p["router_sketch"])
            def keep(args):return *args,dict(overflow=jnp.array(False),merges=jnp.int32(0),promotions=jnp.int32(0))
            recent,levels,event=jax.lax.cond(close,insert,keep,(c["recent"],c["levels"]))
            if cfg.bank_decay=="aligned":
                recent=bank.age_decay(recent,f["g"][heads]);levels=bank.age_decay(levels,f["g"][heads])
            out["recent"],out["levels"]=recent,levels
            stats.update({name:event[name].astype(jnp.float32) for name in event})
            stats["closed"]=close.astype(jnp.float32)
            chosen_steps=tuple(Step(*(jnp.where(close,a,b) for a,b in zip(fs[:4],ts[:4])),write=fs.write) for fs,ts in zip(fresh_steps,arch_steps))
            chosen_state=jnp.where(close,initial,current)
            chosen_d=jnp.where(close,reset_d,c["da"])
            out["da"]=update_preconditioner(af,chosen_d,chosen_state,chosen_steps,cfg)
            if cfg.quant=="leap":
                seed=jax.lax.cond(close,lambda _:leapquant.init(initial,cfg),lambda _:c["active"],None)
                out["active"],active,_=leapquant.append(seed,chosen_steps,cfg)
            else:
                active=(1-mix)*tentative+mix*fresh
                out["active"]=active
            out["start"]=jnp.where(close,position,c["start"])
            count=jnp.where(close,1,c["count"]+1)
            out["density"]=jnp.where(close,info,c["density"]+(info-c["density"])/count.astype(jnp.float32))
            out["count"]=count
            out=_save_warm(out,chosen_steps,cfg)
            if cfg.mode=="local":
                local=einsum("hd,hpdv->hpv",af["q"][:,0],active,mode)
                output=output.at[heads].set(jnp.sum(local*read_partition[...,None],axis=1)/(cfg.d_key**.5))
            banks=hierarchy.candidates(recent,levels)
            level_ids=jnp.concatenate((-jnp.ones(cfg.recent_slots,jnp.int32),jnp.arange(levels["valid"].shape[0])))
            logspans=ct.log_value(ct.span_count(banks["start"],banks["end"]))
            weights=adaptive.weights(h,p,level_ids,logspans,cfg)
            memory_out,route_stats=router.read(aq,banks,p,cfg,position,weights,read_partition)
            output=output.at[heads].add(jnp.tanh(p["arch_gate"])[...,None]*memory_out)
            stats["router_null"]=jnp.mean(route_stats["null_mass"])
        if cfg.hola_size:
            out["hola"]=hola.insert(c["hola"],f["k"][heads,0],f["v"][heads,0],surprise[heads],position)
            ho,nu=hola.read(out["hola"],f["q"][heads],p,mode)
            output=output.at[heads].add(jnp.tanh(p["hola_gate"])[...,None]*ho)
            stats["hola_null"]=jnp.mean(nu)
        if retrieved is not None:
            eo=_external_read(f["q"][heads],retrieved,mode)
            output=output.at[heads].add(jnp.tanh(p["external_gate"])[...,None]*eo)
    flat=output.reshape(cfg.value_width)
    if cfg.backbone=="atomic":
        a=p["atomic_core"]
        x=x+linear(flat,a["o"]["w"],mode)+a["o"]["b"]
        ff=jax.nn.gelu(linear(layernorm(x),a["up"]["w"],mode)+a["up"]["b"])
        x=x+linear(ff,a["down"]["w"],mode)+a["down"]["b"]
    else:
        y=linear(rmsnorm(flat,p["onorm"])*jax.nn.silu(linear(h,p["wog"],mode)),p["wo"],mode)
        x=x+y
        hn=rmsnorm(x,p["norm2"])
        x=x+linear(jax.nn.silu(linear(hn,p["w_gate"],mode))*linear(hn,p["w_up"],mode),p["w_down"],mode)
    health=observe(hc_next,x,output)
    stats.update({name:health[name].astype(jnp.float32) for name in health})
    return x,out,stats
