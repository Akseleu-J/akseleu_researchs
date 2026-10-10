import math
import jax
import jax.numpy as jnp
from .precision import linear,l2norm,rmsnorm,layernorm
from .cells.kda import gate

def init_layer(key,cfg):
    keys=iter(jax.random.split(key,64))
    def matrix(a,b,gain=1.):return jax.random.normal(next(keys),(a,b))*gain/math.sqrt(a)
    d,H,K,V,M=cfg.d_model,cfg.n_heads,cfg.d_key,cfg.d_value,cfg.delta_steps
    pk,pv=H*K,H*V;og=1/math.sqrt(2*cfg.n_layers)
    if cfg.backbone=="atomic":
        core={}
        for name,width in [("q",pk),("k",pk),("v",pv),("b",pk),("w",pv),("g",pk),("o",d),("up",cfg.mlp_mult*d),("down",d)]:
            core[name]={"w":matrix(d,width),"b":jnp.zeros(width)}
        return {"atomic_core":core,"conv_q":jnp.zeros((cfg.d_conv-1,pk)),"conv_k":jnp.zeros((cfg.d_conv-1,pk)),"conv_v":jnp.zeros((cfg.d_conv-1,pv))}
    p=dict(norm1=jnp.ones(d),norm2=jnp.ones(d),wq=matrix(d,pk),wk=matrix(d,pk),wv=matrix(d,pv),wb=matrix(d,pk,.5),bb=jnp.zeros(pk),ww=matrix(d,pv,.5),bw=jnp.zeros(pv),wog=matrix(d,pv),onorm=jnp.ones(pv),wo=matrix(pv,d,og),w_gate=matrix(d,cfg.mlp_mult*d),w_up=matrix(d,cfg.mlp_mult*d),w_down=matrix(cfg.mlp_mult*d,d,og))
    for name,width in [("q",pk),("k",pk),("v",pv)]:
        p["conv_"+name]=jnp.zeros((width,cfg.d_conv)).at[:,-1].set(1)+.02*jax.random.normal(next(keys),(width,cfg.d_conv))
        p["conv_b"+name]=jnp.zeros(width)
    gate_width=H if cfg.decay_mode in {"head","channel"} else (cfg.decay_rank if cfg.decay_mode=="kda_lowrank" else pk)
    p.update(wd=matrix(d,gate_width,.1),bd=jnp.zeros(gate_width),decay_a=jnp.broadcast_to(jnp.linspace(math.log(math.log(2)/cfg.hl_min),math.log(math.log(2)/cfg.hl_max),K),(H,K)))
    if cfg.decay_mode=="kda_lowrank":p["wd_up"]=matrix(cfg.decay_rank,pk,.1)
    for name,width in [("k",pk),("v",pv),("b",pk),("w",pv)]:
        p["extra_"+name]=jax.random.normal(next(keys),(M-1,d,width))/math.sqrt(d)
        p["extra_b"+name]=jnp.zeros((M-1,width))
    if cfg.qed:p["qed_strength"]=jnp.full(H,cfg.qed_init)
    if cfg.eda:p.update(we=matrix(d,pk),eda_strength=jnp.full(H,cfg.eda_init))
    if cfg.cell=="qdelta":p["feedback"]=jnp.full(H,-2.)
    if cfg.cell=="pdn":p.update(pdn_alpha=matrix(d,H,.1),pdn_beta=matrix(d,H,.1),pdn_mu=jnp.ones(H))
    if cfg.ccq!="off":p.update(ccq_w=jnp.zeros((H,K)),ccq_b=jnp.full(H,cfg.ccq_init))
    A=cfg.ha;N=cfg.sse_partitions
    if A:
        p.update(h_init=jnp.zeros((A,N,K,V)),arch_gate=jnp.full(A,cfg.gate_init),route_tau=jnp.full(A,math.log(math.expm1(cfg.router_temperature))),route_age=jnp.full(A,-30. if cfg.router_age==0 else math.log(math.expm1(cfg.router_age))),route_null=jnp.full(A,cfg.router_null),router_sketch=jax.random.normal(next(keys),(V,cfg.router_dim))/math.sqrt(cfg.router_dim),dla_threshold=jnp.asarray(cfg.dla_threshold))
        if cfg.lambda_mode=="shared":
            F=cfg.lambda_hidden
            p.update(lambda_x=matrix(d,F,.1),lambda_head=jnp.zeros((A,F)),lambda_level=matrix(4,F,.1),lambda_out=jnp.zeros((A,F)),lambda_bias=jnp.array(.54))
        elif cfg.lambda_mode.startswith("paper"):
            L=cfg.lambda_paper_levels;F=cfg.lambda_hidden
            p.update(lambda_proj=matrix(d,A*L,.1),lambda_w1=matrix(L,F),lambda_w2=jnp.zeros((F,L)),lambda_bias=jnp.array(.54))
        if N>1:p.update(sse_w=matrix(d,A*N,.1),sse_bias=jnp.zeros((A,N)))
        if cfg.hola_size:p.update(hola_qnorm=jnp.ones((A,K)),hola_knorm=jnp.ones((A,K)),hola_null=jnp.zeros(A),hola_gate=jnp.full(A,cfg.hola_gate_init))
        p["external_gate"]=jnp.full(A,cfg.external_gate_init)
    dtype=jnp.bfloat16 if cfg.precision.storage=="bfloat16" else jnp.float32
    return jax.tree_util.tree_map(lambda x:x.astype(dtype),p)

def init_params(key,cfg):
    keys=jax.random.split(key,cfg.n_layers+1)
    out=dict(embed=.02*jax.random.normal(keys[0],(cfg.vocab,cfg.d_model)),layers=[init_layer(keys[i+1],cfg.for_layer(i)) for i in range(cfg.n_layers)],final_norm=jnp.ones(cfg.d_model))
    if cfg.backbone=="atomic":
        out["head"]={"w":jax.random.normal(keys[0],(cfg.d_model,cfg.vocab))*.02,"b":jnp.zeros(cfg.vocab)}
    return out

def features(p,c,x,cfg):
    H,K,V,M=cfg.n_heads,cfg.d_key,cfg.d_value,cfg.delta_steps
    mode=cfg.precision.dot
    if cfg.backbone=="atomic":
        a=p["atomic_core"];h=layernorm(x)
        proj=lambda name:linear(h,a[name]["w"],mode)+a[name]["b"]
        q=proj("q").reshape(H,K);kl=proj("k").reshape(H,K);v=proj("v").reshape(H,V)
        q,k=l2norm(q,cfg.norm_group_key),l2norm(kl,cfg.norm_group_key)
        b=jax.nn.sigmoid(proj("b").reshape(H,K));w=proj("w").reshape(H,V)
        g=-jnp.minimum(jax.nn.softplus(proj("g").reshape(H,K)),cfg.g_max)
        conv={name:c[name] for name in ("conv_q","conv_k","conv_v")}
    else:
        h=rmsnorm(x,p["norm1"]);vals=[];conv={}
        for name,dim in [("q",K),("k",K),("v",V)]:
            raw=linear(h,p["w"+name],mode)
            history=jnp.concatenate((c["conv_"+name],raw[None]),axis=0)
            out=jnp.sum(history*p["conv_"+name].T,axis=0)+p["conv_b"+name]
            vals.append(jax.nn.silu(out).reshape(H,dim))
            conv["conv_"+name]=history[1:]
        q,kl,v=vals;q,k=l2norm(q,cfg.norm_group_key),l2norm(kl,cfg.norm_group_key)
        b=jax.nn.sigmoid(linear(h,p["wb"],mode)+p["bb"]).reshape(H,K)
        w=jax.nn.sigmoid(linear(h,p["ww"],mode)+p["bw"]).reshape(H,V)
        g=jnp.broadcast_to(gate(h,p,cfg),(H,K))
    kvals=[k];klogs=[kl];vvals=[v];bvals=[b];wvals=[w]
    for i in range(M-1):
        ex=lambda name:linear(h,p["extra_"+name][i],mode)+p["extra_b"+name][i]
        ki=jax.nn.silu(ex("k").reshape(H,K));klogs.append(ki)
        kvals.append(l2norm(ki,cfg.norm_group_key));vvals.append(jax.nn.silu(ex("v").reshape(H,V)))
        bvals.append(jax.nn.sigmoid(ex("b").reshape(H,K)));wvals.append(jax.nn.sigmoid(ex("w").reshape(H,V)))
    f=dict(q=q,k=jnp.stack(kvals,1),k_logits=jnp.stack(klogs,1),v=jnp.stack(vvals,1),b=jnp.stack(bvals,1),w=jnp.stack(wvals,1),g=g)
    if cfg.qed:f["qed_strength"]=jnp.maximum(p["qed_strength"],0)
    if cfg.eda:f.update(erase=l2norm(linear(h,p["we"],mode).reshape(H,K)),gamma=jnp.clip(p["eda_strength"],0,1))
    if cfg.cell=="qdelta":f["feedback"]=jax.nn.sigmoid(p["feedback"])
    return h,f,conv
