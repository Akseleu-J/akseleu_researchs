import jax
import jax.numpy as jnp
from ..precision import rmsnorm,einsum
from .. import counters as ct

def init(H,K,V,W,C,words):
    def records(n):return dict(k=jnp.zeros((H,n,K)),v=jnp.zeros((H,n,V)),score=jnp.full((H,n),-jnp.inf),ids=ct.zero((H,n),words),valid=jnp.zeros((H,n),bool))
    return dict(priority=records(W),block=records(C),cursor=jnp.int32(0))

def insert(cache,k,v,score,position):
    p,b=cache["priority"],cache["block"]
    H=k.shape[0]
    idx=jnp.argmin(jnp.where(p["valid"],p["score"],-jnp.inf),axis=-1)
    h=jnp.arange(H)
    admit=(~p["valid"][h,idx])|(score>p["score"][h,idx])
    old={key:value[h,idx] for key,value in p.items()}
    item=dict(k=k,v=v,score=score,ids=jnp.broadcast_to(position,(H,position.shape[-1])),valid=jnp.ones((H,),bool))
    p={key:value.at[h,idx].set(jnp.where(admit.reshape((H,)+(1,)*(old[key].ndim-1)),item[key],old[key])) for key,value in p.items()}
    cursor=cache["cursor"]
    b={key:value.at[:,cursor].set(item[key]) for key,value in b.items()}
    return dict(priority=p,block=b,cursor=(cursor+1)%b["valid"].shape[1])

def read(cache,q,p,precision="highest"):
    pr,bl=cache["priority"],cache["block"]
    dup=jnp.any(jnp.all(pr["ids"][:,:,None,:]==bl["ids"][:,None,:,:],-1)&bl["valid"][:,None,:],axis=-1)
    k=jnp.concatenate((pr["k"],bl["k"]),axis=1)
    v=jnp.concatenate((pr["v"],bl["v"]),axis=1)
    valid=jnp.concatenate((pr["valid"]&~dup,bl["valid"]),axis=1)
    qn=rmsnorm(q,p["hola_qnorm"])
    kn=rmsnorm(k,p["hola_knorm"][:,None,:])
    logits=einsum("hd,hnd->hn",qn,kn,precision)/(q.shape[-1]**0.5)
    logits=jnp.where(valid,logits,-jnp.inf)
    weights=jax.nn.softmax(jnp.concatenate((logits,p["hola_null"][:,None]),-1),-1)
    return einsum("hn,hnv->hv",weights[:,:-1],v,precision),weights[:,-1]
