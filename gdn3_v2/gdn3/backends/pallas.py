"""Pallas rank-one state update with an analytical Pallas reverse pass.

Each program owns one independent state, padded to TPU-compatible tiles.
No decay clamp or masked derivative is used at g=0.
"""
from functools import lru_cache
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _forward(s,u,r,z,g,ns,e):
    base=s[0,:,:]*jnp.exp(g[0,:])[:,None]
    residual=z[0,:]-jnp.sum(r[0,:][:,None]*base,axis=0)
    ns[0,:,:]=base+u[0,:][:,None]*residual[None,:]
    e[0,:]=residual

def _backward(s,u,r,z,g,dy,de,ds,du,dr,dz,dg):
    base=s[0,:,:]*jnp.exp(g[0,:])[:,None]
    err=z[0,:]-jnp.sum(r[0,:][:,None]*base,axis=0)
    derr=de[0,:]+jnp.sum(u[0,:][:,None]*dy[0,:,:],axis=0)
    db=dy[0,:,:]-r[0,:][:,None]*derr[None,:]
    ds[0,:,:]=db*jnp.exp(g[0,:])[:,None]
    du[0,:]=jnp.sum(dy[0,:,:]*err[None,:],axis=1)
    dr[0,:]=-jnp.sum(base*derr[None,:],axis=1)
    dz[0,:]=derr
    dg[0,:]=jnp.sum(db*base,axis=1)

@lru_cache(maxsize=32)
def _operator(n,k,v,interpret):
    mat=pl.BlockSpec((1,k,v),lambda i:(i,0,0))
    key=pl.BlockSpec((1,k),lambda i:(i,0))
    val=pl.BlockSpec((1,v),lambda i:(i,0))
    ms=jax.ShapeDtypeStruct((n,k,v),jnp.float32)
    ks=jax.ShapeDtypeStruct((n,k),jnp.float32)
    vs=jax.ShapeDtypeStruct((n,v),jnp.float32)
    raw=pl.pallas_call(_forward,out_shape=(ms,vs),grid=(n,),in_specs=(mat,key,key,val,key),out_specs=(mat,val),interpret=interpret)
    bwd=pl.pallas_call(_backward,out_shape=(ms,ks,ks,vs,ks),grid=(n,),in_specs=(mat,key,key,val,key,mat,val),out_specs=(mat,key,key,val,key),interpret=interpret)
    @jax.custom_vjp
    def op(s,u,r,z,g):return raw(s,u,r,z,g)
    def fwd(s,u,r,z,g):return raw(s,u,r,z,g),(s,u,r,z,g)
    def backward(saved,cotangents):return bwd(*saved,*cotangents)
    op.defvjp(fwd,backward)
    return op

def step(state,x,precision="highest",interpret=False):
    leading=state.shape[:-2];dk,dv=state.shape[-2:];n=math.prod(leading)
    k=(dk+127)//128*128;v=(dv+127)//128*128
    s=jnp.pad(state.astype(jnp.float32).reshape(n,dk,dv),((0,0),(0,k-dk),(0,v-dv)))
    def key(a):return jnp.pad(jnp.broadcast_to(a,leading+(dk,)).reshape(n,dk).astype(jnp.float32),((0,0),(0,k-dk)))
    z=jnp.pad(jnp.broadcast_to(x.z,leading+(dv,)).reshape(n,dv).astype(jnp.float32),((0,0),(0,v-dv)))
    ns,err=_operator(n,k,v,interpret)(s,key(x.u),key(x.r),z,key(x.g))
    return ns[:,:dk,:dv].reshape(state.shape),err[:,:dv].reshape(leading+(dv,))
