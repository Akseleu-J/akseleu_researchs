import jax.numpy as jnp
from gdn3.memory.router import select

def test_router_tie_breaks_on_multword_id_and_has_null():
    scores=jnp.array([[1.,1.,-2.]]);valid=jnp.array([[True,True,True]])
    ids=jnp.array([[5,0],[2,0],[1,1]],jnp.uint32)
    idx,w,null=select(scores,valid,ids,2,jnp.array([[0.]]))
    assert idx.tolist()==[[1,0]] and float(null[0,0])>0 and float(w.sum()+null.sum())>0.99
