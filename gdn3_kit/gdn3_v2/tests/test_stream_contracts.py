import pytest
import jax
import jax.numpy as jnp
from gdn3.config import GDN3Config
from gdn3.parameters import init_params
from gdn3.model import prefill,decode_step

def test_masked_token_is_noop():
    c=GDN3Config.baseline(vocab=16,d_model=8,n_heads=2,d_key=4,d_value=4,n_layers=1,d_conv=2)
    p=init_params(jax.random.PRNGKey(0),c);_,s=prefill(p,jnp.array([[1,2]]),c)
    _,t=decode_step(p,s,jnp.array([3]),c,valid=jnp.array([False]))
    for a,b in zip(jax.tree_util.tree_leaves(s),jax.tree_util.tree_leaves(t)):jnp.testing.assert_array_equal(a,b)
