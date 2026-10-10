import jax.numpy as jnp
from gdn3.config import GDN3Config
from gdn3.memory.dla import information,boundary

def test_identical_state_has_zero_information():
    x=jnp.ones((2,3,4));assert float(information(x,x))<1e-5

def test_boundary_minmax_contract():
    c=GDN3Config(segment_min=2,segment_max=4)
    assert not bool(boundary(jnp.array(10.),jnp.array(1),jnp.array(.1),c)[0])
    assert bool(boundary(jnp.array(0.),jnp.array(4),jnp.array(.1),c)[0])
