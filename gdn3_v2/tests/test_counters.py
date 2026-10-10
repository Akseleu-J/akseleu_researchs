import jax.numpy as jnp
from gdn3 import counters as c

def test_counter_carries_past_uint32():
    x=jnp.array([0xfffffffe,3],jnp.uint32);y=c.add(x,5)
    assert c.to_int(y)==(3<<32)+3

def test_counter_span_and_log():
    a=c.from_int(100,2);b=c.from_int(999,2);n=c.span_count(a,b)
    assert c.to_int(n)==900 and float(c.log_value(n))>6.7
