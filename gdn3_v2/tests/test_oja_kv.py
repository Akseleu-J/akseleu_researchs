import numpy as np
from gdn3.compression.oja_kv import OjaKV

def test_anchor_exact_and_roundtrip():
    c=OjaKV(4,5,capacity=4,rank=2,anchors=1);k=np.arange(4.);v=np.arange(5.)
    c.put("x",k,v,priority=1);np.testing.assert_array_equal(c.get("x")[0],k)
    d=OjaKV.from_state(c.state_dict());np.testing.assert_array_equal(d.get("x")[1],v)
