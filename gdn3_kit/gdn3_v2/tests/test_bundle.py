import numpy as np
from gdn3.persistence.bundle import save_bundle,load_bundle

def test_array_bundle_roundtrip(tmp_path):
    path=tmp_path/"x.bundle";tree={"x":np.arange(5,dtype=np.float32),"ids":np.array([1,2],np.uint32)}
    save_bundle(path,tree,{"schema_test":1});got,meta=load_bundle(path,as_jax=False)
    np.testing.assert_array_equal(got["x"],tree["x"]);assert meta["schema_test"]==1
