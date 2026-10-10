import numpy as np
import jax.numpy as jnp
from gdn3.cells.qed import address
from gdn3.cells.osdn import paper_gradient,generalized_gradient,feasible

def test_qed_projector_is_orthogonal_for_unit_key():
    k=jnp.array([1.,0.]);q=jnp.array([0.25,0.75]);b=jnp.array([0.2,0.8]);r=address(k,q,b,jnp.array(.5))
    np.testing.assert_allclose(float(k@ (r-b*k)),0.,atol=1e-6)

def test_osdn_gradient_shapes_and_box():
    k=jnp.array([.5,.25]);r=jnp.array([.3,.7]);d=jnp.array([1.,1.]);g=generalized_gradient(k,r,d)
    assert g.shape==d.shape and bool(jnp.all(jnp.isfinite(g)))
    p=paper_gradient(k,.4,d);assert p.shape==d.shape
    fixed=feasible(jnp.array([9.,-4.]),k,r,.25,2.,.01);assert bool(jnp.all((fixed>=.25)&(fixed<=2.)))
