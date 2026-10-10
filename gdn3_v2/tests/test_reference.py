import numpy as np
from reference.gdn2 import apply,direct_matrix

def test_scalar_closed_form():
    s=np.array([[.4]],np.float64);step=dict(u=np.array([.3]),r=np.array([.2]),z=np.array([.7]),g=np.array([-.1]))
    out,read,_=apply(s,[step]);p,u=direct_matrix(step)
    np.testing.assert_allclose(out,p@s+u,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(read,out)
