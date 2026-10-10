"""Independent NumPy float64 recurrence oracle for the asymmetric ABI."""
import numpy as np

def apply(state,steps,scale=1.):
    s=np.asarray(state,dtype=np.float64).copy()
    if s.ndim<2:raise ValueError("state must end in (Dk,Dv)")
    surprise=0.
    for i,step in enumerate(steps):
        u,r,z,g=map(lambda k:np.asarray(step[k],dtype=np.float64),("u","r","z","g"))
        if u.shape!=r.shape or u.shape!=g.shape or z.shape!=s.shape[-1:] or u.shape!=s.shape[-2:-1]:raise ValueError("substep dimension mismatch")
        dec=np.exp(g)[..., :,None]*s
        residual=z-np.einsum("...d,...dv->...v",r,dec)
        s=dec+u[..., :,None]*residual[...,None,:]
        if step.get("write",True):surprise+=float(np.linalg.norm(u)*np.linalg.norm(residual))
    return s,scale*s,surprise

def direct_matrix(step):
    u,r,z,g=(np.asarray(step[k],np.float64) for k in ("u","r","z","g"))
    d=np.diag(np.exp(g));return (np.eye(len(u))-np.outer(u,r))@d,np.outer(u,z)
