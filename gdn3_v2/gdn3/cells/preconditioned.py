"""PDN practical Eq.9; exact SM lives in the independent reference package."""
import jax.numpy as jnp

def update_moments(A,k,alpha,beta,mu,x):
    A=alpha[...,None]*A+beta[...,None]*k*k
    ell=jnp.log(jnp.maximum(A,1e-12))-mu[...,None]
    squash=ell/(1+jnp.abs(ell))
    return A,jnp.exp(-jnp.log(x)*squash)
