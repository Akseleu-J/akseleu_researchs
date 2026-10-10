"""Pallas (atomic_gdn2) vs serial backend. TPU only: python tests/test_pallas.py   (skips elsewhere)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from gdn3.backend import serial_backend, pallas_backend


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))


def main():
    if jax.default_backend() not in ("tpu",):
        print("SKIP: not on TPU"); return 0
    B, H, L, D = 1, 2, 512, 64
    ks = jax.random.split(jax.random.PRNGKey(0), 6)
    l2 = lambda t: t * jax.lax.rsqrt(jnp.sum(t * t, -1, keepdims=True) + 1e-6)
    a = (l2(jax.random.normal(ks[0], (B, H, L, D))), l2(jax.random.normal(ks[1], (B, H, L, D))),
         0.5 * jax.random.normal(ks[2], (B, H, L, D)), jax.nn.sigmoid(jax.random.normal(ks[3], (B, H, L, D))),
         jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, L, D))), -0.2 * jax.nn.softplus(jax.random.normal(ks[5], (B, H, L, D))))
    sb, pb = serial_backend(D), pallas_backend(D)
    o1, h1 = sb.call(*a); o2, h2 = pb.call(*a)
    e_o, e_h = rel(o2, o1), rel(h2, h1)
    print(f"forward  o={e_o:.2e} h={e_h:.2e}")
    ct = jax.random.normal(jax.random.PRNGKey(9), o1.shape)
    g1 = jax.grad(lambda *x: jnp.sum(sb.call(*x)[0] * ct), argnums=range(6))(*a)
    g2 = jax.grad(lambda *x: jnp.sum(pb.call(*x)[0] * ct), argnums=range(6))(*a)
    errs = {n: rel(y, x) for n, x, y in zip("qkvwbg", g1, g2)}
    print("grads", {k: f"{v:.2e}" for k, v in errs.items()})
    ok = e_o < 2e-3 and e_h < 2e-3 and max(errs.values()) < 5e-3
    print("PASS" if ok else "FAIL"); return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
