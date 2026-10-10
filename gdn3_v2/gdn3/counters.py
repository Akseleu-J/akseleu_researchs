"""Little-endian uint32 counters; host can widen them without losing old spans."""
import jax.numpy as jnp

def zero(shape=(), words=2): return jnp.zeros(tuple(shape)+(words,), jnp.uint32)

def add(a, amount=1):
    carry = jnp.asarray(amount, jnp.uint32)
    out = []
    for i in range(a.shape[-1]):
        old = a[..., i]
        v = old + carry
        out.append(v)
        carry = (v < old).astype(jnp.uint32)
    return jnp.stack(out, -1)

def sub(a, b):
    borrow = jnp.zeros(jnp.broadcast_shapes(a.shape[:-1], b.shape[:-1]), jnp.uint32)
    out = []
    for i in range(a.shape[-1]):
        x, y = a[..., i], b[..., i]
        out.append(x-y-borrow)
        borrow = ((x < y) | ((x == y) & (borrow > 0))).astype(jnp.uint32)
    return jnp.stack(out, -1)

def equal(a, b): return jnp.all(a == b, axis=-1)

def less(a, b):
    out = jnp.zeros(jnp.broadcast_shapes(a.shape[:-1], b.shape[:-1]), bool)
    for i in range(a.shape[-1]): out = jnp.where(a[...,i] == b[...,i], out, a[...,i] < b[...,i])
    return out

def log_value(a):
    """Natural log with bounded f32 mantissa; log(0)=-inf, no huge integer cast."""
    val = jnp.full(a.shape[:-1], -jnp.inf, jnp.float32)
    for i in range(a.shape[-1]):
        w = a[...,i].astype(jnp.float32)
        val = jnp.logaddexp(val, jnp.where(w > 0, jnp.log(jnp.maximum(w, 1)) + (32*i)*jnp.log(2.0), -jnp.inf))
    return val

def log1p(a): return jnp.logaddexp(0.0, log_value(a))
def span_count(start, end): return add(sub(end, start))
def small_at_least(a, n): return log_value(a) >= jnp.log(jnp.float32(n))
def bit(a, index):
    word, offset = divmod(index, 32)
    return jnp.zeros(a.shape[:-1], bool) if word >= a.shape[-1] else ((a[...,word] >> offset) & 1).astype(bool)

def trailing_zeros(a, maximum):
    out = jnp.zeros(a.shape[:-1], jnp.int32)
    still = jnp.ones(a.shape[:-1], bool)
    for i in range(maximum):
        still = still & ~bit(a, i)
        out = out + still.astype(jnp.int32)
    return out

def from_int(value, words=2):
    if value < 0 or value.bit_length() > 32*words: raise ValueError("counter width insufficient")
    return jnp.asarray([(value >> (32*i)) & 0xffffffff for i in range(words)], jnp.uint32)

def to_int(a):
    return sum(int(w) << (32*i) for i, w in enumerate(a.tolist()))
