import jax.numpy as jnp


def empty():
    return jnp.zeros((3, 3))


def create(t, p, ff, n):
    return jnp.array([p, n, [t, ff, 1]])


def unpack(rc):
    p, n, (t, ff, _) = rc
    return (t, p, ff, n)


def exists(rc):
    return rc.any()
