import jax.numpy as jnp


def create(e0=0, e1=0, e2=0):
    return jnp.array([e0, e1, e2]).astype(jnp.float32)


def unit(v):
    return v / jnp.linalg.norm(v)


def x(v):
    return v[0]


def y(v):
    return v[1]


def z(v):
    return v[2]


def pad(s, to=3):
    if isinstance(s, jnp.ndarray):
        d = to - len(s)
        return jnp.pad(s, (0, d))
    else:
        # assume scalar
        return jnp.zeros(to).at[0].set(s)


def equal(v1, v2):
    return jnp.linalg.norm(v1 - v2) < 1e-6
