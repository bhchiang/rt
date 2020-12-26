import jax.numpy as jnp


def vec(e0=0, e1=0, e2=0):
    return jnp.array([e0, e1, e2]).astype(jnp.float32)


def unit(v):
    return v / jnp.linalg.norm(v)


def x(v):
    return v[0]


def y(v):
    return v[1]


def z(v):
    return v[2]


def assert_vec_eq(v1, v2):
    assert jnp.linalg.norm(v1 - v2) < 1e-6
