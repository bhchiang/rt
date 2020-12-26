import jax.numpy as jnp


def vec(e0=0, e1=0, e2=0):
    return jnp.array([e0, e1, e2]).astype(float)


def unit(v):
    return v / jnp.linalg.norm(v)


def assert_vec_eq(v1, v2):
    assert jnp.linalg.norm(v1 - v2) < 1e-6
