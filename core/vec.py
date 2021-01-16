from collections import namedtuple
from IPython import embed

import jax
from jax import lax, vmap
import jax.numpy as jnp


def create(x=0, y=0, z=0):
    return jnp.float32([x, y, z])


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
    return jnp.linalg.norm(v1 - v2) < 1e-8


def random(key):
    return jax.random.uniform(key, (3,), minval=-1, maxval=1)


def sphere(key):
    # random vector in unit sphere
    Value = namedtuple("Value", ["key", "vec"])

    def cf(val):
        return jnp.power(jnp.linalg.norm(val.vec), 2) > 1

    def bf(val):
        key, subkey = jax.random.split(val.key)
        vec = random(subkey)
        return Value(key=key, vec=vec)

    init_val = Value(key=key, vec=create(1, 1, 1))
    final_val = lax.while_loop(cf, bf, init_val)
    return final_val.vec


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    # v = sphere(key)
    vs = vmap(sphere)(jax.random.split(key, 10))
    def check(v): assert jnp.power(jnp.linalg.norm(v), 2) <= 1
    for v in vs:
        check(v)
