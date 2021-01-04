from IPython import embed

import jax
from jax import lax, vmap
import jax.numpy as jnp


def create(e0=0, e1=0, e2=0):
    return jnp.float32(e0, e1, e2])


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


def random(key):
    return jax.random.uniform(key, (1, 3))


def sphere(key):
    # random vector in unit sphere

    def pack(key, v): return jnp.vstack(([0, *key], v))

    def unpack(d):
        (_, *key), v = d
        return jnp.uint32(key), v

    def cf(d):
        _, v = unpack(d)
        return jnp.linalg.norm(v) > 1

    def bf(d):
        key, _ = unpack(d)
        key, subkey = jax.random.split(key)
        v = random(subkey)
        return pack(key, v)

    iv = pack(key, create(1, 1, 1))
    # embed()
    fv = lax.while_loop(cf, bf, iv)
    _, v = unpack(fv)
    return v


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    # v = sphere(key)
    vs = vmap(sphere)(jax.random.split(key, 10))
    def check(v): assert jnp.linalg.norm(v) <= 1
    for v in vs:
        check(v)
    embed()
