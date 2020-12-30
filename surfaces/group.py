from functools import partial

import jax.numpy as jnp
from jax import lax, vmap

from . import sphere, record


def create(surfaces):
    return jnp.array(surfaces)


def hit(r, t_min, t_max, g):
    # trace all hits (no hit = empty record)
    h = partial(sphere.hit, r, t_min, t_max)
    rcs = vmap(h)(g)
    # print(g)
    # print(rcs)

    # lax scan for earliest hit time
    def f(carry, rc):
        t, *_ = record.unpack(carry)
        t_, *_ = record.unpack(rc)
        c = lax.bitwise_or(lax.bitwise_not(
            record.exists(carry)), lax.lt(t_, t))  # either rc is earlier or carry doesn't exis
        return jnp.where(lax.bitwise_and(c, record.exists(rc)), rc, carry), None

    first, *_ = lax.scan(f, record.empty(), rcs)
    # print(first)
    return first
