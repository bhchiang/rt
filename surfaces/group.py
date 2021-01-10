from functools import partial

import jax.numpy as jnp
from jax import lax, vmap

from . import record
from . import Sphere
from utils import jax_dataclass, pytrees_vmap


@jax_dataclass
class Group:
    surfaces: list

    def hit(self, r, t_min, t_max):
        # trace all hits (no hit = empty record)
        rcs = pytrees_vmap(lambda sp: sp.hit(r, t_min, t_max))(self.surfaces)

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
