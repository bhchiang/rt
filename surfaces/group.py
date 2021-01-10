from functools import partial

import jax.numpy as jnp
from jax import lax, vmap

from . import Record
from . import Sphere
from utils import jax_dataclass, pytrees_vmap


@jax_dataclass
class Group:
    surfaces: list

    def hit(self, r, t_min, t_max):
        # trace all hits (no hit = empty record)
        rcs = pytrees_vmap(lambda sp: sp.hit(r, t_min, t_max))(self.surfaces)

        # lax scan for earliest hit time
        def f(earliest, current):
            replace = lax.bitwise_or(lax.bitwise_not(
                earliest.exists), lax.lt(current.t, earliest.t))  # either rc is earlier or carry doesn't exist
            # jnp.where requires broadcastable arrays to work
            return lax.cond(lax.bitwise_and(replace, current.exists), lambda _: current, lambda _: earliest, 0), 0

        first, *_ = lax.scan(f, Record(exists=False), rcs)
        return first
