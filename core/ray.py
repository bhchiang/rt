import dataclasses
import jax.numpy as jnp

from utils import jax_dataclass
from . import vec


@jax_dataclass
class Ray:
    origin: jnp.ndarray
    direction: jnp.ndarray

    def at(self, t):
        return self.origin + t*self.direction


if __name__ == "__main__":
    origin = vec.create(0, 0, 0)
    direction = vec.create(1, 0, 0)

    r = Ray(origin=origin, direction=direction)
    assert vec.equal(r.at(2), (2, 0, 0))
