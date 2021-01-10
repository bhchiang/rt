import jax.numpy as jnp
from core import vec
from utils import jax_dataclass


@jax_dataclass
class Record:
    t: jnp.float32
    p: jnp.ndarray
    normal: jnp.ndarray
    front_face: bool
    exists: bool = True

    # TODO: come up with better way to do this
    @classmethod
    def empty(cls):
        obj = object.__new__(cls)
        obj.t = jnp.float32(0)
        obj.p = vec.create()
        obj.normal = vec.create()
        obj.front_face = jnp.bool_(False)
        obj.exists = jnp.bool_(False)
        return obj
