import jax.numpy as jnp
from core import vec
from utils import jax_dataclass


@jax_dataclass
class Record:
    t: jnp.float32 = 0.
    p: jnp.ndarray = jnp.zeros(3)
    normal: jnp.ndarray = jnp.zeros(3)
    front_face: bool = False
    exists: bool = True
