
import IPython
from jax import vmap
from jax import tree_util
from core import Camera, Ray


import jax.numpy as jnp


def f(a, c):
    return c.lower_left_corner


aa = jnp.array([[1, 2], [3, 4]])
cc = Ray(origin=jnp.zeros(3), direction=jnp.zeros(3))
cc = Camera(16/9)
result = vmap(vmap(f, in_axes=(0, None)), in_axes=(0, None))(aa, cc)
print(result)
IPython.embed()
