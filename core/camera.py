
from dataclasses import field

from jax import tree_util
import jax.numpy as jnp

from . import vec, Ray
from utils import jax_dataclass


@jax_dataclass
class Camera:
    aspect_ratio: jnp.float32
    viewport_height: jnp.float32 = 2.0
    viewport_width: jnp.float32 = None
    focal_length: jnp.float32 = 1
    origin: jnp.ndarray = vec.create()
    horizontal: jnp.ndarray = None
    vertical: jnp.ndarray = None
    lower_left_corner: jnp.ndarray = None

    def __post_init__(self):
        # (1) ignore dummy init for vmap - https://github.com/google/jax/blob/master/jax/api_util.py#L176
        # (2) ignore if constructing from stacked pytrees resulting from tree_multimap
        if type(self.aspect_ratio) == object or not isinstance(self.aspect_ratio, float):
            return
        self.viewport_width = self.aspect_ratio * self.viewport_height
        self.horizontal = vec.create(self.viewport_width, 0, 0)
        self.vertical = vec.create(0, self.viewport_height, 0)
        self.lower_left_corner = self.origin - \
            (self.horizontal/2) - (self.vertical/2) - \
            vec.create(0, 0, self.focal_length)

    def shoot(self, u, v):
        end = self.lower_left_corner + u*self.horizontal + v*self.vertical
        return Ray(origin=self.origin, direction=end - self.origin)
