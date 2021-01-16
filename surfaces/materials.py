import abc
from overrides import overrides

import jax.numpy as jnp
from core import vec, Ray

from utils import jax_dataclass


@jax_dataclass
class MaterialBase(abc.ABC):

    @abc.abstractmethod
    def scatter(self, r, record, attenuation, key):
        pass


@jax_dataclass
class Lambertian(MaterialBase):

    @overrides
    def scatter(self, r, record, attenuation, key):
        scatter_direction = record.normal + vec.sphere(key)
        scattered = Ray(origin=record.p, direction=scatter_direction)
        return (scattered)
