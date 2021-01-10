import unittest
from collections import namedtuple

import jax.numpy as jnp
from jax import tree_multimap, vmap

from utils import pytrees_stack, pytrees_vmap
from core import Camera
from surfaces import Sphere


class TestTransforms(unittest.TestCase):
    # utils.transforms

    def __init__(self, *args, **kwargs):
        super(TestTransforms, self).__init__(*args, **kwargs)
        Value = namedtuple("Value", ["a", "b", "c"])
        self.values = [
            Value(2, jnp.array([1, 5, 3.]), jnp.array([[1, 0], [8, 3]])),
            Value(129123, jnp.array([1, 2., 212.]),
                  jnp.array([[9, 2], [2, 1]]))
        ]

        self.surfaces = [
            Sphere(center=jnp.zeros(3), radius=5),
            Sphere(center=jnp.array([0, 1, 0]), radius=1)
        ]

        self.cameras = [
            Camera(16/9),
            Camera(4/3)
        ]

    def test_pytree_stack(self):
        # values_stacked = pytrees_stack(self.values)
        cameras_stacked = pytrees_stack(self.cameras)
        # print(cameras_stacked)

    def test_pytree_vmap(self):
        def f1(sp: Sphere) -> jnp.ndarray:
            return sp.center + sp.radius

        results = pytrees_vmap(f1)(self.surfaces)
        # print(results)

        def f2(camera: Camera) -> jnp.ndarray:
            return jnp.where(camera.lower_left_corner[0] < -1.5, 1, 2)

        results = pytrees_vmap(f2)(self.cameras)
        print(results)


if __name__ == '__main__':
    unittest.main()
