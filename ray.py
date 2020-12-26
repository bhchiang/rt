import jax.numpy as jnp
from vec3 import vec, assert_vec_eq


class Ray():
    def __init__(self, origin=None, direction=None):
        self.orig = origin
        self.dir = direction

    def at(self, t):
        return self.orig + t*self.dir


if __name__ == "__main__":
    # run tests
    origin = vec(0, 0, 0)
    direction = vec(1, 0, 0)

    r = Ray(origin, direction)
    assert_vec_eq(r.at(2), vec(2, 0, 0))

    print("passed tests")
