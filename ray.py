import jax.numpy as jnp


class Ray():
    def __init__(self, origin=None, direction=None):
        self.orig = origin
        self.dir = direction

    def at(self, t):
        return self.orig + t*self.dir


if __name__ == "__main__":
    # run tests
    r = Ray()
