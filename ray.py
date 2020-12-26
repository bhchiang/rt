import jax.numpy as jnp
from vec3 import vec, assert_vec_eq


def create(origin, direction):
    return jnp.array([origin, direction])


def at(r, t):
    orig, dir = r
    return orig + t*dir


if __name__ == "__main__":
    # run tests
    origin = vec(0, 0, 0)
    direction = vec(1, 0, 0)

    r = create(origin, direction)
    assert_vec_eq(at(r, 2), vec(2, 0, 0))

    print("passed tests")
