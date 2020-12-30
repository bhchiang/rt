import jax.numpy as jnp
import vec


def create(origin, direction):
    return jnp.array([origin, direction])


def at(r, t):
    orig, dir = r
    return orig + t*dir


if __name__ == "__main__":
    origin = vec.create(0, 0, 0)
    direction = vec.create(1, 0, 0)

    r = create(origin, direction)
    assert vec.equal(at(r, 2), (2, 0, 0))
