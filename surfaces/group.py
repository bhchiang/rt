import jax.numpy as jnp


def create(surfaces):
    return jnp.array(surfaces)


def hit(surfaces, r, t_min, t_max):
    # trace all hits (no hit = empty record)

    # lax scan for earliest hit time
    pass
