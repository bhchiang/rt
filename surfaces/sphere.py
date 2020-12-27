import jax.numpy as jnp


def hit(center, radius, r):
    orig, dir = r
    a = dir.dot(dir)
    h = (orig.dot(dir) - dir.dot(center))  # b = 2h
    c = (orig - center).dot(orig - center) - radius**2
    d = h*h - a*c
    t = (-h - jnp.sqrt(d)) / a
    return jnp.where(d > 0, t, -1)
