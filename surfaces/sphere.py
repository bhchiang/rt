import jax.numpy as jnp
from jax import lax


from . import record
import vec
import ray


def create(center, radius):
    return jnp.array([center, vec.pad(radius)])


def unpack(sp):
    center, (radius, *_) = sp
    return center, radius


def hit(sp, r, t_min, t_max):
    # returns empty record if there are no valid hits

    center, radius = unpack(sp)  # sphere.unpack
    # print(center, radius)
    orig, dir = r
    # print(orig, dir)

    a = dir.dot(dir)
    h = (orig.dot(dir) - dir.dot(center))  # b = 2h
    c = (orig - center).dot(orig - center) - radius**2
    d = h*h - a*c

    empty = record.empty()
    # print(h, d, a)

    def solve(op):
        (h, d, a), dir = op

        def valid(t):
            return jnp.array([lax.ge(t, t_min), lax.le(t, t_max)]).all()

        t_1 = (-h - jnp.sqrt(d)) / a  # first hit
        t_2 = (-h + jnp.sqrt(d)) / a  # second hit
        t_2 = jnp.where(valid(t_2), t_2, -jnp.inf)
        t = jnp.where(valid(t_1), t_1, t_2)

        # print(f't = {t}')

        def create(dir):
            p = ray.at(r, t)  # get point of intersection
            o_n = vec.unit(p - center)  # outward facing normal
            # print(f'o_n = {o_n}, dir = {dir}')
            ff = lax.lt(dir.dot(o_n), 0.)  # front face = ray, o_n in opp dirs

            n = jnp.where(ff, o_n, -o_n)  # points against ray
            rec = record.create(t, p, ff, n)
            return rec

        return lax.cond(t != -jnp.inf, create, lambda _: empty, dir)

    return lax.cond(d > 0, solve, lambda _: record.empty(), jnp.array([[h, d, a], dir]))
