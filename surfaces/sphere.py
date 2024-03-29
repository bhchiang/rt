import jax.numpy as jnp
from jax import lax


from utils import jax_dataclass
from . import Record
from core import vec


@jax_dataclass
class Sphere:
    center: jnp.ndarray
    radius: jnp.float32

    def hit(self, r, t_min, t_max):
        # returns empty record if there are no valid hits
        center, radius = self.center, self.radius
        orig, dir = r.origin, r.direction

        a = jnp.dot(dir, dir)
        h = jnp.dot(orig, dir) - jnp.dot(dir, center)  # b = 2h
        oc = orig - center
        c = jnp.dot(oc, oc) - radius*radius
        d = h*h - a*c

        def empty(v):
            return Record(exists=False)
        # print(h, d, a)

        def solve(v):
            h, d, a = v

            def valid(t):
                return jnp.array([lax.ge(t, t_min), lax.le(t, t_max)]).all()

            t_1 = (-h - jnp.sqrt(d)) / a  # first hit
            t_2 = (-h + jnp.sqrt(d)) / a  # second hit
            t_2 = jnp.where(valid(t_2), t_2, -jnp.inf)
            t = jnp.where(valid(t_1), t_1, t_2)

            # print(f't = {t}')

            def create(dir):
                p = r.at(t)  # get point of intersection
                o_n = (p - center) / radius  # outward facing normal
                # print(f'o_n = {o_n}, dir = {dir}')
                # front face = ray, o_n in opp dirs
                ff = lax.lt(jnp.dot(dir, o_n), 0.)

                n = jnp.where(ff, o_n, -o_n)  # points against ray
                rec = Record(t=t, p=p, normal=n, front_face=ff)
                return rec

            return lax.cond(t != -jnp.inf, create, empty, dir)

        return lax.cond(d > 0, solve, empty, jnp.array([h, d, a]))
