import sys
import argparse
import time
from collections import namedtuple

from IPython import embed
import jax.numpy as jnp
from jax import vmap, lax, jit, random

from core import Ray, vec, camera, pixels
from surfaces import sphere, record, group
from core.common import IMAGE_HEIGHT, IMAGE_WIDTH, SAMPLES_PER_PIXEL, MAX_DEPTH


# indices
xs = jnp.arange(IMAGE_WIDTH)
ys = jnp.arange(IMAGE_HEIGHT)
us, vs = jnp.meshgrid(xs, ys)
idxs = jnp.dstack((us, jnp.flip(vs)))

# print(us)
# print(jnp.flip(vs))
# print(idxs[IMAGE_HEIGHT, 0])
# print(idxs[0, IMAGE_WIDTH])

surfaces = [
    sphere.create(vec.create(0, 0, -1), 0.5),
    sphere.create(vec.create(0, -100.5, -1), 100)
]
g = group.create(surfaces)


def color(u, v, key):
    r = camera.shoot(u, v)

    # propagate ray iteratively until no surface hit or max depth reached
    Value = namedtuple("Value", ["idx", "key", "ray", "record"])
    init_val = Value(idx=0, key=key, ray=r, record=record.empty())

    def cf(val):
        # continue if first iter OR surface hit and max depth not reached
        return lax.bitwise_or(
            lax.eq(val.idx, 0), lax.bitwise_and(record.exists(val.record), lax.lt(val.idx, MAX_DEPTH)))

    def bf(val):

        # avoid hitting surface we are reflecting off of (shadow acne)
        rc = group.hit(val.ray, 0.001, jnp.inf, g)

        key, subkey = random.split(val.key)
        base_val = Value(idx=val.idx+1, key=key, ray=val.ray, record=rc)

        def tf(rc):
            # generate next ray
            _, p, _, n = record.unpack(rc)
            target = p + n + vec.sphere(subkey)
            n_r = Ray(origin=p, direction=target-p)
            return base_val._replace(ray=n_r)

        def ff(rc):
            return base_val  # no hit

        return lax.cond(record.exists(rc), tf, ff, rc)

    def bg(r):
        # get bg color given y component of ray
        u_d = vec.unit(r.direction)
        t = 0.5 * (vec.y(u_d) + 1)  # -1 < y < 1 -> 0 < t < 1
        return (1-t) * vec.create(1, 1, 1) + t*vec.create(0.5, 0.7, 1.0)

    # return bg if missed, else black (max depth exceeded)
    final_val = lax.while_loop(cf, bf, init_val)
    color = jnp.power(0.5, final_val.idx - 1) * \
        jnp.where(lax.bitwise_not(record.exists(final_val.record)),
                  bg(final_val.ray), vec.create())
    # embed()
    return color


def trace(d, key):
    # (i, j) = (0, 0) is lower left
    # (IMAGE_WIDTH, IMAGE_HEIGHT) is upper right
    i, j = d

    # create perturbations for anti-aliasing
    ps = random.uniform(key, (SAMPLES_PER_PIXEL, 2))
    keys = random.split(key, (SAMPLES_PER_PIXEL))

    def sample(d, sample_key):
        pu, pv = d
        u = (i + pu) / (IMAGE_WIDTH - 1)
        v = (j + pv) / (IMAGE_HEIGHT - 1)
        return color(u, v, sample_key)

    # embed()
    colors = vmap(sample)(ps, keys)
    c = jnp.sum(colors, axis=0) / SAMPLES_PER_PIXEL
    return jnp.int32(255.99 * jnp.sqrt(c))  # gamma correction


@jit
def render(idxs, rng):
    return vmap(vmap(trace))(idxs, rng)


parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="", action="store_true")
args = parser.parse_args()

key = random.PRNGKey(0)
keys = random.split(key, IMAGE_HEIGHT *
                    IMAGE_WIDTH).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 2)

if args.debug:
    # i = idxs_rng[IMAGE_HEIGHT * 3//4, IMAGE_WIDTH // 2]
    i = 194
    j = 115
    trace(idxs[i, j], keys[i, j])
    sys.exit()

img = render(idxs, keys)
pl = pixels.flatten(img)
pixels.write(pl)
