import sys
import argparse
import time

from IPython import embed
import jax.numpy as jnp
from jax import vmap, lax, jit, random

import ray
import vec
import camera
import pixels
from surfaces import sphere, record, group
from common import IMAGE_HEIGHT, IMAGE_WIDTH, SAMPLES_PER_PIXEL, MAX_DEPTH


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


def pack(idx, key, r, rc):
    return jnp.vstack(([float(idx), *key], r, rc))


def unpack(v):
    (idx, *key), r, *rc = v
    return idx, jnp.array(key), r, jnp.array(rc)


def color(u, v, key):
    r = camera.shoot(u, v)
    _, dir = r

    # def n(rc):
    #     _, _, _, n = record.unpack(rc)
    #     return 0.5*(n + 1)

    # iterative: propagate ray until no surface hit or max depth reached
    iv = pack(0, key, r, record.empty())
    print(iv)

    # condition
    def cf(v):
        # terminate loop if false
        idx, _, _, rc = unpack(v)
        return lax.bitwise_or(
            lax.bitwise_and(
                lax.gt(idx, 0), lax.bitwise_not(record.exists(rc))),  # missed
            lax.ge(idx, MAX_DEPTH)  # max depth exceeded
        )

    # body
    def bf(v):
        idx, key, r, rc = unpack(v)
        rc = group.hit(r, 0., jnp.inf, g)

        # generate new key
        def tf(rc):
            # generate next ray
            t, p, ff, n = record.unpack(rc)
            target = p + n + vec.sphere()
            n_r = ray.create(p, target - p)
            return pack(idx+1, key, n_r, rc)

        def ff(rc):
            return pack(idx+1, key, r, record.empty())  # no hit

        return lax.cond(record.exists(rc), tf, ff, rc)

    def bg(r):
        # get bg color given y component of ray
        _, dir = ray.unpack(r)
        u_d = vec.unit(dir)
        t = 0.5 * (vec.y(u_d) + 1)  # -1 < y < 1 -> 0 < t < 1
        return (1-t) * vec.create(1, 1, 1) + t*vec.create(0.5, 0.7, 1.0)

    # return (1) black if depth exceeded or (2) modulated background
    v = lax.while_loop(cf, bf, iv)
    print(v)

    return vec.empty()


def trace(d):
    # (i, j) = (0, 0) is lower left corner
    # (IMAGE_WIDTH, IMAGE_HEIGHT) is upper right
    i, j, *k = d
    key = jnp.array(k)

    # create perturbations for anti-aliasing
    ps = random.uniform(key, (SAMPLES_PER_PIXEL, 2))

    def sample(d):
        pu, pv = d
        u = (i + pu) / (IMAGE_WIDTH - 1)
        v = (j + pv) / (IMAGE_HEIGHT - 1)
        return color(u, v, key)

    colors = vmap(sample)(ps)
    c = jnp.sum(colors, axis=0) / SAMPLES_PER_PIXEL
    # embed()

    return (255.99 * c).astype(jnp.int32)


parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="", action="store_true")
args = parser.parse_args()

key = random.PRNGKey(0)
keys = random.split(key, IMAGE_HEIGHT *
                    IMAGE_WIDTH).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 2)  # uint32
final_idxs = jnp.dstack(
    (idxs.astype(jnp.uint32), keys))  # prevent key overflow

if args.debug:
    embed()
    sys.exit()

img = vmap(vmap(trace))(final_idxs)
pl = pixels.flatten(img)
pixels.write(pl)
