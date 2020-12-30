import sys
import argparse

from IPython import embed
import jax.numpy as jnp
from jax import vmap, lax, jit, random

import ray
import vec
import camera
import pixels
from surfaces import sphere, record, group
from common import IMAGE_HEIGHT, IMAGE_WIDTH, SAMPLES_PER_PIXEL


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


def color(u, v):
    r = camera.shoot(u, v)
    _, dir = r

    rc = group.hit(r, 0., jnp.inf, g)
    # center, radius =
    # sp = sphere.create(center, radius)
    # rc = sphere.hit(sp, r,  0., jnp.inf)

    def n(rc):
        _, _, _, n = record.unpack(rc)
        return 0.5*(n + 1)

    def bg(_):
        u_d = vec.unit(dir)
        t = 0.5 * (vec.y(u_d) + 1)  # -1 < y < 1 -> 0 < t < 1
        return (1-t) * vec.create(1, 1, 1) + t*vec.create(0.5, 0.7, 1.0)

    return lax.cond(record.exists(rc), n, bg, rc)


def trace(d):
    # (i, j) = (0, 0) is lower left corner
    # (IMAGE_WIDTH, IMAGE_HEIGHT) is upper right
    i, j, *key = d

    # create perturbations for anti-aliasing
    ps = random.uniform(
        jnp.array(key), (SAMPLES_PER_PIXEL, 2))

    def sample(d):
        pu, pv = d
        u = (i + pu) / (IMAGE_WIDTH - 1)
        v = (j + pv) / (IMAGE_HEIGHT - 1)
        return color(u, v)

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
