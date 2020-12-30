import sys
import argparse

from IPython import embed
import jax.numpy as jnp
from jax import vmap, lax, jit

import ray
import vec
import pixels
from surfaces import sphere, record, group
from common import IMAGE_HEIGHT, IMAGE_WIDTH, VIEWPORT_HEIGHT, VIEWPORT_WIDTH, FOCAL_LENGTH

# camera
origin = vec.create()
horizontal = vec.create(VIEWPORT_WIDTH, 0, 0)
vertical = vec.create(0, VIEWPORT_HEIGHT, 0)
lower_left_corner = origin - (horizontal/2) - \
    (vertical/2) - vec.create(0, 0, FOCAL_LENGTH)

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


def color(r):
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
    i, j = jnp.array(d, jnp.float32)

    u = i / (IMAGE_WIDTH-1)
    v = j / (IMAGE_HEIGHT-1)

    begin = origin
    end = lower_left_corner + u*horizontal + v*vertical

    r = ray.create(origin, end - begin)
    c = color(r)

    # scale
    return (255.99 * c).astype(jnp.int32)


parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="", action="store_true")
args = parser.parse_args()

if args.debug:
    embed()
    sys.exit()

# print(idxs.shape)
img = vmap(vmap(trace))(idxs)
pl = pixels.flatten(img)
# print(pl[:10], pl.shape)

pixels.write(pl)
