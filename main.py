import sys

from IPython import embed
import jax.numpy as jnp
from jax import vmap, lax, jit

import ray
import vec
import pixels
from surfaces import sphere

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


def color(r):
    orig, dir = r
    center = vec.create(0, 0, -1)
    radius = 0.5
    # earliest intersection time
    h_t = sphere.hit(center, radius, r)
    # print(r.at(h_t))
    n = vec.unit(ray.at(r, h_t) - center)
    n_c = 0.5*(n + 1)
    # print(n, jnp.linalg.norm(n), n_c, 255.99*n_c)

    unit_dir = vec.unit(dir)
    t = 0.5 * (vec.y(unit_dir) + 1)  # -1 < y < 1 -> 0 < t < 1
    bg = (1-t) * vec.create(1, 1, 1) + t*vec.create(0.5, 0.7, 1.0)

    return jnp.where(h_t > 0, n_c, bg)


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


# embed()
# print(idxs.shape)

img = vmap(vmap(trace))(idxs)
pl = pixels.flatten(img)
# print(pl[:10], pl.shape)


pixels.write(pl)
