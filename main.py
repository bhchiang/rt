import sys

from IPython import embed
import jax.numpy as jnp
from jax import vmap, lax, jit

from ray import Ray
from vec3 import vec, unit, x, y, z
from common import IMAGE_HEIGHT, IMAGE_WIDTH, VIEWPORT_HEIGHT, VIEWPORT_WIDTH, FOCAL_LENGTH
from utils import create_pixel_list, write_pixel_list, eprint

# camera
origin = vec()
horizontal = vec(VIEWPORT_WIDTH, 0, 0)
vertical = vec(0, VIEWPORT_HEIGHT, 0)
lower_left_corner = origin - (horizontal/2) - \
    (vertical/2) - vec(0, 0, FOCAL_LENGTH)

# indices
xs = jnp.arange(IMAGE_WIDTH)
ys = jnp.arange(IMAGE_HEIGHT)
us, vs = jnp.meshgrid(xs, ys)
idxs = jnp.dstack((us, jnp.flip(vs)))

# print(us)
# print(jnp.flip(vs))
# print(idxs[IMAGE_HEIGHT, 0])
# print(idxs[0, IMAGE_WIDTH])


def hit_sphere(center, radius, r):
    orig, dir = r
    a = dir.dot(dir)
    b = 2 * (orig.dot(dir) - dir.dot(center))
    c = (orig - center).dot(orig - center) - radius**2
    d = b**2 - 4*a*c
    t = (-b - jnp.sqrt(d)) / 2*a
    # eprint(d, t)
    return jnp.where(d > 0, t, -1)


def color(r):
    center = vec(0, 0, -1)
    # earliest intersection time
    h_t = hit_sphere(center, 0.5, r)
    # eprint(r.at(h_t))
    n = unit(r.at(h_t) - center)
    n_c = 0.5*(n + 1)
    # eprint(n, jnp.linalg.norm(n), n_c, 255.99*n_c)

    unit_dir = unit(r.dir)
    t = 0.5 * (y(unit_dir) + 1)  # -1 < y < 1 -> 0 < t < 1
    bg = (1-t) * vec(1, 1, 1) + t*vec(0.5, 0.7, 1.0)

    return jnp.where(h_t > 0, n_c, bg)


def trace(d):
    # (i, j) = (0, 0) is lower left corner
    # (IMAGE_WIDTH, IMAGE_HEIGHT) is upper right
    i, j = jnp.array(d, jnp.float32)

    u = i / (IMAGE_WIDTH-1)
    v = j / (IMAGE_HEIGHT-1)

    begin = origin
    end = lower_left_corner + u*horizontal + v*vertical

    r = Ray(origin, end - begin)
    c = color(r)

    # scale
    return (255.99 * c).astype(jnp.int32)


# embed()
# print(idxs.shape)

img = vmap(vmap(trace))(idxs)
pxls = create_pixel_list(img)
eprint(pxls[:10], pxls.shape)


write_pixel_list(pxls)
