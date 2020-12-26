import jax.numpy as jnp
from jax import vmap

from ray import Ray
from vec3 import vec, unit, x, y, z
from common import IMAGE_HEIGHT, IMAGE_WIDTH, VIEWPORT_HEIGHT, VIEWPORT_WIDTH, FOCAL_LENGTH
from utils import create_pixel_list, write_pixel_list

# camera
origin = vec()
horizontal = vec(VIEWPORT_WIDTH, 0, 0)
vertical = vec(0, VIEWPORT_HEIGHT, 0)
lower_left_corner = origin - (horizontal/2) - \
    (vertical/2) - vec(0, 0, FOCAL_LENGTH)

# indices
X, Y = jnp.meshgrid(jnp.arange(IMAGE_WIDTH), jnp.arange(IMAGE_HEIGHT))
a = jnp.array(list(zip(X.ravel(), Y.ravel()))).reshape(
    IMAGE_WIDTH, IMAGE_HEIGHT,  2)
# print(a.shape)


def color(r):
    unit_dir = unit(r.dir)
    t = 0.5 * (y(unit_dir) + 1)  # -1 < y < 1 -> 0 < t < 1
    c = (1-t) * vec(1, 1, 1) + t*vec(0.5, 0.7, 1.0)
    print(t, c)
    return c


def trace(d):
    i, jj = jnp.array(d, jnp.float32)  # i = width, j = height
    j = IMAGE_HEIGHT - jj - 1  # (0, 0) at lower left corner

    u = i / (IMAGE_WIDTH-1)
    v = j / (IMAGE_HEIGHT-1)

    begin = origin
    end = lower_left_corner + u*horizontal + v*vertical

    r = Ray(origin, end - begin)
    c = color(r)

    # scale
    return (255.99 * c).astype(jnp.int32)


img = vmap(vmap(trace))(a)
# print(img)
pxls = create_pixel_list(img)
# print(pxls[:10], pxls.shape)


write_pixel_list(pxls)
