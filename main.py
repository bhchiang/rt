import jax.numpy as jnp
from jax import vmap

from common import *
from utils import create_pixel_list, write_pixel_list

# create indices
X, Y = jnp.meshgrid(jnp.arange(IMAGE_WIDTH), jnp.arange(IMAGE_HEIGHT))
a = jnp.array(list(zip(X.ravel(), Y.ravel()))).reshape(
    IMAGE_WIDTH, IMAGE_HEIGHT,  2)
# print(a.shape)


def trace(d):
    i, j = jnp.array(d, float)  # i = width, j = height
    jj = IMAGE_HEIGHT - j - 1

    r = i / (IMAGE_WIDTH-1)
    g = jj / (IMAGE_HEIGHT-1)
    b = 0.25

    # scale
    return (255.99 * jnp.array([r, g, b])).astype(int)


img = vmap(vmap(trace))(a)
# print(img)
pxls = create_pixel_list(img)
# print(pxls[:10], pxls.shape)


write_pixel_list(pxls)
