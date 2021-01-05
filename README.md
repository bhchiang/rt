# rt-jax

[JAX](https://github.com/google/jax) implementation of a simple path tracer, based on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Next Steps

- [ ] Use pytree-compatible data classes instead of manually packing

## Usage

```
python main.py > out/image.ppm

python main.py --debug
```

## Notes

Do the following inside `vmap` because of the tracing performed for vectorization:

- Avoid classes for storing data, use `jnp.ndarray`
- Avoid conditional statements like `if / else`, use `lax.cond` (if further computation necessarey) or `jnp.where`
- Avoid `p_a and p_b` or `p_a or p_b`, use `jnp.array([p_a, p_b]).any() / .all()` or `lax.bitwise_(and/or/not)`
- Given `jnp.ndarray` `a` and `b`, use `jnp.dot(a, b)` instead of `a.dot(b)`
