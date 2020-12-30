# rt-jax

[JAX](https://github.com/google/jax) implementation of a simple path tracer, based on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Usage

```
python main.py > out/image.ppm

python main.py --debug # dev
```

## Notes

Inside `vmap`, because of the tracing:

- Avoid classes for storing data, use `jnp.ndarray`
- Avoid conditional statements like `if / else`, use `lax.cond` (if further computation necessarey) or `jnp.where`
- Avoid `p_a and p_b` or `p_a or p_b`, use `jnp.array([p_a, p_b]).any() / .all()` or `lax.bitwise_(and/or/not)`
- Given `jnp.ndarray` `a` and `b`, use `jnp.dot(a, b)` instead of `a.dot(b)`

In general:

- When debugging stack traces, start from your bottom and go up a few frames to find the offending line in your program.
