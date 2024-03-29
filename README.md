# rt-jax

[JAX](https://github.com/google/jax) implementation of a simple path tracer, based on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Items

- [x] Use data classes (registered as pytrees) and namedtuples instead of manual packing
- [ ] Benchmark full JIT performance

## Usage

```
# trace
python main.py > out/image.ppm

# view
eog out/image.ppm
```

## Lessons

Do the following inside `vmap` because of the tracing performed for vectorization:

- Avoid classes for storing data, use `jnp.ndarray` or pytree containers
- Avoid conditional statements like `if / else`, use `lax.cond` (if further computation necessarey) or `jnp.where`
- Avoid `p_a and p_b` or `p_a or p_b`, use `jnp.array([p_a, p_b]).any() / .all()` or `lax.bitwise_(and/or/not)`
- Given `jnp.ndarray` `a` and `b`, use `jnp.dot(a, b)` instead of `a.dot(b)`
- Be careful about passing `jax.random` keys around. They're of type `jnp.uint32`, casting to `jnp.int32` will lose the precision and can truncate to 0
- Avoid using list / `[]`, use `jnp.array(...)` instead

More JAX notes and experiments here: https://github.com/bryanhpchiang/jax-notes.
