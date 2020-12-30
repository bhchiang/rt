# rt-jax

[JAX](https://github.com/google/jax) implementation of a simple path tracer, based on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Usage

## Notes

Inside `vmap`:

- Conditional statements like `if` are not permitted due to the tracing, use `lax.cond` or `jnp.where`
- Doing `p_a and p_b` or `p_a or p_b` will throw a tracing error, use `jnp.array([p_a, p_b]).any() / .all()` or `lax.bitwise_and(), bitwise_not(), bitwise_or()`
- Given `jnp.ndarray` `a` and `b`, use `jnp.dot(a, b)` instead of `a.dot(b)`
- Avoid classes for storing data, everything flowing through must be `jnp.ndarray`

In general:

- When debugging stack traces, start from your bottom and go up a few frames to find the offending line in your program. JAX spews out a lot of garbage that you can ignore.
