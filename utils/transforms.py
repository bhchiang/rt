from jax import vmap, tree_multimap
import jax.numpy as jnp


def pytrees_stack(pytrees, axis=0):
    results = tree_multimap(
        lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results


def pytrees_vmap(fn):
    def g(pytrees):
        stacked = pytrees_stack(pytrees)
        results = vmap(fn)(stacked)
        return results
    return g
