import dataclasses
from jax import tree_util


def register_jax_dataclass(cls):
    """Registers a dataclass as a JAX pytree."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError('%s is not a dataclass.' % cls)

    keys = [field.name for field in dataclasses.fields(cls) if field.init]

    def _flatten(obj):
        return [getattr(obj, key) for key in keys], None

    def _unflatten(_, children):
        return cls(**dict(zip(keys, children)))

    tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls


def jax_dataclass(cls):
    """Decorator function to define a dataclass with JAX bindings."""
    return register_jax_dataclass(dataclasses.dataclass(cls))  # frozen=True disables __post_init__
