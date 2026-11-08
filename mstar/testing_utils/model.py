"""A ``Model`` for tests, with no config, weights or tokenizer.

A stand-in that is not a ``Model`` has to spell out every method the code under
test reads, so each new call the engine makes on a model means editing every such
stand-in. These inherit ``Model``'s defaults instead, and a new abstract method
needs one edit, in ``StubModel``.

Use ``make_model(get_node_resources=lambda: [kv])`` for a few fixed answers, and
subclass ``StubModel`` for a stand-in with state or helpers of its own.
"""
from mstar.model.base import Model


class StubModel(Model):
    """Declares no resources or walks, and its nodes run in dummy mode. A request
    has no empty answer for how it starts or moves on, so those methods raise."""

    def get_node_resources(self):
        return []

    def get_graph_walk_graphs(self):
        return {}

    def get_submodule(self, *args, **kwargs):
        return None

    def get_initial_forward_pass_args(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not define get_initial_forward_pass_args")

    def get_partition_forward_pass_args(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not define get_partition_forward_pass_args")

    def process_prompt(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not define process_prompt")


def _bound(value):
    """``value`` as a class attribute: a callable takes the call's arguments without ``self``."""
    if not callable(value):
        return value

    def method(self, *args, **kwargs):
        del self
        return value(*args, **kwargs)
    return method


def make_model(**overrides) -> StubModel:
    """A ``StubModel`` with ``overrides`` replacing its methods and attributes by name.

    A name ``Model`` does not have raises ``TypeError``, so a test cannot go on
    overriding a method that has been renamed. The overrides go on a new class rather
    than the instance, since ``names_checkpoint`` and the chunk wrapping of
    ``get_graph_walk_graphs`` read the class, as they do for a real model.
    """
    for name in overrides:
        if not hasattr(StubModel, name):
            raise TypeError(f"Model has no attribute {name!r}")
    return type("StubModel", (StubModel,), {name: _bound(value) for name, value in overrides.items()})()
