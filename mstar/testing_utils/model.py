"""A ``Model`` for tests, with no config, weights or tokenizer.

A stand-in that is not a ``Model`` has to spell out every method the code under
test reads, so each new call the engine makes on a model means editing every such
stand-in. These inherit ``Model``'s defaults instead, and a new abstract method
needs one edit, in ``StubModel``.

Use ``make_model(get_node_resources=lambda: [kv])`` for a few fixed answers, and
subclass ``StubModel`` for a stand-in with state or helpers of its own.
"""
from mstar.model.base import Model
from mstar.testing_utils._overrides import stub_with


class StubModel(Model):
    """Declares no resources or walks, and its nodes run in dummy mode. A request
    has no neutral plan or prompt, so those methods raise."""

    def get_node_resources(self):
        return []

    def get_graph_walk_graphs(self):
        return {}

    def get_submodule(self, *args, **kwargs):
        return None

    def walk_plan(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not define walk_plan")

    def process_prompt(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not define process_prompt")


def make_model(**overrides) -> StubModel:
    """A ``StubModel`` with ``overrides`` replacing its methods and attributes by name;
    a name ``Model`` does not have raises ``TypeError``. A real class holds them, so the
    chunk wrapping of ``get_graph_walk_graphs`` applies as it does for a real model."""
    return stub_with(StubModel, overrides)
