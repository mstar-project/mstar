"""A ``Resource`` for tests.

A stand-in that is not a ``Resource`` has to spell out every attribute the engine
reads, so each new read means editing every such stand-in. These inherit ``Resource``'s
defaults instead, and a new abstract method needs one edit, in ``StubResource``.

Use ``make_resource(can_capture=lambda: False)`` for a few fixed answers, and subclass
``StubResource`` for a stand-in with state of its own.
"""
from mstar.engine.resources.base import Resource
from mstar.testing_utils._overrides import stub_with


class StubResource(Resource):
    """Holds nothing and admits everything, by ``Resource``'s defaults."""

    @classmethod
    def build(cls, spec, info):
        return cls()


def make_resource(**overrides) -> StubResource:
    """A ``StubResource`` with ``overrides`` replacing its methods and attributes by name;
    a name ``Resource`` does not have raises ``TypeError``."""
    return stub_with(StubResource, overrides)
