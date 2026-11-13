"""Overrides by name for the stubs in this package."""


def _bound(value):
    """``value`` as a class attribute: a callable takes the call's arguments without ``self``."""
    if not callable(value):
        return value

    def method(self, *args, **kwargs):
        del self
        return value(*args, **kwargs)
    return method


def stub_with(base: type, overrides: dict):
    """A ``base`` with ``overrides`` replacing its methods and attributes by name.

    A name ``base`` does not have raises ``TypeError``, so a test cannot go on
    overriding one that has been renamed. The overrides go on a new class rather than
    the instance, since production code compares methods on the class, as in
    ``names_checkpoint`` or ``StepRunner``'s choice of which resources publish.
    """
    for name in overrides:
        if not hasattr(base, name):
            raise TypeError(f"{base.__name__} has no attribute {name!r}")
    return type(base.__name__, (base,), {name: _bound(value) for name, value in overrides.items()})()
