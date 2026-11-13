"""Stand-ins for state handles, for tests that drive a submodule's host bookkeeping without an engine."""

from collections.abc import Mapping


class StubHostFields:
    """A Value of host fields as the engine keeps them for a submodule's `host` calls: each starts at ``defaults``,
    and what a step's ``ArrayStep(host=)`` writes under ``name`` applies when ``commit`` is called with the step, as
    the engine applies it at the step's commit."""

    def __init__(self, name: str, defaults: Mapping[str, object]):
        self.name = name
        self.defaults = dict(defaults)
        self.fields: dict[tuple[int, str, int], dict[str, object]] = {}

    @staticmethod
    def _layer_id(layer: int) -> int:
        return int(layer)

    def _host(self, request: int, label: str, layer: int) -> Mapping[str, object]:
        return {**self.defaults, **self.fields.get((request, label, layer), {})}

    def commit(self, step) -> None:
        declared = step.state.get(self.name)
        if declared is not None:
            for key, written in declared.host.items():
                self.fields.setdefault(key, {}).update(written)
