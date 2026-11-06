"""The state contract only grows once it is released, and this snapshot is how a pull request shows that.

Every model written against the contract relies on its names, fields and defaults, so renaming a field or changing a
default breaks models outside this repo with no error of their own. Any change to the surface fails here. A new name, or
a new keyword field whose default keeps every declaration's meaning, adds lines to the snapshot and passes. Any other
change has to edit or remove a line of the snapshot to pass, which the pull request's diff shows.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping

sys.path.insert(0, ".")

from mstar.engine import state

# Each public name's parameters as ``[name, kind, default]``. A dataclass's parameters are its fields.
SNAPSHOT = {
    "Array": [
        ["name", "POSITIONAL_OR_KEYWORD", None],
        ["nodes", "KEYWORD_ONLY", None],
        ["layers", "KEYWORD_ONLY", None],
        ["entry", "KEYWORD_ONLY", None],
        ["max_len", "KEYWORD_ONLY", None],
        ["reads", "KEYWORD_ONLY", '{}'],
        ["updates", "KEYWORD_ONLY", '{}'],
        ["rewritable", "KEYWORD_ONLY", 'False'],
    ],
    "Attend": [
        ["q_heads", "POSITIONAL_OR_KEYWORD", None],
    ],
    "Causal": [],
    "CausalConv": [
        ["activation", "KEYWORD_ONLY", "'silu'"],
    ],
    "Entry": [
        ["tensors", "POSITIONAL_OR_KEYWORD", None],
        ["host", "KEYWORD_ONLY", '{}'],
    ],
    "GatedDeltaNet": [
        ["qk_l2norm", "KEYWORD_ONLY", 'True'],
        ["scale", "KEYWORD_ONLY", 'None'],
    ],
    "KVEntry": [
        ["heads", "KEYWORD_ONLY", None],
        ["head_dim", "KEYWORD_ONLY", None],
    ],
    "Read": [
        ["mask", "KEYWORD_ONLY", None],
        ["op", "KEYWORD_ONLY", 'None'],
        ["layers", "KEYWORD_ONLY", 'None'],
        ["walks", "KEYWORD_ONLY", 'None'],
    ],
    "TensorType": [
        ["shape", "POSITIONAL_OR_KEYWORD", None],
        ["dtype", "KEYWORD_ONLY", None],
        ["shard", "KEYWORD_ONLY", 'None'],
    ],
    "Value": [
        ["name", "POSITIONAL_OR_KEYWORD", None],
        ["nodes", "KEYWORD_ONLY", None],
        ["layers", "KEYWORD_ONLY", None],
        ["entry", "KEYWORD_ONLY", None],
        ["updates", "KEYWORD_ONLY", '{}'],
    ],
    "Window": [
        ["size", "POSITIONAL_OR_KEYWORD", None],
    ],
}


def _default(parameter: inspect.Parameter) -> str | None:
    if parameter.default is inspect.Parameter.empty:
        return None
    # the type of a mapping is engine-private, so renaming it isn't a change to the surface
    return repr(dict(parameter.default) if isinstance(parameter.default, Mapping) else parameter.default)


def _surface() -> dict[str, list[list[str | None]]]:
    return {
        name: [
            [p.name, p.kind.name, _default(p)]
            for p in inspect.signature(getattr(state, name)).parameters.values()
        ]
        for name in sorted(state.__all__)
    }


def test_all_is_exactly_the_frozen_surface():
    assert sorted(state.__all__) == [
        "Array", "Attend", "Causal", "CausalConv", "Entry", "GatedDeltaNet", "KVEntry", "Read", "TensorType", "Value",
        "Window",
    ]


def test_the_surface_equals_the_snapshot():
    assert _surface() == SNAPSHOT, "an addition updates SNAPSHOT, and a change in meaning comes as a new name"


def test_every_field_after_the_first_is_keyword_only():
    positional = {name: [p[0] for p in params[1:] if p[1] != "KEYWORD_ONLY"] for name, params in _surface().items()}
    assert not any(positional.values()), positional
