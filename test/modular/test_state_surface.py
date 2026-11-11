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

import pytest

from mstar.engine import state
from mstar.engine.state.adapters import PagedArrayHandle, ValueHandle, WithinStepHandle
from mstar.engine.state.calls import ArrayHandle

# Today's types the surface re-exports, and a NamedTuple, whose fields stay positional.
_POSITIONAL = {"Reached", "Segment", "SlotLease", "Span"}
# The calls take the handle first, then what every call names, and only then keywords.
_CALLS = {
    "at_last_token", "causal_conv", "gated_delta_net", "last_sampled", "position", "positions",
    "rope", "sample",
}
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
        ["labels", "KEYWORD_ONLY", "frozenset({'main'})"],
        ["positions", "KEYWORD_ONLY", 'None'],
    ],
    "ArrayStep": [
        ["segments", "KEYWORD_ONLY", 'None'],
        ["keep", "KEYWORD_ONLY", 'True'],
        ["forks", "KEYWORD_ONLY", '()'],
        ["placeholders", "KEYWORD_ONLY", 'frozenset()'],
        ["eager_add", "KEYWORD_ONLY", 'False'],
        ["advance", "KEYWORD_ONLY", 'None'],
        ["position_ids", "KEYWORD_ONLY", 'None'],
    ],
    "Attend": [
        ["q_heads", "POSITIONAL_OR_KEYWORD", None],
    ],
    "Block": [],
    "Causal": [],
    "CausalConv": [
        ["activation", "KEYWORD_ONLY", "'silu'"],
    ],
    "Entry": [
        ["tensors", "POSITIONAL_OR_KEYWORD", None],
        ["host", "KEYWORD_ONLY", '{}'],
    ],
    "Fork": [
        ["src", "KEYWORD_ONLY", None],
        ["dst", "KEYWORD_ONLY", None],
        ["at", "KEYWORD_ONLY", None],
        ["rows", "KEYWORD_ONLY", 'None'],
    ],
    "GatedDeltaNet": [
        ["qk_l2norm", "KEYWORD_ONLY", 'True'],
        ["scale", "KEYWORD_ONLY", 'None'],
    ],
    "KVEntry": [
        ["heads", "KEYWORD_ONLY", None],
        ["head_dim", "KEYWORD_ONLY", None],
    ],
    "Layout": [
        ["spans", "KEYWORD_ONLY", '()'],
        ["growth", "KEYWORD_ONLY", '()'],
        ["scratch", "KEYWORD_ONLY", '0'],
    ],
    "Positions": [
        ["rope", "KEYWORD_ONLY", 'None'],
    ],
    "Reached": [
        ["fields", "POSITIONAL_OR_KEYWORD", None],
        ["row_offsets", "POSITIONAL_OR_KEYWORD", None],
    ],
    "Read": [
        ["mask", "KEYWORD_ONLY", None],
        ["op", "KEYWORD_ONLY", 'None'],
        ["layers", "KEYWORD_ONLY", 'None'],
        ["walks", "KEYWORD_ONLY", 'None'],
    ],
    "Rope": [
        ["theta", "KEYWORD_ONLY", '10000.0'],
        ["rotary_dim", "KEYWORD_ONLY", 'None'],
        ["interleave", "KEYWORD_ONLY", 'False'],
        ["scale", "KEYWORD_ONLY", '1.0'],
        ["low_freq_factor", "KEYWORD_ONLY", 'None'],
        ["high_freq_factor", "KEYWORD_ONLY", 'None'],
        ["old_context_len", "KEYWORD_ONLY", 'None'],
        ["dtype", "KEYWORD_ONLY", 'None'],
        ["table_dtype", "KEYWORD_ONLY", 'None'],
    ],
    "SampleSettings": [
        ["temperature", "KEYWORD_ONLY", '0.6'],
        ["top_k", "KEYWORD_ONLY", '0'],
        ["top_p", "KEYWORD_ONLY", '1.0'],
        ["min_p", "KEYWORD_ONLY", '0.0'],
        ["repetition_penalty", "KEYWORD_ONLY", '1.0'],
        ["repetition_window", "KEYWORD_ONLY", '0'],
        ["penalize_prompt", "KEYWORD_ONLY", 'True'],
        ["min_tokens", "KEYWORD_ONLY", '0'],
        ["ignore_eos", "KEYWORD_ONLY", 'False'],
        ["top_p_first", "KEYWORD_ONLY", 'False'],
        ["top_p_min_keep", "KEYWORD_ONLY", '1'],
        ["return_logprobs", "KEYWORD_ONLY", 'False'],
        ["logprobs_mode", "KEYWORD_ONLY", "'raw'"],
    ],
    "SampleStep": [
        ["apply_penalty", "KEYWORD_ONLY", 'True'],
        ["apply_filters", "KEYWORD_ONLY", 'True'],
        ["prompt_tokens", "KEYWORD_ONLY", '{}'],
        ["kept_rows", "KEYWORD_ONLY", 'None'],
    ],
    "Sampler": [
        ["name", "POSITIONAL_OR_KEYWORD", None],
        ["nodes", "KEYWORD_ONLY", None],
        ["vocab_size", "KEYWORD_ONLY", None],
        ["penalty", "KEYWORD_ONLY", 'True'],
        ["min_p", "KEYWORD_ONLY", 'False'],
        ["eos_ids", "KEYWORD_ONLY", '()'],
        ["draws", "KEYWORD_ONLY", '1'],
    ],
    "Segment": [
        ["request_id", "POSITIONAL_OR_KEYWORD", None],
        ["label", "POSITIONAL_OR_KEYWORD", None],
        ["span", "POSITIONAL_OR_KEYWORD", None],
    ],
    "SlotLease": [
        ["slot", "POSITIONAL_OR_KEYWORD", None],
        ["bucket", "POSITIONAL_OR_KEYWORD", None],
    ],
    "Span": [
        ["kind", "POSITIONAL_OR_KEYWORD", None],
        ["walk", "POSITIONAL_OR_KEYWORD", None],
        ["length", "POSITIONAL_OR_KEYWORD", None],
        ["advance", "POSITIONAL_OR_KEYWORD", None],
        ["source", "POSITIONAL_OR_KEYWORD", None],
        ["params", "POSITIONAL_OR_KEYWORD", '()'],
        ["scratch", "POSITIONAL_OR_KEYWORD", '0'],
        ["opens_step", "POSITIONAL_OR_KEYWORD", 'True'],
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
    "at_last_token": [
        ["h", "POSITIONAL_OR_KEYWORD", None],
        ["x", "POSITIONAL_OR_KEYWORD", None],
    ],
    "causal_conv": [
        ["h", "POSITIONAL_OR_KEYWORD", None],
        ["layer", "POSITIONAL_OR_KEYWORD", None],
        ["x", "POSITIONAL_OR_KEYWORD", None],
        ["weight", "KEYWORD_ONLY", None],
        ["bias", "KEYWORD_ONLY", 'None'],
    ],
    "gated_delta_net": [
        ["h", "POSITIONAL_OR_KEYWORD", None],
        ["layer", "POSITIONAL_OR_KEYWORD", None],
        ["q", "POSITIONAL_OR_KEYWORD", None],
        ["k", "POSITIONAL_OR_KEYWORD", None],
        ["v", "POSITIONAL_OR_KEYWORD", None],
        ["a", "POSITIONAL_OR_KEYWORD", None],
        ["b", "POSITIONAL_OR_KEYWORD", None],
        ["a_log", "POSITIONAL_OR_KEYWORD", None],
        ["dt_bias", "POSITIONAL_OR_KEYWORD", None],
    ],
    "last_sampled": [
        ["s", "POSITIONAL_OR_KEYWORD", None],
    ],
    "position": [
        ["h", "POSITIONAL_OR_KEYWORD", None],
        ["rid_handle", "POSITIONAL_OR_KEYWORD", None],
        ["label", "POSITIONAL_OR_KEYWORD", None],
    ],
    "positions": [
        ["h", "POSITIONAL_OR_KEYWORD", None],
    ],
    "rope": [
        ["h", "POSITIONAL_OR_KEYWORD", None],
        ["q", "POSITIONAL_OR_KEYWORD", None],
        ["k", "POSITIONAL_OR_KEYWORD", None],
    ],
    "sample": [
        ["s", "POSITIONAL_OR_KEYWORD", None],
        ["logits", "POSITIONAL_OR_KEYWORD", None],
    ],
}


HANDLE_SNAPSHOT = {
    "add": [
        ["self", "POSITIONAL_OR_KEYWORD", None],
        ["layer", "POSITIONAL_OR_KEYWORD", None],
        ["keep", "KEYWORD_ONLY", "None"],
        ["fields", "VAR_KEYWORD", None],
    ],
    "read": [
        ["self", "POSITIONAL_OR_KEYWORD", None],
        ["layer", "POSITIONAL_OR_KEYWORD", None],
        ["q", "POSITIONAL_OR_KEYWORD", "None"],
        ["mask_args", "VAR_KEYWORD", None],
    ],
    "write": [
        ["self", "POSITIONAL_OR_KEYWORD", None],
        ["layer", "POSITIONAL_OR_KEYWORD", None],
        ["at", "KEYWORD_ONLY", "None"],
        ["fields", "VAR_KEYWORD", None],
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
    assert sorted(state.__all__) == sorted(SNAPSHOT)


def test_the_surface_equals_the_snapshot():
    assert _surface() == SNAPSHOT, "an addition updates SNAPSHOT, and a change in meaning comes as a new name"


@pytest.mark.parametrize("handle", [ArrayHandle, PagedArrayHandle, ValueHandle, WithinStepHandle])
def test_a_handles_calls_equal_the_snapshot(handle):
    surface = {
        name: [[p.name, p.kind.name, _default(p)] for p in inspect.signature(getattr(handle, name)).parameters.values()]
        for name in HANDLE_SNAPSHOT
    }
    assert surface == HANDLE_SNAPSHOT, "`add`, `read` and `write` are the surface a layer calls"


def test_every_field_after_the_first_is_keyword_only():
    positional = {
        name: [p[0] for p in params[1:] if p[1] != "KEYWORD_ONLY"]
        for name, params in _surface().items() if name not in _POSITIONAL | _CALLS
    }
    assert not any(positional.values()), positional
