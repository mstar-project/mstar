"""``PromptsJsonDataset``: the row shapes a prompts JSON file may take."""

from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, ".")

from benchmark.dataset import PromptsJsonDataset  # noqa: E402


def _load(tmp_path, payload, num_requests=None):
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps(payload))
    rows = payload["prompts"] if isinstance(payload, dict) else payload
    n = num_requests if num_requests is not None else len(rows)
    return PromptsJsonDataset(str(f), num_requests=n)


def test_dict_rows_carry_text_and_per_prompt_budget(tmp_path):
    ds = _load(tmp_path, {"_comment": "ignored", "prompts": [
        {"id": 1, "text": "hello", "max_tokens": 7},
        {"id": 2, "text": "world"},
    ]})
    assert [it.prompt for it in ds.items] == ["hello", "world"]
    assert ds.items[0].model_kwargs == {"max_tokens": 7, "max_output_tokens": 7}
    assert ds.items[1].model_kwargs == {}


def test_bare_string_rows_are_prompts(tmp_path):
    # used to die with AttributeError: 'str' object has no attribute 'get'
    ds = _load(tmp_path, ["prompt one", "prompt two"])
    assert [it.prompt for it in ds.items] == ["prompt one", "prompt two"]
    assert all(it.model_kwargs == {} for it in ds.items)


def test_string_and_dict_rows_mix(tmp_path):
    # num_requests=2: _resize_data pads/cycles to the requested count
    ds = _load(tmp_path, ["bare", {"text": "object", "max_tokens": 3}, ""], num_requests=2)
    assert [it.prompt for it in ds.items] == ["bare", "object"]  # empty text skipped
    assert ds.items[1].model_kwargs["max_output_tokens"] == 3


@pytest.mark.parametrize("bad", [[1, 2], [["nested"]], [None]])
def test_other_row_types_name_the_expected_shape(tmp_path, bad):
    shape = r'row 0 is .*; expected a \{"id", "text", "max_tokens"\} object or a prompt string'
    with pytest.raises(ValueError, match=shape):
        _load(tmp_path, bad)


def test_non_list_payload_is_refused(tmp_path):
    with pytest.raises(ValueError, match="expected a JSON list"):
        _load(tmp_path, {"prompts": "not a list"}, num_requests=1)
