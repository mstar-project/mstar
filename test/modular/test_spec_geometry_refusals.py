"""Spec configs refuse geometry that fails at boot, before any worker starts.

A model copies ``config.json`` fields into these configs, a serving YAML overrides
some of them, and nothing checked what they hold. A Qwen3.5 config with every layer
linear attention passes ``validate_deployment`` with a KV cache of zero layers, then
its worker dies at build on a division by zero. These tests pin each refusal, with a
message naming the field and the value it got, and the shapes that still load.
"""

import pytest
import torch

from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)

# field, a value out of range, and the field's smallest legal value
_PAGED_SIZES = [
    *(
        (field, bad, 1)
        for field in ("max_seq_len", "max_num_pages", "page_size")
        for bad in (2048.0, 0, -1, True, "64")
    ),
    *(("cpu_offload_pages", bad, 0) for bad in (2048.0, -1, True, "64")),
]
_POOL_SIZES = [
    *(("max_slots", bad, 1) for bad in (256.0, 0, -1, True, "8")),
    *(("prefix_slots", bad, 0) for bad in (128.0, -1, True, "8")),
]


def _paged(**overrides) -> PagedKVConfig:
    return PagedKVConfig(**{"num_layers": 1, "num_kv_heads": 4, "head_dim": 8, "max_seq_len": 64, **overrides})


def _delta_net_blocks(kernel: int) -> dict:
    """``to_blocks`` asks for a conv tail of ``kernel - 1``: width 0 at kernel 1, -1 at kernel 0."""
    return DeltaNetGeometry(
        num_k_heads=2, num_v_heads=4, head_k_dim=8, head_v_dim=8, conv_kernel_size=kernel,
    ).to_blocks()


def _pool(**overrides) -> RecurrentStateConfig:
    return RecurrentStateConfig(**{"num_layers": 1, "blocks": _delta_net_blocks(4), **overrides})


def test_a_kv_cache_with_no_layers_is_refused():
    """Qwen3.5 with every layer linear attention."""
    with pytest.raises(ValueError, match="num_layers=0"):
        _paged(num_layers=0)


@pytest.mark.parametrize("qo, kv", [(16, 3), (4, 8)])
def test_query_heads_that_are_not_a_multiple_of_kv_heads_are_refused(qo, kv):
    with pytest.raises(ValueError, match=f"num_qo_heads={qo} must be a positive multiple of num_kv_heads={kv}"):
        _paged(num_qo_heads=qo, num_kv_heads=kv)


@pytest.mark.parametrize("bad", [0, -4])
def test_a_kv_head_count_below_1_is_refused(bad):
    with pytest.raises(ValueError, match=f"num_kv_heads={bad}"):
        _paged(num_qo_heads=16, num_kv_heads=bad)


@pytest.mark.parametrize("bad", [0, -4])
def test_a_query_head_count_below_1_is_refused(bad):
    """Both pass the multiple check on their own: 0 % 4 and -4 % 4 are 0."""
    with pytest.raises(ValueError, match=f"num_qo_heads={bad} must be a positive multiple of num_kv_heads=4"):
        _paged(num_qo_heads=bad, num_kv_heads=4)


@pytest.mark.parametrize("field, bad, smallest", _PAGED_SIZES)
def test_a_paged_size_out_of_range_is_refused(field, bad, smallest):
    with pytest.raises(ValueError, match=f"{field} must be an integer >= {smallest}; got {bad!r}"):
        _paged(**{field: bad})


@pytest.mark.parametrize("field, bad, smallest", _PAGED_SIZES)
def test_a_yaml_paged_size_out_of_range_is_refused(field, bad, smallest):
    """``resources:`` sets the field after construction."""
    with pytest.raises(ValueError, match=f"{field} must be an integer >= {smallest}; got {bad!r}"):
        _paged().apply_yaml_overrides(**{field: bad})


def test_a_derived_page_and_no_host_pool_stay_legal():
    config = _paged(page_size=None, cpu_offload_pages=0)
    config.apply_yaml_overrides(cpu_offload_pages=0)
    assert (config.page_size, config.cpu_offload_pages) == (None, 0), "None derives the page and 0 turns the pool off"


@pytest.mark.parametrize("field, bad, smallest", _POOL_SIZES)
def test_a_pool_size_out_of_range_is_refused(field, bad, smallest):
    with pytest.raises(ValueError, match=f"{field} must be an integer >= {smallest}; got {bad!r}"):
        _pool(**{field: bad})


@pytest.mark.parametrize("field, bad, smallest", _POOL_SIZES)
def test_a_yaml_pool_size_out_of_range_is_refused(field, bad, smallest):
    spec = RecurrentStateSpec(resource_key="pool", nodes={"llm"}, config=_pool())
    with pytest.raises(ValueError, match=f"{field} must be an integer >= {smallest}; got {bad!r}"):
        spec.apply_yaml_overrides(**{field: bad})


def test_a_pool_that_keeps_no_snapshots_stays_legal():
    assert _pool(prefix_slots=0).prefix_slots == 0, "0 turns the prefix snapshots off"


def test_a_block_with_an_empty_axis_is_refused():
    with pytest.raises(ValueError, match=r"blocks\['conv'\]\.shape=\(64, 0\)"):
        _pool(blocks=_delta_net_blocks(1))


def test_a_negative_axis_is_refused_in_a_pool_with_no_layers():
    """``torch.zeros`` refuses it whatever the layer count."""
    with pytest.raises(ValueError, match=r"blocks\['conv'\]\.shape=\(64, -1\)"):
        _pool(num_layers=0, blocks=_delta_net_blocks(0))


def test_an_empty_axis_on_a_per_slot_block_is_refused_in_a_pool_with_no_layers():
    """A block that is not per layer is allocated per slot whatever the layer count."""
    blocks = {"count": RecurrentBlockConfig(shape=(0,), dtype=torch.int32, per_layer=False)}
    with pytest.raises(ValueError, match=r"blocks\['count'\]\.shape=\(0,\)"):
        _pool(num_layers=0, blocks=blocks)


def test_a_pool_with_no_layers_takes_a_per_layer_block_with_an_empty_axis():
    """Qwen3.5 with every layer full attention and a kernel of 1 declares this pool."""
    config = _pool(num_layers=0, blocks=_delta_net_blocks(1))
    assert config.blocks["conv"].shape == (64, 0), "no layer reads this block, so nothing fails on the empty axis"


@pytest.mark.parametrize("limit", [32, None, 0, True, "32"])
def test_deployment_context_default_and_resource_override(limit):
    from mstar.engine.resources.kv.config import KVSpec
    from mstar.engine.resources.spec import apply_yaml_overrides

    first = KVSpec(resource_key="first", nodes={"node"}, config=_paged())
    second = KVSpec(resource_key="second", nodes={"node"}, config=_paged())
    config = {"max_seq_len": limit, "resources": {"second": {"max_seq_len": 128}}}
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        with pytest.raises(ValueError, match="max_seq_len"):
            apply_yaml_overrides([first, second], config)
    else:
        apply_yaml_overrides([first, second], config)
        assert first.config.max_seq_len == (64 if limit is None else limit)
        assert second.config.max_seq_len == 128
