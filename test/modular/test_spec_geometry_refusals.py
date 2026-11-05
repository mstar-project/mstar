"""Spec configs refuse geometry that fails at boot, before any worker starts.

A model copies ``config.json`` fields into these configs, and nothing checked what
they hold. A Qwen3.5 config with every layer linear attention passes
``validate_deployment`` with a KV cache of zero layers, then its worker dies at
build on a division by zero. These tests pin each refusal, with a message naming
the field and the value it got, and one shape that still loads.
"""

import pytest

from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStateConfig


def _paged(**overrides) -> PagedKVConfig:
    return PagedKVConfig(**{"num_layers": 1, "num_kv_heads": 4, "head_dim": 8, "max_seq_len": 64, **overrides})


def _kernel_1_blocks() -> dict:
    """A delta net of kernel 1 keeps no conv tail: ``to_blocks`` asks for width 0."""
    return DeltaNetGeometry(
        num_k_heads=2, num_v_heads=4, head_k_dim=8, head_v_dim=8, conv_kernel_size=1,
    ).to_blocks()


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


@pytest.mark.parametrize("bad", [262144.0, 0, -1, True, "64"])
def test_a_max_seq_len_that_is_not_a_positive_int_is_refused(bad):
    with pytest.raises(ValueError, match=f"max_seq_len must be an integer >= 1; got {bad!r}"):
        _paged(max_seq_len=bad)


@pytest.mark.parametrize("bad", [32768.0, 0, -1, True, "64"])
def test_a_yaml_max_seq_len_that_is_not_a_positive_int_is_refused(bad):
    """``resources:`` sets the field after construction."""
    with pytest.raises(ValueError, match=f"max_seq_len must be an integer >= 1; got {bad!r}"):
        _paged().apply_yaml_overrides(max_seq_len=bad)


def test_a_block_with_an_empty_axis_is_refused():
    with pytest.raises(ValueError, match=r"blocks\['conv'\]\.shape=\(64, 0\)"):
        RecurrentStateConfig(num_layers=1, blocks=_kernel_1_blocks())


def test_a_pool_with_no_layers_takes_a_block_with_an_empty_axis():
    """Qwen3.5 with every layer full attention and a kernel of 1 declares this pool."""
    config = RecurrentStateConfig(num_layers=0, blocks=_kernel_1_blocks())
    assert config.blocks["conv"].shape == (64, 0), "no layer reads this pool, so nothing fails on the empty axis"
