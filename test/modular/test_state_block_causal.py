"""`BlockCausal` is how a kept array is read block by block, as CosyVoice3's flow attends its cache: a query in block b
sees every entry of blocks 0 to b, so what the pages hold never changes and a step runs only its new entries.

It runs on the paged block-causal kernel, which takes packed rows and plans whole blocks. A step that ends inside a
block would cut what the block's queries see, so a walk that regroups steps refuses the read. The kernel takes no split
steps either, which a node on SP ranks hears at validation instead of when the engine builds.
"""

from __future__ import annotations

import pytest
from test_state_windows import _array, _Model, _read

from mstar.engine.resources.attn.config import AttentionConfig, AttentionStep, BlockCausalAttentionSpec
from mstar.engine.resources.sequence_shard import SeqShardSpec, sequence_parallel_refusals
from mstar.engine.resources.step import Segment
from mstar.engine.state import ArrayStep, BlockCausal
from mstar.engine.state.adapters import _bindings, build_specs, storage_of
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import KVLabel, PrefixKeys

BLOCK = 50


def _blocks(**fields):
    return _array(reads={"attn": _read(BlockCausal(BLOCK), **fields)})


def test_block_causal_builds_main_s_spec_and_step():
    array = _blocks()
    assert BlockCausalAttentionSpec(
        resource_key="attn", nodes={"dit"}, config=AttentionConfig(kv_cache="kv"), block_size=BLOCK,
    ) in build_specs([array]), "the read should build the block-causal spec a model writes by hand"
    (bound,) = _bindings(storage_of(array), array)
    segments, steps = (Segment(1, "main", BLOCK),), {}
    bound.binding.steps(array, "attn", bound.declared, ArrayStep(), segments, steps)
    assert steps == {"attn": AttentionStep(segments=segments, causal=False)}, (
        "the kernel takes its mask from the spec, so the step plans no causal mask on top"
    )


@pytest.mark.parametrize(("fields", "message"), [
    ({"packed": False}, r"reads\['attn'\] declares packed=False, and the block-causal kernel takes packed rows"),
    ({"walks": {"prefill"}}, r"reads\['attn'\].walks must be None"),
], ids=["unpacked", "walks"])
def test_a_block_causal_read_takes_packed_rows_on_every_walk(fields, message):
    with pytest.raises(ValueError, match=f"Array 'kv': no layout yet: {message}"):
        build_specs([_blocks(**fields)])


@pytest.mark.parametrize(("kwargs", "cause"), [
    ({"chunked": ("prefill",)}, "which a chunking policy splits across steps"),
    ({"kv_labels": {"kv": {"main": KVLabel(keys=PrefixKeys("ids", ("prefill",)))}}}, "keys for prefix reuse"),
    ({"rebuilt": True}, "which dit's rebuild policy runs it again"),
], ids=["chunked", "keyed", "rebuilt"])
def test_a_block_causal_read_refuses_a_step_that_splits_a_block(kwargs, cause):
    (problem,) = state_declaration_problems(_Model(_blocks(), **kwargs))
    assert problem.startswith("Array 'kv'.reads['attn']: BlockCausal on walk 'prefill'") and cause in problem
    assert "A step boundary inside a block cuts what the block's queries see, so " in problem, (
        "the refusal should give a block-causal read's own cause"
    )


def test_a_node_that_splits_its_steps_refuses_a_block_causal_read():
    specs = [*build_specs([_blocks()]), SeqShardSpec("shard", {"dit"}, ("main",), 0)]
    assert sequence_parallel_refusals(specs, "dit", 1, 2) == [
        "block-causal attention 'attn' takes no split steps, since its builder sets no sequence shard, so node 'dit' "
        "can't split them: run it on TP ranks alone"
    ], "a node on SP ranks should hear the kernel's refusal at validation, not when the engine builds"
