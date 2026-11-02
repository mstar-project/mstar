"""The serving config's ``speculate_mid_prefill``: which chunk Loops let a row
mid-prefill take its next chunk in the speculative step."""

import pytest

from mstar.worker.chunk_loops import ChunkLoops, mid_prefill_speculation_off

CHUNKED = {("LLM", "prefill"): None, ("LLM", "vision"): None, ("Talker", "prefill"): None}
LOOPS = frozenset({("LLM", "prefill"), ("LLM", "vision")})  # this worker's


def _speculated(setting):
    return ChunkLoops(LOOPS, LOOPS - mid_prefill_speculation_off(setting, CHUNKED)).speculated


def test_every_chunk_loop_here_speculates_by_default():
    assert ChunkLoops(LOOPS).speculated == LOOPS
    assert _speculated(None) == LOOPS


def test_a_node_or_one_of_its_walks_can_be_turned_off():
    assert _speculated({"LLM": False}) == frozenset()
    assert _speculated({"LLM": {"vision": False}}) == {("LLM", "prefill")}


def test_a_node_chunked_on_another_worker_is_accepted():
    assert _speculated({"Talker": False}) == LOOPS


@pytest.mark.parametrize("setting", [{"Vision": False}, {"LLM": {"decode": False}}, {"LLM": "no"}])
def test_a_name_the_model_does_not_chunk_or_a_non_bool_is_refused(setting):
    with pytest.raises(ValueError, match="speculate_mid_prefill"):
        mid_prefill_speculation_off(setting, CHUNKED)
