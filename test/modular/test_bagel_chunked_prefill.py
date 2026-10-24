"""BAGEL text prefill runs in chunks: a chunk keeps its row's guidance flag,
a guided prompt forks main -> cfg_text on its first chunk only (cfg_text keeps
the pre-text context), and only the last chunk's sampled token is kept."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mstar.model.bagel.submodules import LLMSubmodule
from mstar.model.submodule_base import ARNodeInputs


@pytest.fixture
def llm():
    llm = object.__new__(LLMSubmodule)
    llm.__dict__["node_name"] = "LLM"
    return llm


def _prompt(n=10, guided=True):
    return ARNodeInputs(input_seq_len=n, input_ids=torch.arange(n), resource_step_info=guided)


def test_a_chunk_keeps_its_rows_guidance_flag(llm):
    cut = llm.split_inputs("prefill_text", None, _prompt(), 4, 8)

    assert cut.input_ids.tolist() == [4, 5, 6, 7] and cut.resource_step_info is True


def test_a_guided_prompt_forks_on_its_first_chunk_only(llm):
    first = replace(llm.split_inputs("prefill_text", None, _prompt(), 0, 4), chunk_start=0, chunk_total=10)
    later = replace(llm.split_inputs("prefill_text", None, _prompt(), 4, 8), chunk_start=4, chunk_total=10)

    kv = llm.declare_step("prefill_text", ["a", "b"], [first, later]).steps["kv"]
    assert kv.pre_forks == (("main", "cfg_text"),) and kv.fork_rids == frozenset({"a"})

    kv = llm.declare_step("prefill_text", ["b"], [later]).steps["kv"]
    assert kv.pre_forks == ()


def test_a_non_final_chunk_drops_its_token(llm):
    info = SimpleNamespace(graph_walk="prefill_text", step_metadata={"sample_prefill_token": True})
    chunk = replace(_prompt(4), chunk_start=0, chunk_total=10)
    outputs = {"new_token": [torch.tensor([7])]}

    llm.postprocess("a", info, outputs, inputs=chunk)

    assert "new_token" not in outputs and "text_inputs" not in outputs
