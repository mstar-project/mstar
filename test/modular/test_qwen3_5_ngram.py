"""Qwen3.5 behind the ``draft`` switch (``model_kwargs: {draft: {method: ngram, ...}}``).

Off, nothing changes. On, the node declares the draft-verify resource and the n-gram
proposer, its sampler refuses what the verify can't keep exact, the mixed walk is gone (a
verify row needs a step of its own), the KV layout carries the verify rows' lookahead, the
decode loop emits the engine's cut under ``new_token``, a decode step verifies (the GDN step
included) and a prefill hands the proposer its prompt. The real model's verify runs on a GPU
(``test/integration/test_qwen3_5_ngram_gpu.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_qwen3_5_video import write_synthetic_checkpoint  # noqa: E402

from mstar.engine.resources import (  # noqa: E402
    NGRAM,
    DraftConfig,
    DraftVerifySpec,
    ProposeStep,
    VerifyKey,
    VerifyStep,
)
from mstar.engine.resources.draft_verify.config import DRAFT_VERIFY  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.qwen3_5.config import ATTN, KV_CACHE, LINEAR_ATTN, LLM_MIXED, SAMPLER  # noqa: E402
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel  # noqa: E402
from mstar.model.qwen3_5.submodules import LLMSubmodule  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs  # noqa: E402

DRAFT = {"method": "ngram", "max_drafts": 3, "max_ngram": 5}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    return write_synthetic_checkpoint(tmp_path_factory.mktemp("qwen3_5_ngram"))


def _client_edges(model) -> list[str]:
    loop = model.get_graph_walk_graphs()["decode"]
    return [edge.name for edge in loop.section.outputs if edge.next_node == EMIT_TO_CLIENT]


def test_off_changes_nothing(ckpt):
    model = Qwen3_5DenseModel(str(ckpt))
    assert model.draft is None
    assert model.get_combined_graph_walks() == {"LLM": {LLM_MIXED: {"prefill_text", "decode"}}}
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    assert DRAFT_VERIFY not in specs and NGRAM not in specs
    assert not specs[SAMPLER].draft_verify
    assert specs[ATTN].config.splits and specs[LINEAR_ATTN].config.splits
    assert _client_edges(model) == ["text_inputs"]
    layout = model.process_prompt("hello", ["text"], ["text"]).metadata["kv_layout"][KV_CACHE]["main"]
    assert layout.lookahead == 0


def test_on_declares_the_draft_verify_node(ckpt):
    model = Qwen3_5DenseModel(str(ckpt), draft=DRAFT)
    assert model.draft == DraftConfig(**DRAFT)
    assert model.get_combined_graph_walks() == {}
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    assert isinstance(specs[DRAFT_VERIFY], DraftVerifySpec) and specs[DRAFT_VERIFY].sampler == SAMPLER
    assert specs[NGRAM].max_drafts == 3 and specs[NGRAM].max_ngram == 5
    assert specs[SAMPLER].draft_verify and specs[SAMPLER].enable_sampled_verify
    assert specs[ATTN].config.splits == () and specs[LINEAR_ATTN].config.splits == ()
    assert _client_edges(model) == ["new_token"]
    layout = model.process_prompt("hello", ["text"], ["text"]).metadata["kv_layout"][KV_CACHE]["main"]
    assert layout.lookahead == 2 * 3 + 1
    greedy = Qwen3_5DenseModel(str(ckpt), draft={**DRAFT, "sampled_verify": False})
    sampler = {spec.resource_key: spec for spec in greedy.get_node_resources()}[SAMPLER]
    assert sampler.draft_verify and not sampler.enable_sampled_verify


def test_an_unknown_setting_is_refused(ckpt):
    with pytest.raises(ValueError, match="unknown draft settings"):
        Qwen3_5DenseModel(str(ckpt), draft={"method": "ngram", "k": 4})


def _submodule() -> LLMSubmodule:
    """Only what `declare_step` and `prepare_inputs` read."""
    llm = object.__new__(LLMSubmodule)
    draft = DraftConfig(**DRAFT)
    llm.__dict__.update(_draft=draft, _draft_policy=draft.policy(stop_ids={7}), reads_device_loopback=False)
    return llm


def test_decode_verifies_and_prefill_hands_over_the_prompt():
    llm = _submodule()
    assert llm.get_draft_policy("decode").proposer == NGRAM
    assert llm.get_draft_policy("prefill_text") is None

    rows = [llm.prepare_inputs("decode", None, {"text_inputs": [torch.tensor([[t]])]}) for t in (11, 12)]
    assert [r.input_seq_len for r in rows] == [4, 4] and rows[1].input_ids.tolist() == [12]
    step = llm.declare_step("decode", [0, 1], rows)
    assert isinstance(step.steps[DRAFT_VERIFY], VerifyStep) and step.steps[DRAFT_VERIFY].drafts == 3
    assert isinstance(step.steps[NGRAM], ProposeStep) and step.steps[NGRAM].drafts == 3
    assert step.steps[LINEAR_ATTN].verify
    assert step.steps[SAMPLER].verify
    assert [seg.span for seg in step.segments] == [4, 4]

    prompt = torch.tensor([5, 6, 7, 5, 6])
    inp = ARNodeInputs(input_seq_len=5, input_ids=prompt)
    step = llm.declare_step("prefill_text", [3], [inp], per_request_info={})
    history = step.steps[NGRAM]
    assert isinstance(history, ProposeStep) and history.drafts == 0 and history.history[3] is prompt
    assert not step.steps[LINEAR_ATTN].verify


def test_an_adaptive_decode_step_runs_a_block_or_one_token():
    """Prompt lookup decides each decode step (``adaptive``, its default): the engine sets a
    row's span to the step's block, and a step of one token takes the plain GDN path, keyed to
    its own capture."""
    llm = _submodule()
    assert llm.get_draft_policy("decode").adaptive
    rows = [ARNodeInputs(input_ids=torch.tensor([t]), input_seq_len=1) for t in (11, 12)]
    step = llm.declare_step("decode", [0, 1], rows)
    assert step.steps[DRAFT_VERIFY].drafts == 0 and step.steps[NGRAM].drafts == 0
    assert not step.steps[LINEAR_ATTN].verify and step.cg_key_info == VerifyKey(0)
    rows = [ARNodeInputs(input_ids=torch.tensor([t]), input_seq_len=4) for t in (11, 12)]
    assert llm.declare_step("decode", [0, 1], rows).cg_key_info == VerifyKey(3)
    configs = [c for c in llm.get_accelerator_graph_configs(torch.device("cpu")) if c.capture_graph_walk == "decode"]
    assert [(c.single_request_inputs.input_seq_len, c.additional_key_info) for c in configs] == [
        (1, VerifyKey(0)), (4, VerifyKey(3)),
    ]
