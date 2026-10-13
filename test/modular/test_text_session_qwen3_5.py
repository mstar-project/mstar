"""The ``test_text_session_qwen3_5`` deployment: Qwen3.5's LLM with sessions on
both its KV cache and its GDN recurrent state.

CPU-only: needs Qwen3.5's config and tokenizer (not its weights), and skips when
they are not cached.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import yaml

from mstar.model.higgs_audio.config import SAMPLER
from mstar.model.qwen3_5.config import GDN_STATE, KV_CACHE
from mstar.model.registry import get_model_class, model_init_kwargs
from mstar.model.sessions import RequestSession, apply_sessions_yaml_overrides
from mstar.model.submodule_base import OVERSHOT_LAST_ITER, NodeSubmodule
from mstar.model.test_text_session.qwen3_5 import (
    TEXT_WALKS,
    TextSessionQwen3_5LLMSubmodule,
)

NAME = "test_text_session_qwen3_5"
CONFIG = Path("configs/test_text_session_qwen3_5.yaml")


@pytest.fixture(scope="module")
def model():
    try:
        return get_model_class(NAME)(cache_dir=None, **model_init_kwargs(NAME))
    except Exception as e:  # noqa: BLE001 — any download/auth failure is a skip
        pytest.skip(f"Qwen3.5's config/tokenizer is not available: {e!r}")


# ── what it serves ──────────────────────────────────────────────────────────

def test_it_defaults_to_4b_and_a_config_can_pick_another_size():
    assert model_init_kwargs(NAME)["model_path_hf"] == "Qwen/Qwen3.5-4B"


def test_only_the_text_walks_are_declared(model):
    assert sorted(model.get_graph_walk_graphs()) == sorted(TEXT_WALKS)
    assert model.nodes == ["LLM"]


def test_the_vision_tower_s_resources_are_dropped(model):
    specs = model.get_node_resources()

    assert all(spec.nodes == {"LLM"} for spec in specs)
    assert {KV_CACHE, GDN_STATE} <= {spec.resource_key for spec in specs}


def test_it_holds_both_the_kv_and_the_gdn_state(model):
    # holding only one is refused at load; see test_engine_session_state
    assert set(model.get_sessions_config().resources) == {KV_CACHE, GDN_STATE}


def test_the_shipped_config_merges_over_the_declaration(model):
    serving = yaml.safe_load(CONFIG.read_text())
    merged = apply_sessions_yaml_overrides(model.get_sessions_config(), serving)

    assert serving["model"] == NAME
    assert set(merged.resources) == {KV_CACHE, GDN_STATE}
    # every parked session's slot plus one per running request, and the sink
    assert serving["resources"][GDN_STATE]["max_slots"] > (
        merged.max_concurrent_sessions + serving["max_concurrent_requests"]
    )


# ── thinking is off, whatever the request asks ──────────────────────────────

def _rendered(model, **kwargs) -> str:
    out = model.process_prompt("Hi", ["text"], ["text"], **kwargs)
    return model.tokenizer.decode(torch.cat(out["text_inputs"]))


@pytest.mark.parametrize("asked", [None, True, False])
def test_every_turn_opens_with_a_closed_think_block(model, asked):
    kwargs = {} if asked is None else {"enable_thinking": asked}

    rendered = _rendered(model, **kwargs)

    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_a_resumed_turn_renders_like_an_opening_one(model):
    # no system block to leave out: one user turn is already one more turn
    opening = _rendered(model, session=RequestSession("s"))
    resumed = _rendered(model, session=RequestSession("s", resumed=True))

    assert opening == resumed
    assert opening.startswith("<|im_start|>user\n")


def test_anything_but_text_is_refused(model):
    with pytest.raises(ValueError, match="text only"):
        model.process_prompt("Hi", ["text", "image"], ["text"])


# ── the join, with Qwen's two stop tokens ───────────────────────────────────

IM_END, ENDOFTEXT, NEWLINE, TOKEN = 248046, 248044, [198], 42


def test_the_model_hands_its_llm_its_tokenizer_s_ids(model):
    tok = model.tokenizer
    assert tok.convert_tokens_to_ids("<|im_end|>") == IM_END
    assert tok.encode("\n", add_special_tokens=False) == NEWLINE
    assert {IM_END, ENDOFTEXT} <= set(model.config.stop_token_ids)


def _llm():
    # no weights: only the session bookkeeping is under test
    sub = TextSessionQwen3_5LLMSubmodule.__new__(TextSessionQwen3_5LLMSubmodule)
    NodeSubmodule.__init__(sub)
    sub.config = SimpleNamespace(stop_token_ids=frozenset({IM_END, ENDOFTEXT}))
    sub.turn_close_id = IM_END
    sub.turn_stop_ids = frozenset({IM_END, ENDOFTEXT})
    sub.newline_ids = list(NEWLINE)
    return sub


def _info(session, walk="decode", max_tokens=10, iters=8):
    return SimpleNamespace(
        session=session, graph_walk=walk, max_tokens=max_tokens,
        dynamic_loop_iter_counts={"decode_loop": iters},
        resource_configs={SAMPLER: SimpleNamespace(ignore_eos=False)},
        step_metadata={},
    )


def _resumed():
    return _info(RequestSession("s", resumed=True), walk="prefill_text")


@pytest.mark.parametrize("token, in_kv, join", [
    (IM_END, True, NEWLINE),
    (IM_END, False, [IM_END, *NEWLINE]),
    # finished on a stop token the template never closes with
    (ENDOFTEXT, True, [IM_END, *NEWLINE]),
    (ENDOFTEXT, False, [ENDOFTEXT, IM_END, *NEWLINE]),
    # cut: left open, only its own missing token comes back
    (TOKEN, True, []),
    (TOKEN, False, [TOKEN]),
])
def test_the_resumed_turn_puts_back_what_the_kv_is_missing(token, in_kv, join):
    sub = _llm()
    stops = sub.check_stop(
        "r0", _info(RequestSession("s"), iters=9),  # the budget: 9 + 1 >= 10
        {"new_token": [torch.tensor([token])]},
    )
    assert stops == {"decode_loop"}
    if in_kv:
        sub.session_state("s").add(OVERSHOT_LAST_ITER, True)

    assert sub.turn_join(_resumed()) == join


def test_the_batched_stop_check_records_the_turn_end_too():
    # the engine prefers it when the submodule has one, which Qwen's does
    sub = _llm()
    host_rows = SimpleNamespace(
        buffers={"new_token": torch.tensor([[TOKEN], [IM_END]])},
        request_ids=["a", "b"],
    )
    infos = {
        "a": _info(RequestSession("sa"), iters=0),  # still running
        "b": _info(RequestSession("sb"), iters=0),  # finished on <|im_end|>
    }

    stops = sub.check_stop_batched(["a", "b"], infos, host_rows)

    assert stops == {"b": {"decode_loop"}}
    assert "sa" not in sub.session_states
    resumed_b = _info(RequestSession("sb", resumed=True), walk="prefill_text")
    assert sub.turn_join(resumed_b) == [IM_END, *NEWLINE]


def test_get_submodule_builds_and_configures_the_session_llm(model, monkeypatch):
    # the way a worker builds it: `get_submodule` passes every argument
    # positionally, so an override has to take them however they come
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    def _bare(self, node_name, device, autocast_dtype=None, tp_group=None):
        sub = TextSessionQwen3_5LLMSubmodule.__new__(TextSessionQwen3_5LLMSubmodule)
        NodeSubmodule.__init__(sub)
        return sub

    monkeypatch.setattr(Qwen3_5DenseModel, "_create_submodule", _bare)
    model._submodule_cache.pop("LLM", None)
    try:
        sub = model.get_submodule("LLM", "cpu", None, None)
    finally:
        model._submodule_cache.pop("LLM", None)

    assert sub.turn_close_id == IM_END
    assert sub.newline_ids == NEWLINE
    assert {IM_END, ENDOFTEXT} <= sub.turn_stop_ids
