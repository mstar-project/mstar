"""The ``test_text_session`` deployment: BAGEL's LLM with sessions on its KV.

This is the model the session machinery is exercised against, so what matters is
that its declarations line up — the walks it serves, the resources it opens, the
session state it asks for, and the shipped config that tunes it. Everything here
is CPU-only and needs BAGEL's config and tokenizer (not its weights); it skips
when those are not cached.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest
import yaml

from mstar.conductor.request_info import DEFAULT_PARTITION
from mstar.engine.resources.attn.ragged.config import RaggedAttentionSpec
from mstar.graph.base import TensorPointerInfo
from mstar.model.registry import HF_MODELS, get_model_class
from mstar.model.sessions import (
    RequestSession,
    SessionOverflowPolicy,
    apply_sessions_yaml_overrides,
)
from mstar.model.test_text_session.model import TEXT_WALKS

CONFIG = Path("configs/test_text_session.yaml")


@pytest.fixture(scope="module")
def model():
    cls = get_model_class("test_text_session")
    try:
        return cls(
            model_path_hf=HF_MODELS["test_text_session"]["model_path_hf"],
        )
    except Exception as e:  # noqa: BLE001 — any download/auth failure is a skip
        pytest.skip(f"BAGEL's config/tokenizer is not available: {e!r}")


@pytest.fixture(scope="module")
def serving_config():
    return yaml.safe_load(CONFIG.read_text())


# ── what it serves ──────────────────────────────────────────────────────────

def test_only_the_text_walks_are_declared(model):
    assert sorted(model.get_graph_walk_graphs()) == sorted(TEXT_WALKS)
    assert model.nodes == ["LLM"]


def test_the_llm_s_resources_are_declared_and_the_vit_s_are_not(model):
    specs = model.get_node_resources()

    assert [spec.resource_key for spec in specs] == [
        "kv", "attn", "rope", "sampler",
    ]
    assert all(spec.nodes == {"LLM"} for spec in specs)
    assert not any(isinstance(spec, RaggedAttentionSpec) for spec in specs)


def test_prefix_reuse_is_off_so_a_session_miss_cannot_hide(model):
    assert model.prefix_key_streams() == {}


def test_a_text_prompt_tokenizes_to_one_span(model):
    out = model.process_prompt("Who painted Guernica?", ["text"], ["text"])

    assert list(out) == ["text_inputs"]
    assert len(out["text_inputs"]) == 1
    assert out["text_inputs"][0].numel() > 0


# ── what a turn renders to ──────────────────────────────────────────────────

def _ids(model, prompt, session=None):
    out = model.process_prompt(prompt, ["text"], ["text"], session=session)
    assert list(out) == ["text_inputs"]
    assert len(out["text_inputs"]) == 1
    return model.tokenizer.decode(out["text_inputs"][0].tolist())


def test_a_turn_that_opens_a_session_renders_the_system_prompt(model):
    rendered = _ids(model, "Hello", session=RequestSession("s"))

    assert model.BAGEL_DEFAULT_SYSTEM_PROMPT in rendered


def test_a_resuming_turn_leaves_the_system_prompt_out(model):
    # it is appended to a KV that already holds the conversation, so the system
    # block would land mid-conversation and introduce the model to itself again
    rendered = _ids(model, "Hello", session=RequestSession("s", resumed=True))

    assert model.BAGEL_DEFAULT_SYSTEM_PROMPT not in rendered
    assert rendered == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"


def test_every_turn_names_the_role_that_speaks_it(model):
    # without a role label the model cannot tell its own turn from the user's,
    # and answers "what is my name?" as though the name were its own
    opening = _ids(model, "Hello", session=RequestSession("s"))

    assert "<|im_start|>system\n" in opening
    assert "<|im_start|>user\nHello" in opening
    assert opening.endswith("<|im_start|>assistant\n")


def test_a_resuming_turn_is_the_opening_turn_minus_its_system_block(model):
    opening = _ids(model, "Hello", session=RequestSession("s"))
    resumed = _ids(model, "Hello", session=RequestSession("s", resumed=True))

    assert opening.endswith(resumed)


def test_a_sessionless_turn_renders_like_one_opening_a_session(model):
    assert _ids(model, "Hello") == _ids(model, "Hello", RequestSession("s"))


@pytest.mark.parametrize(
    ("in_mods", "out_mods", "match"),
    [
        (["image", "text"], ["text"], "no encoder for image"),
        (["text"], ["image"], "generates text only"),
    ],
)
def test_anything_but_text_is_refused_at_the_api_server(
    model, in_mods, out_mods, match,
):
    with pytest.raises(ValueError, match=match):
        model.process_prompt("hi", in_mods, out_mods)


# ── the request path the conductor drives ───────────────────────────────────

def _signals(model, text: str):
    """The tokenized prompt as the data worker would hand it over."""
    tokens = model.process_prompt(text, ["text"], ["text"])["text_inputs"][0]
    return {"text_inputs": [TensorPointerInfo(
        dims=list(tokens.shape), dtype="int64", nbytes=tokens.numel() * 8,
        address=0, stride=[1], uuid="u0", source_session_id="h:0",
        source_entity="api_server",
    )]}


def _initial(model, text="Remember 8675309."):
    return model.get_initial_forward_pass_args(
        partition_name=DEFAULT_PARTITION,
        input_modalities=["text"], output_modalities=["text"],
        input_signals=_signals(model, text), model_kwargs={},
    )


def test_a_turn_starts_in_prefill_text_and_moves_to_decode(model):
    args = _initial(model)

    assert args.full_metadata.graph_walk == "prefill_text"
    assert args.full_metadata.is_prefill is True
    assert [(e.next_node, e.name) for e in args.inputs] == [("LLM", "text_inputs")]
    assert len(args.full_metadata.kwargs["prefill_schedule"]) == 1

    following = model.get_partition_forward_pass_args(
        partition_name=DEFAULT_PARTITION,
        partition_metadata=args.full_metadata,
        persist_signals={"new_token": []},
    )

    assert following.full_metadata.graph_walk == "decode"
    assert following.request_done is False


def test_the_per_request_resource_configs_cover_the_text_walks(model):
    configs = model.get_request_resource_configs(
        {DEFAULT_PARTITION: _initial(model)}, {},
    )

    assert sorted(configs) == ["kv", "sampler"]
    labels = configs["kv"].needed_labels_per_node_walk
    assert all(
        labels[("LLM", walk)] == ["main"] for walk in TEXT_WALKS
    ), "a text walk opened cache labels other than the one stream it writes"


# ── the session declaration and the config over it ──────────────────────────

def test_it_holds_its_kv_for_the_session(model):
    config = model.get_sessions_config()

    assert set(config.resources) == {"kv"}
    assert config.resources["kv"].max_state == 64
    assert config.max_concurrent_sessions == 4


def test_every_session_resource_is_one_the_model_declares(model):
    # the check the engine makes at load, so a typo fails here instead of on a
    # GPU box after the weights are in
    declared = {spec.resource_key for spec in model.get_node_resources()}

    assert model.get_sessions_config().resources.keys() <= declared


def test_the_shipped_config_merges_over_the_declaration(model, serving_config):
    merged = apply_sessions_yaml_overrides(
        model.get_sessions_config(), serving_config,
    )

    assert merged is not None
    assert set(merged.resources) == {"kv"}
    assert (
        merged.resources["kv"].overflow_policy is SessionOverflowPolicy.ERROR
    )
    assert merged.max_timeout_s == 1800.0


def test_the_shipped_config_names_this_model_and_the_llm_node(serving_config):
    assert serving_config["model"] == "test_text_session"
    assert [g["node_names"] for g in serving_config["node_groups"]] == [["LLM"]]


def test_the_config_puts_both_walks_on_rank_zero(model):
    graphs = model.get_worker_graphs(str(CONFIG))

    # one worker graph per walk: their sections differ (a node, then a loop)
    assert {walk for wg in graphs for walk in wg.graph_walks} == set(TEXT_WALKS)
    assert all(set(wg.section.get_nodes()) == {"LLM"} for wg in graphs)
    assert all(wg.ranks == [0] and wg.tp_size == 1 for wg in graphs)
