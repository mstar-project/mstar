"""Tests for the multimodal graph + scheduling wiring (step 5c).

Covers ``get_graph_walk_graphs``, ``get_partitions``, the prefill-
schedule helpers, ``get_initial_forward_pass_args`` and
``get_partition_forward_pass_args`` — all routed by
``input_modalities`` instead of the text-only `prefill`/`decode`
walks from step 3f.

These tests build a bare ``MingFlashOmniModel`` via ``__new__`` so
they exercise the routing/scheduling code paths without loading the
~238 GB ckpt. Snapshot-gated end-to-end serve verification is a
separate task (the 4-GPU dev box can't fit the full TP=8 model).
"""

from __future__ import annotations

from typing import Any

import pytest

from mminf.conductor.request_info import CurrentForwardConductorMetadata
from mminf.engine.base import EngineType
from mminf.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mminf.graph.special_destinations import EMIT_TO_CLIENT
from mminf.model.ming_omni_flash.config import (
    AudioEncoderConfig,
    MingFlashOmniModelConfig,
    ThinkerLLMConfig,
    VisionEncoderConfig,
)
from mminf.model.ming_omni_flash.ming_omni_flash_model import MingFlashOmniModel


# ---------------------------------------------------------------------------
# Tiny model instance (no weights, no tokenizer)
# ---------------------------------------------------------------------------


def _bare_model() -> MingFlashOmniModel:
    """Return a MingFlashOmniModel with just enough state for graph routing.

    Bypasses __init__ (which downloads the snapshot + tokenizer); injects
    a tiny config so the prefill scheduler / partition state machine can
    run without loading the 100B-param ckpt.
    """
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.config = MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        thinker_llm=ThinkerLLMConfig(),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
    )
    inst._submodule_cache = {}
    return inst


# Stub TensorPointerInfo: the scheduling code only ever reads its
# presence (length checks + per-step dict construction), not any field,
# so a plain object is enough for unit tests.
class _StubTI:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __repr__(self) -> str:
        return f"<TI {self.tag}>"


# ---------------------------------------------------------------------------
# get_graph_walk_graphs / get_partitions
# ---------------------------------------------------------------------------


def test_graph_walk_graphs_emits_five_walks() -> None:
    model = _bare_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks.keys()) == {
        "prefill_text", "prefill_audio",
        "prefill_vision", "prefill_video",
        "thinker_decode",
    }


def test_prefill_text_walk_is_single_thinker_node() -> None:
    """Text prefill is a bare Thinker node with one EMIT_TO_CLIENT edge."""
    model = _bare_model()
    walks = model.get_graph_walk_graphs()
    node = walks["prefill_text"]
    assert isinstance(node, GraphNode)
    assert node.name == "Thinker"
    assert set(node.input_names) == {"text_inputs"}
    assert len(node.outputs) == 1
    assert node.outputs[0].next_node == EMIT_TO_CLIENT
    assert node.outputs[0].name == "new_token"
    assert node.outputs[0].output_modality == "text"
    assert node.outputs[0].persist is True


def test_prefill_audio_walk_routes_encoder_then_thinker() -> None:
    model = _bare_model()
    walks = model.get_graph_walk_graphs()
    seq = walks["prefill_audio"]
    assert isinstance(seq, Sequential)
    assert len(seq.sections) == 2
    encoder, thinker = seq.sections
    assert encoder.name == "audio_encoder"
    assert set(encoder.input_names) == {"audio_features", "audio_seqlens"}
    assert len(encoder.outputs) == 1
    assert encoder.outputs[0].next_node == "Thinker"
    assert encoder.outputs[0].name == "audio_embeds"
    # Second node is the Thinker; its only input is the encoder's audio_embeds.
    assert thinker.name == "Thinker"
    assert set(thinker.input_names) == {"audio_embeds"}


def test_prefill_vision_walk_threads_grid_to_thinker() -> None:
    """vision_encoder is first; Thinker also reads image_grid_thw."""
    model = _bare_model()
    walks = model.get_graph_walk_graphs()
    seq = walks["prefill_vision"]
    assert isinstance(seq, Sequential)
    encoder, thinker = seq.sections
    assert encoder.name == "vision_encoder"
    assert set(encoder.input_names) == {"pixel_values", "image_grid_thw"}
    assert thinker.name == "Thinker"
    # Thinker needs image_grid_thw for the 3D MRoPE math.
    assert "vision_embeds" in thinker.input_names
    assert "image_grid_thw" in thinker.input_names


def test_prefill_video_walk_adds_video_second_per_grid() -> None:
    model = _bare_model()
    walks = model.get_graph_walk_graphs()
    seq = walks["prefill_video"]
    assert isinstance(seq, Sequential)
    encoder, thinker = seq.sections
    assert encoder.name == "vision_encoder"
    assert "video_second_per_grid" in thinker.input_names


def test_thinker_decode_is_loop() -> None:
    model = _bare_model()
    walks = model.get_graph_walk_graphs()
    loop = walks["thinker_decode"]
    assert isinstance(loop, Loop)
    assert loop.section.name == "Thinker"
    # The loop must produce a feedback edge so prior token reaches next iter.
    feedback = [e for e in loop.section.outputs if e.next_node == "Thinker"]
    assert len(feedback) == 1
    assert feedback[0].name == "text_inputs"


def test_get_partitions_lists_all_five_walks() -> None:
    model = _bare_model()
    parts = model.get_partitions()
    assert len(parts) == 1
    p = parts[0]
    assert p.name == "Thinker"
    assert p.initial_walk == "prefill_text"
    assert p.graph_walks == {
        "prefill_text", "prefill_audio",
        "prefill_vision", "prefill_video",
        "thinker_decode",
    }


# ---------------------------------------------------------------------------
# _build_thinker_prefill_schedule
# ---------------------------------------------------------------------------


def test_build_schedule_text_only() -> None:
    model = _bare_model()
    text_ti = _StubTI("text")
    sched = model._build_thinker_prefill_schedule(
        input_modalities=["text"],
        input_signals={"text_inputs": [text_ti]},
    )
    assert sched == [("prefill_text", {"text_inputs": text_ti})]


def test_build_schedule_text_then_audio_then_image() -> None:
    """Schedule honors input_modalities order."""
    model = _bare_model()
    sig = {
        "text_inputs": [_StubTI("t0")],
        "audio_features": [_StubTI("a0")],
        "audio_seqlens": [_StubTI("aseq0")],
        "pixel_values": [_StubTI("p0")],
        "image_grid_thw": [_StubTI("g0")],
    }
    sched = model._build_thinker_prefill_schedule(
        input_modalities=["text", "audio", "image"],
        input_signals=sig,
    )
    assert [w for w, _ in sched] == [
        "prefill_text", "prefill_audio", "prefill_vision",
    ]
    # Audio step carries the optional seqlens.
    assert sched[1][1]["audio_seqlens"] is sig["audio_seqlens"][0]
    # Image step carries the grid.
    assert sched[2][1]["image_grid_thw"] is sig["image_grid_thw"][0]


def test_build_schedule_video_carries_second_per_grid() -> None:
    model = _bare_model()
    sig = {
        "pixel_values_videos": [_StubTI("v0")],
        "video_grid_thw": [_StubTI("vg0")],
        "video_second_per_grid": [_StubTI("vspg0")],
    }
    sched = model._build_thinker_prefill_schedule(
        input_modalities=["video"], input_signals=sig,
    )
    assert sched[0][0] == "prefill_video"
    entry = sched[0][1]
    assert entry["pixel_values"] is sig["pixel_values_videos"][0]
    assert entry["image_grid_thw"] is sig["video_grid_thw"][0]
    assert entry["video_second_per_grid"] is sig["video_second_per_grid"][0]


def test_build_schedule_skips_modalities_without_inputs() -> None:
    """input_modalities=['audio'] but no audio_features → empty schedule."""
    model = _bare_model()
    sched = model._build_thinker_prefill_schedule(
        input_modalities=["audio"], input_signals={},
    )
    assert sched == []


def test_build_schedule_unknown_modality_silently_ignored() -> None:
    """An unknown modality string doesn't crash — it just produces no step."""
    model = _bare_model()
    sched = model._build_thinker_prefill_schedule(
        input_modalities=["holographic"], input_signals={},
    )
    assert sched == []


# ---------------------------------------------------------------------------
# _get_thinker_prefill_inputs
# ---------------------------------------------------------------------------


def _make_metadata(schedule: list[tuple[str, dict[str, Any]]], step: int = 0):
    return CurrentForwardConductorMetadata(
        input_modalities=[],
        output_modalities=["text"],
        graph_walk=schedule[step][0],
        is_prefill=True,
        kwargs={"prefill_schedule": schedule, "prefill_step": step},
    )


def test_prefill_inputs_text_routes_only_to_thinker() -> None:
    model = _bare_model()
    text_ti = _StubTI("text")
    md = _make_metadata([("prefill_text", {"text_inputs": text_ti})])
    edges = model._get_thinker_prefill_inputs(md, {"text_inputs": [text_ti]})
    assert len(edges) == 1
    assert edges[0].next_node == "Thinker"
    assert edges[0].name == "text_inputs"
    assert edges[0].tensor_info == [text_ti]


def test_prefill_inputs_audio_routes_to_audio_encoder() -> None:
    model = _bare_model()
    af = _StubTI("af")
    aseq = _StubTI("aseq")
    md = _make_metadata([(
        "prefill_audio",
        {"audio_features": af, "audio_seqlens": aseq},
    )])
    edges = model._get_thinker_prefill_inputs(md, {})
    target_names = sorted((e.next_node, e.name) for e in edges)
    # Both audio inputs target the audio_encoder node.
    assert ("audio_encoder", "audio_features") in target_names
    assert ("audio_encoder", "audio_seqlens") in target_names


def test_prefill_inputs_vision_dual_edges_for_grid() -> None:
    """image_grid_thw goes to BOTH vision_encoder AND Thinker.

    The encoder needs the grid to compute spatial positions on the
    pixel patches; the Thinker also needs it for the 3D MRoPE math
    (sentinel position layout around the vision span).
    """
    model = _bare_model()
    pv = _StubTI("pv")
    grid = _StubTI("grid")
    md = _make_metadata([(
        "prefill_vision",
        {"pixel_values": pv, "image_grid_thw": grid},
    )])
    edges = model._get_thinker_prefill_inputs(md, {})
    pairs = sorted((e.next_node, e.name) for e in edges)
    assert ("vision_encoder", "pixel_values") in pairs
    assert ("vision_encoder", "image_grid_thw") in pairs
    assert ("Thinker", "image_grid_thw") in pairs


def test_prefill_inputs_video_routes_second_per_grid_to_thinker() -> None:
    model = _bare_model()
    md = _make_metadata([(
        "prefill_video",
        {
            "pixel_values": _StubTI("pv"),
            "image_grid_thw": _StubTI("grid"),
            "video_second_per_grid": _StubTI("spg"),
        },
    )])
    edges = model._get_thinker_prefill_inputs(md, {})
    pairs = sorted((e.next_node, e.name) for e in edges)
    assert ("Thinker", "video_second_per_grid") in pairs


# ---------------------------------------------------------------------------
# get_initial_forward_pass_args
# ---------------------------------------------------------------------------


def test_initial_args_text_only_starts_in_prefill_text() -> None:
    model = _bare_model()
    text_ti = _StubTI("text")
    args = model.get_initial_forward_pass_args(
        partition_name="Thinker",
        input_modalities=["text"],
        output_modalities=["text"],
        input_signals={"text_inputs": [text_ti]},
    )
    assert args.full_metadata.graph_walk == "prefill_text"
    assert args.full_metadata.is_prefill is True
    assert args.full_metadata.kwargs["prefill_step"] == 0
    assert len(args.full_metadata.kwargs["prefill_schedule"]) == 1
    # Single-modality request → is_last_prefill = True from the start.
    assert args.step_metadata["is_last_prefill"] is True


def test_initial_args_text_plus_image_orders_walks() -> None:
    model = _bare_model()
    args = model.get_initial_forward_pass_args(
        partition_name="Thinker",
        input_modalities=["text", "image"],
        output_modalities=["text"],
        input_signals={
            "text_inputs": [_StubTI("text")],
            "pixel_values": [_StubTI("pv")],
            "image_grid_thw": [_StubTI("grid")],
        },
    )
    assert args.full_metadata.graph_walk == "prefill_text"
    schedule = args.full_metadata.kwargs["prefill_schedule"]
    assert [w for w, _ in schedule] == ["prefill_text", "prefill_vision"]
    # Two-step schedule → first step is NOT the last.
    assert args.step_metadata["is_last_prefill"] is False


def test_initial_args_no_modalities_returns_done() -> None:
    """Empty schedule → request_done so the conductor doesn't hang."""
    model = _bare_model()
    args = model.get_initial_forward_pass_args(
        partition_name="Thinker",
        input_modalities=[],
        output_modalities=["text"],
        input_signals={},
    )
    assert args.request_done is True


def test_initial_args_rejects_unknown_partition() -> None:
    # Talker is now a valid partition name (step 6e-3); use an unported
    # partition (ImageGen, step 9) as the canonical 'unknown' here.
    model = _bare_model()
    with pytest.raises(ValueError, match="Unknown partition: 'ImageGen'"):
        model.get_initial_forward_pass_args(
            partition_name="ImageGen",
            input_modalities=["text"],
            output_modalities=["text"],
            input_signals={"text_inputs": [_StubTI("text")]},
        )


# ---------------------------------------------------------------------------
# get_partition_forward_pass_args state machine
# ---------------------------------------------------------------------------


def test_state_machine_advances_schedule_then_decodes_then_finishes() -> None:
    """Drive Thinker state machine across a 2-step prefill + decode + finish."""
    model = _bare_model()
    init = model.get_initial_forward_pass_args(
        partition_name="Thinker",
        input_modalities=["text", "audio"],
        output_modalities=["text"],
        input_signals={
            "text_inputs": [_StubTI("text")],
            "audio_features": [_StubTI("af")],
            "audio_seqlens": [_StubTI("aseq")],
        },
    )
    metadata = init.full_metadata
    assert metadata.graph_walk == "prefill_text"

    # Step 2: advance to second prefill walk (prefill_audio).
    args2 = model.get_partition_forward_pass_args(
        partition_name="Thinker",
        partition_metadata=metadata,
        persist_signals={"new_token": [_StubTI("ntok")]},
        new_tokens={"new_token": [42]},
    )
    assert args2.full_metadata.graph_walk == "prefill_audio"
    assert args2.full_metadata.is_prefill is True
    assert args2.step_metadata["is_last_prefill"] is True

    # Step 3: schedule exhausted → transition to thinker_decode.
    args3 = model.get_partition_forward_pass_args(
        partition_name="Thinker",
        partition_metadata=args2.full_metadata,
        persist_signals={"new_token": [_StubTI("ntok")]},
        new_tokens={"new_token": [42]},
    )
    assert args3.full_metadata.graph_walk == "thinker_decode"
    assert args3.full_metadata.is_prefill is False
    # Decode loop feedback edge is text_inputs <- new_token.
    assert any(e.name == "text_inputs" for e in args3.inputs)

    # Step 4: decode loop unwound → request_done.
    args4 = model.get_partition_forward_pass_args(
        partition_name="Thinker",
        partition_metadata=args3.full_metadata,
        persist_signals={},
        new_tokens={},
    )
    assert args4.request_done is True


# ---------------------------------------------------------------------------
# get_worker_graphs partial-deploy skipping (regression for the live
# bring-up KeyError: 'audio_encoder' — see model/base.py fix c06c99a)
# ---------------------------------------------------------------------------


def _talker_enabled_model() -> MingFlashOmniModel:
    """Bare model whose config DOES declare a talker (so the talker walk exists)."""
    from mminf.model.ming_omni_flash.config import (
        AudioVAEConfig,
        DiTBlockConfig,
        TalkerConfig,
        TalkerLLMConfig,
    )
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.config = MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        thinker_llm=ThinkerLLMConfig(),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
        talker=TalkerConfig(
            llm=TalkerLLMConfig(), flowmodel=DiTBlockConfig(),
            aggregator=DiTBlockConfig(), vae=AudioVAEConfig(),
        ),
    )
    inst._submodule_cache = {}
    return inst


def _write_yaml(tmp_path, node_groups: list[dict]) -> str:
    import json
    p = tmp_path / "cfg.yaml"
    # node_groups is JSON-compatible; YAML is a JSON superset so json.dumps
    # is a valid (if ugly) serialization the yaml loader accepts.
    p.write_text("model: ming_flash_omni\nmax_seq_len: 4096\n"
                 "node_groups: " + json.dumps(node_groups) + "\n")
    return str(p)


def _worker_graph_node_names(worker_graphs) -> set[str]:
    """Collect the real GraphNode names across a list of WorkerGraphs.

    A worker graph's ``section`` may be a Loop wrapper (e.g.
    ``thinker_decode_loop``) rather than a GraphNode, so reach the actual
    nodes via ``get_nodes()`` instead of reading ``section.name``.
    """
    names: set[str] = set()
    for wg in worker_graphs:
        names |= set(wg.section.get_nodes().keys())
    return names


def test_get_worker_graphs_thinker_only_skips_encoder_and_talker_walks(tmp_path) -> None:
    """Regression: a thinker-only node_groups must NOT KeyError on the
    encoder/talker walks — they're skipped because their nodes are absent.

    This is exactly the live-bring-up crash that motivated the
    model/base.py fix (KeyError: 'audio_encoder' during
    _divide_into_worker_graphs of the prefill_audio walk).
    """
    model = _talker_enabled_model()  # all walks (incl. talker) are emitted
    cfg = _write_yaml(tmp_path, [
        {"node_names": ["Thinker"], "ranks": [0, 1, 2, 3], "tp_size": 4},
    ])
    # Must not raise.
    wgs = model.get_worker_graphs(cfg)
    node_names = _worker_graph_node_names(wgs)
    # Only the Thinker is ever a worker-graph node; encoder/talker nodes
    # never appear because their walks were skipped.
    assert node_names == {"Thinker"}
    assert "audio_encoder" not in node_names
    assert "vision_encoder" not in node_names
    assert "Talker" not in node_names


def test_get_worker_graphs_full_omni_includes_all_nodes(tmp_path) -> None:
    """With encoders + Talker declared, their walks divide cleanly."""
    model = _talker_enabled_model()
    cfg = _write_yaml(tmp_path, [
        {"node_names": ["vision_encoder", "audio_encoder", "Talker"], "ranks": [0]},
        {"node_names": ["Thinker"], "ranks": [0, 1, 2, 3], "tp_size": 4},
    ])
    wgs = model.get_worker_graphs(cfg)
    node_names = _worker_graph_node_names(wgs)
    # All node types now present across the divided worker graphs.
    assert "Thinker" in node_names
    assert "vision_encoder" in node_names
    assert "audio_encoder" in node_names
    assert "Talker" in node_names


def test_get_worker_graphs_thinker_only_no_talker_config(tmp_path) -> None:
    """A model whose config has no talker emits no talker walk at all, and a
    thinker-only deploy still divides without error."""
    model = _bare_model()  # talker=None → no talker walk emitted
    cfg = _write_yaml(tmp_path, [
        {"node_names": ["Thinker"], "ranks": [0, 1, 2, 3], "tp_size": 4},
    ])
    wgs = model.get_worker_graphs(cfg)
    assert _worker_graph_node_names(wgs) == {"Thinker"}
