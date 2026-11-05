"""Graph-shape + routing tests for V-JEPA 2 Phase-3.E streaming rollout.

Phase 3.E adds a second rollout walk (``prefill_video_rollout_streaming``)
that places ``EMIT_TO_CLIENT`` directly on the ``rollout_predictor``
section so each iter's ``predicted_hidden`` is delivered to the client as
soon as the iter completes, instead of being accumulated and emitted once
at loop completion (the batched walk's behavior).  Same ``rollout_predictor``
node, same submodule, same engine type, same ``register_loop_stop``
semantics — only the emit topology differs.

These tests:
  * Confirm both walks appear in ``get_graph_walk_graphs()`` for masked
    and AC configs.
  * Confirm the streaming walk's section has the EMIT_TO_CLIENT edge and
    the Loop has empty ``accumulated_outputs``.
  * Confirm the batched walk still has ``accumulated_outputs`` and no
    section-level EMIT_TO_CLIENT (regression check — a double-emit would
    deliver 2*H chunks instead of H).
  * Confirm ``_initial_walk`` routes to the right walk based on the
    ``stream_rollout`` flag combined with ``rollout_horizon``.

Pure CPU, ``skip_weight_loading=True`` — no GPU or HF cache needed.
"""

from __future__ import annotations

import pytest

# mstar.engine import fails on certain local torch builds that are missing
# newer dynamo-config attributes.  Guard at module level so collection of
# the pure-graph tests doesn't crash in those envs.
try:
    from mstar.graph.base import DynamicLoop, GraphNode, Sequential
    from mstar.graph.special_destinations import EMIT_TO_CLIENT
    from mstar.model.vjepa2.vjepa2_model import VJepa2Model
except (ImportError, AttributeError) as e:  # pragma: no cover - env-specific
    pytest.skip(
        f"Cannot import VJepa2Model in this env: {e}", allow_module_level=True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_masked_model() -> VJepa2Model:
    return VJepa2Model(
        model_path_hf="facebook/vjepa2-vitl-fpc64-256",
        skip_weight_loading=True,
        predictor_kind="masked",
    )


def _make_ac_model() -> VJepa2Model:
    return VJepa2Model(
        model_path_hf="facebook/vjepa2-ac-vitg",
        skip_weight_loading=True,
        predictor_kind="ac",
    )


def _find_rollout_loop(walk: Sequential) -> DynamicLoop:
    """Rollout walks are ``Sequential([video_encoder_node, DynamicLoop(...)])``."""
    assert isinstance(walk, Sequential), f"expected Sequential, got {type(walk).__name__}"
    for sec in walk.sections:
        if isinstance(sec, DynamicLoop):
            return sec
    raise AssertionError("no DynamicLoop found in rollout walk")


def _section_graph_node(loop: DynamicLoop) -> GraphNode:
    """The section of our rollout loop is a single ``rollout_predictor``
    GraphNode (no Sequential / Parallel nesting)."""
    assert isinstance(loop.section, GraphNode), (
        f"expected GraphNode section, got {type(loop.section).__name__}"
    )
    return loop.section


def _has_emit_to_client(outputs: list, name: str) -> bool:
    return any(
        edge.next_node == EMIT_TO_CLIENT and edge.name == name for edge in outputs
    )


def _loopback_names(section: GraphNode) -> set[str]:
    return {
        edge.name
        for edge in section.outputs
        if edge.next_node == "rollout_predictor"
    }


# ---------------------------------------------------------------------------
# Graph-shape: masked predictor
# ---------------------------------------------------------------------------


class TestStreamingWalkShapeMasked:
    def test_both_walks_registered(self):
        m = _make_masked_model()
        walks = m.get_graph_walk_graphs()
        assert VJepa2Model.PREFILL_VIDEO_ROLLOUT in walks
        assert VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING in walks

    def test_streaming_section_emits_to_client(self):
        m = _make_masked_model()
        walk = m.get_graph_walk_graphs()[
            VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING
        ]
        section = _section_graph_node(_find_rollout_loop(walk))
        assert _has_emit_to_client(section.outputs, "predicted_hidden"), (
            "streaming section must have EMIT_TO_CLIENT edge on predicted_hidden"
        )

    def test_streaming_emit_edge_is_non_persist(self):
        """Per-iter emit: each tensor is one-shot to the client; no
        cross-iter retention needed.  Matches bagel decode + qwen3_omni
        thinker_decode, which both use ``persist=False`` for per-iter
        EMIT_TO_CLIENT edges."""
        m = _make_masked_model()
        section = _section_graph_node(
            _find_rollout_loop(
                m.get_graph_walk_graphs()[
                    VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING
                ]
            )
        )
        emit_edges = [
            e for e in section.outputs
            if e.next_node == EMIT_TO_CLIENT and e.name == "predicted_hidden"
        ]
        assert len(emit_edges) == 1
        assert emit_edges[0].persist is False

    def test_streaming_loop_has_empty_accumulated_outputs(self):
        m = _make_masked_model()
        loop = _find_rollout_loop(
            m.get_graph_walk_graphs()[VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING]
        )
        assert loop.accumulated_outputs == [], (
            "streaming walk must not accumulate — per-iter emit from section "
            "would combine with accumulated_outputs to produce a double emit"
        )

    def test_batched_walk_retains_accumulated_and_no_section_emit(self):
        """Regression: the pre-P3.E accumulated-outputs path must stay
        intact.  A double emit (per-iter section + completion-time
        accumulated) would deliver 2*H chunks to the client."""
        m = _make_masked_model()
        walk = m.get_graph_walk_graphs()[VJepa2Model.PREFILL_VIDEO_ROLLOUT]
        loop = _find_rollout_loop(walk)
        section = _section_graph_node(loop)

        assert any(
            edge.next_node == EMIT_TO_CLIENT and edge.name == "predicted_hidden"
            for edge in loop.accumulated_outputs
        ), "batched walk must emit via accumulated_outputs"

        assert not _has_emit_to_client(section.outputs, "predicted_hidden"), (
            "batched walk must NOT have section-level EMIT_TO_CLIENT"
        )

    def test_streaming_section_preserves_sliding_window_loopbacks(self):
        """Streaming adds an extra output edge, it must not remove any
        loop-back edges — otherwise the sliding-window geometry breaks.
        """
        m = _make_masked_model()
        section = _section_graph_node(
            _find_rollout_loop(
                m.get_graph_walk_graphs()[
                    VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING
                ]
            )
        )
        lb = _loopback_names(section)
        assert "encoder_hidden" in lb, "missing encoder_hidden loop-back"
        assert "predicted_hidden" in lb, (
            "missing predicted_hidden self-ref (required for Loop.cache_outputs "
            "bookkeeping even though streaming doesn't use accumulated_outputs)"
        )


# ---------------------------------------------------------------------------
# Graph-shape: AC predictor
# ---------------------------------------------------------------------------


class TestStreamingWalkShapeAC:
    def test_ac_both_walks_registered(self):
        m = _make_ac_model()
        walks = m.get_graph_walk_graphs()
        assert VJepa2Model.PREFILL_VIDEO_ROLLOUT in walks
        assert VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING in walks

    def test_ac_streaming_section_emits_plus_loopbacks(self):
        m = _make_ac_model()
        section = _section_graph_node(
            _find_rollout_loop(
                m.get_graph_walk_graphs()[
                    VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING
                ]
            )
        )
        assert _has_emit_to_client(section.outputs, "predicted_hidden")
        # AC identity loop-backs preserved (submodule slices from these per
        # iter via iter_idx; they must keep routing every iter or the
        # per-iter action/state slice falls off the end).
        required = {"encoder_hidden", "predicted_hidden", "actions", "states"}
        assert required.issubset(_loopback_names(section))

    def test_ac_streaming_input_ids_match_batched(self):
        """Section-level ``input_ids`` must be identical between variants —
        the submodule signature + registration don't change between
        batched and streaming walks."""
        m = _make_ac_model()
        walks = m.get_graph_walk_graphs()
        batched = _section_graph_node(_find_rollout_loop(walks[VJepa2Model.PREFILL_VIDEO_ROLLOUT]))
        streaming = _section_graph_node(
            _find_rollout_loop(walks[VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING])
        )
        assert batched.input_ids == streaming.input_ids


# ---------------------------------------------------------------------------
# _initial_walk routing
# ---------------------------------------------------------------------------


class TestInitialWalkRouting:
    @pytest.mark.parametrize("model_factory", [_make_masked_model, _make_ac_model])
    def test_stream_flag_with_horizon_selects_streaming(self, model_factory):
        m = model_factory()
        walk = m._initial_walk({"rollout_horizon": 4, "stream_rollout": True})
        assert walk == VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING

    @pytest.mark.parametrize("model_factory", [_make_masked_model, _make_ac_model])
    def test_default_horizon_selects_batched(self, model_factory):
        m = model_factory()
        walk = m._initial_walk({"rollout_horizon": 4})
        assert walk == VJepa2Model.PREFILL_VIDEO_ROLLOUT

    @pytest.mark.parametrize("model_factory", [_make_masked_model, _make_ac_model])
    def test_explicit_false_stream_selects_batched(self, model_factory):
        m = model_factory()
        walk = m._initial_walk({"rollout_horizon": 4, "stream_rollout": False})
        assert walk == VJepa2Model.PREFILL_VIDEO_ROLLOUT

    @pytest.mark.parametrize("model_factory", [_make_masked_model, _make_ac_model])
    def test_stream_flag_without_horizon_stays_single_pass(self, model_factory):
        """``stream_rollout`` only matters when ``rollout_horizon > 1``;
        single-pass prefill stays single-pass."""
        m = model_factory()
        walk = m._initial_walk({"stream_rollout": True})
        assert walk == VJepa2Model.PREFILL_VIDEO

    @pytest.mark.parametrize("model_factory", [_make_masked_model, _make_ac_model])
    def test_no_kwargs_single_pass(self, model_factory):
        m = model_factory()
        assert m._initial_walk({}) == VJepa2Model.PREFILL_VIDEO
        assert m._initial_walk(None) == VJepa2Model.PREFILL_VIDEO

    @pytest.mark.parametrize("model_factory", [_make_masked_model, _make_ac_model])
    def test_h1_falls_through_to_single_pass(self, model_factory):
        """H=1 is degenerate (one forward) — save the loop overhead.
        Stream flag shouldn't override this: H=1 single-pass stays
        single-pass even when ``stream_rollout=True``."""
        m = model_factory()
        assert m._initial_walk({"rollout_horizon": 1}) == VJepa2Model.PREFILL_VIDEO
        assert m._initial_walk(
            {"rollout_horizon": 1, "stream_rollout": True}
        ) == VJepa2Model.PREFILL_VIDEO
