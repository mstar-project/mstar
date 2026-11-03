"""The captured backbone path's attention layout. No GPU, no weights: each
request owns two documents (a zero-length unconditional one with CFG off), and
the step spreads them over its ids. Its numerics against the reference are in
``test_omnivoice_parity.py``.
"""

from mstar.model.omnivoice.submodules import (
    BACKBONE_ATTN,
    OmniVoiceBackboneSubmodule,
    document_pairs,
)


def test_document_pairs_keep_a_slot_per_unguided_request():
    assert document_pairs([(5, 3), (4, 0)]) == [5, 3, 4, 0]


def _spans(step):
    return [(seg.request_id, seg.span) for seg in step.segments]


def test_attention_step_two_documents_per_request():
    step = OmniVoiceBackboneSubmodule._attention_step(["a", "b"], [5, 3, 4, 0])
    assert _spans(step) == [("a", 5), ("a", 3), ("b", 4), ("b", 0)]
    assert not step.steps[BACKBONE_ATTN].causal


def test_attention_step_padded_replay_and_capture_layouts():
    # replay: two documents per real request, one zero per padding row
    replay = OmniVoiceBackboneSubmodule._attention_step(["a", "b", "p0", "p1"], [5, 3, 4, 0, 0, 0])
    assert [n for _, n in _spans(replay)] == [5, 3, 4, 0, 0, 0]
    assert [rid for rid, _ in _spans(replay)][:4] == ["a", "a", "b", "b"]
    # capture: one length per row
    capture = OmniVoiceBackboneSubmodule._attention_step(["p0", "p1"], [8, 8])
    assert _spans(capture) == [("p0", 8), ("p1", 8)]


def test_node_resources_do_not_follow_backbone_graphs():
    from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel

    on = OmniVoiceModel("k2-fsa/OmniVoice", skip_weight_loading=True)
    off = OmniVoiceModel("k2-fsa/OmniVoice", skip_weight_loading=True, backbone_graphs=False)
    # the eager layer loop attends through the same resource: the flag gates capture only
    for model in (on, off):
        (spec,) = model.get_node_resources()
        assert spec.resource_key == BACKBONE_ATTN and spec.config.max_segments_per_request == 2
