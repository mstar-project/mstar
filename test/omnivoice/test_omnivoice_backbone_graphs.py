"""The captured backbone path: its attention layout and its numerics.

``test_document_*`` / ``test_attention_step_*`` / ``test_node_resources_*``
    No GPU, no weights. Each request owns two documents (a zero-length
    unconditional one with CFG off), and the step spreads them over its ids.

``test_encode_head_matches_forward``
    GPU + checkpoint. ``encode`` + ``head`` -- what the captured region and the
    eager fallback run -- against the reference's packed ``forward`` through
    the same planned kernel: the same logits up to half-precision rounding.
"""

import os

import pytest
import torch

from mstar.model.omnivoice.components.backbone import CanvasItem, build_packed_canvas
from mstar.model.omnivoice.config import OmniVoiceConfig
from mstar.model.omnivoice.submodules import (
    BACKBONE_ATTN,
    OmniVoiceBackboneSubmodule,
    document_pairs,
)

MODEL_PATH = os.environ.get("OMNIVOICE_PATH", "k2-fsa/OmniVoice")
MASK_ID = 1024
NUM_CODEBOOK = 8


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


def test_node_resources_follow_backbone_graphs():
    from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel

    on = OmniVoiceModel("k2-fsa/OmniVoice", skip_weight_loading=True)
    off = OmniVoiceModel("k2-fsa/OmniVoice", skip_weight_loading=True, backbone_graphs=False)
    (spec,) = on.get_node_resources()
    assert spec.resource_key == BACKBONE_ATTN and spec.config.max_segments_per_request == 2
    assert off.get_node_resources() == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_encode_head_matches_forward():
    pytest.importorskip("omnivoice.models.omnivoice")
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

    from mstar.model.omnivoice.components.backbone import OmniVoiceBackbone

    model = OmniVoice.from_pretrained(MODEL_PATH, dtype=torch.float16).eval().cuda()
    apply_flashinfer(model, enable_cuda_graph=False)
    backbone = OmniVoiceBackbone(model, torch.float16).eval()

    gen = torch.Generator().manual_seed(0)
    items = []
    for rid, (prefix_len, target_len, guidance) in enumerate([(23, 40, 2.0), (31, 25, 0.0), (17, 52, 2.0)]):
        tokens = torch.full((1, NUM_CODEBOOK, target_len), MASK_ID, dtype=torch.long)
        tokens[..., ::3] = torch.randint(0, MASK_ID, tokens[..., ::3].shape, generator=gen)
        items.append(CanvasItem(
            request_id=str(rid),
            prefix_ids=torch.randint(0, 1000, (NUM_CODEBOOK, prefix_len), generator=gen),
            prefix_audio_mask=torch.zeros(prefix_len, dtype=torch.bool),
            tokens=tokens,
            guidance_scale=guidance,
        ))
    canvas = build_packed_canvas(items, OmniVoiceConfig().audio_mask_id, torch.device("cuda"))

    with torch.inference_mode():
        want = backbone(canvas)
        hidden = backbone.encode(
            canvas.packed_ids[0].t(), canvas.audio_mask[0], canvas.position_ids[0].to(torch.int32),
            attn=model._fi_runner.wrapper,  # planned for this canvas by forward
        )
        got = backbone.head(hidden, canvas.tgt_index)
    # the residual adds fold into the next RMSNorm (a float sum, not a rounded
    # half-precision one), so the logits agree to rounding, not bit for bit
    assert torch.allclose(got.float(), want.float(), atol=0.25, rtol=0.02)
    assert (got.argmax(-1) == want.argmax(-1)).float().mean() > 0.95
