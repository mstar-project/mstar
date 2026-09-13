"""Parity of the OmniVoice port against the reference implementation.

The port reuses the reference's own fused-kernel forward, so the per-step
numerics should be *identical* to its fastest mode and merely *close* to its
unoptimised dense one.  The tiers say which is which:

``test_packed_layout_*`` / ``test_reveal_schedule_*`` / ``test_prefix_parity``
    No GPU, no weights (the prefix test needs the tokenizer).  Structure only.

``test_packed_parity_single``  — tier A
    ``apply_flashinfer``'d reference vs this port, one request, fully greedy.
    Same kernels, so: token-exact.

``test_packed_parity_batched`` — tier B
    Three requests of different lengths in one packed step, each compared to
    its own batch-of-one result.  This is the test that proves cross-request
    packing does not leak: it is the one thing M* does that the reference
    never does.

``test_dense_agreement``       — tier C
    Against the reference's dense path.  Greedy argmax flips on last-bit
    differences between kernels, so this asserts an agreement *rate*, not
    exactness.  Claiming bit-equality here would be claiming something untrue.

Run the structural tier anywhere::

    pytest test/omnivoice/test_parity.py -k "layout or schedule or prefix or mask_class"

and the rest on a box with a GPU and the checkpoint::

    OMNIVOICE_PATH=/path/to/OmniVoice pytest test/omnivoice/test_parity.py
"""

import os

import pytest
import torch

from mstar.model.omnivoice.components.backbone import CanvasItem, build_packed_canvas
from mstar.model.omnivoice.components.text import build_prefix
from mstar.model.omnivoice.components.unmask import (
    apply_reveal,
    build_reveal_schedule,
    predict_tokens_with_scoring,
)
from mstar.model.omnivoice.config import OmniVoiceConfig

MODEL_PATH = os.environ.get("OMNIVOICE_PATH", "k2-fsa/OmniVoice")
NUM_CODEBOOK = 8
MASK_ID = 1024

GREEDY = dict(
    num_step=8, guidance_scale=2.0, t_shift=0.1, layer_penalty_factor=5.0,
    # Both zero: no Gumbel noise on the reveal order, no sampling on the token.
    # Nothing random is left, so two correct implementations must agree exactly.
    position_temperature=0.0, class_temperature=0.0,
)


def _item(request_id: str, prefix_len: int, target_len: int, guidance_scale: float = 2.0):
    return CanvasItem(
        request_id=request_id,
        prefix_ids=torch.randint(0, 1000, (NUM_CODEBOOK, prefix_len)),
        prefix_audio_mask=torch.zeros(prefix_len, dtype=torch.bool),
        tokens=torch.full((1, NUM_CODEBOOK, target_len), MASK_ID, dtype=torch.long),
        guidance_scale=guidance_scale,
    )


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------


def test_packed_layout_doc_boundaries():
    """Two documents per item, laid out cond/uncond in request order."""
    items = [
        _item("a", prefix_len=40, target_len=100),
        _item("b", prefix_len=12, target_len=20),
    ]
    canvas = build_packed_canvas(items, MASK_ID, torch.device("cpu"))

    assert canvas.doc_lens == [140, 100, 32, 20]
    assert canvas.packed_ids.shape == (1, NUM_CODEBOOK, 292)
    # No padding at all: that is the point of packing over a dense batch.
    assert canvas.packed_ids.shape[-1] == sum(canvas.doc_lens)
    assert canvas.flat_target_total == 120
    assert canvas.tgt_index.numel() == 240


def test_packed_layout_positions_restart_per_document():
    """RoPE must not read document n+1 as a continuation of document n."""
    items = [
        _item("a", prefix_len=5, target_len=7),
        _item("b", prefix_len=3, target_len=4),
    ]
    canvas = build_packed_canvas(items, MASK_ID, torch.device("cpu"))
    pos = canvas.position_ids[0]

    cursor = 0
    for length in canvas.doc_lens:
        assert torch.equal(pos[cursor : cursor + length], torch.arange(length)), (
            "positions must restart at 0 in every packed document"
        )
        cursor += length


def test_packed_layout_target_gather_is_the_canvas():
    """tgt_index must point at the target region, not the prefix."""
    item = _item("a", prefix_len=9, target_len=6)
    canvas = build_packed_canvas([item], MASK_ID, torch.device("cpu"))

    gathered = canvas.audio_mask[0][canvas.tgt_index]
    assert gathered.all(), "every gathered position must be an audio position"

    # The conditional half sits at the tail of document 0, the unconditional
    # half is the whole of document 1.
    assert torch.equal(canvas.tgt_index[:6], torch.arange(9, 15))
    assert torch.equal(canvas.tgt_index[6:], torch.arange(15, 21))


def test_reveal_schedule_covers_the_canvas():
    """Whatever the rounding, every cell is revealed exactly once."""
    for num_step in (1, 8, 16, 32, 64):
        for target_len in (1, 7, 137, 750):
            schedule = build_reveal_schedule(
                target_len=target_len, num_codebook=NUM_CODEBOOK,
                num_step=num_step, t_shift=0.1,
            )
            assert len(schedule) == num_step
            assert min(schedule) >= 0
            assert sum(schedule) == target_len * NUM_CODEBOOK


def test_mask_class_is_never_predicted():
    item = _item("solo", prefix_len=10, target_len=6)
    canvas = build_packed_canvas([item], MASK_ID, torch.device("cpu"))
    logits = torch.randn(1, NUM_CODEBOOK, 2 * canvas.flat_target_total, 1025)
    c_logits, u_logits = canvas.slice_logits(logits, item)
    assert c_logits.shape == (1, NUM_CODEBOOK, 6, 1025)

    pred, _ = predict_tokens_with_scoring(
        c_logits, u_logits, MASK_ID, guidance_scale=2.0, class_temperature=0.0
    )
    assert (pred != MASK_ID).all()


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def test_prefix_parity():
    """The canvas prefix must match the reference token for token."""
    pytest.importorskip("omnivoice")
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceConfig as RefConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # _prepare_inference_inputs reads only config, text_tokenizer and device.
    # OmniVoice.__new__ would skip __init__ and leave `.device` -- a
    # PreTrainedModel property over parameters() -- raising, so stand in a
    # plain object carrying the three attributes it actually touches.
    class _Ref:
        config = RefConfig()
        text_tokenizer = tokenizer
        device = torch.device("cpu")

    ref_model = _Ref()

    cases = [
        dict(text="Xin chào, đây là giọng đọc tiếng Việt.", language="Vietnamese",
             instruct=None, ref_text=None, ref_len=0),
        dict(text="Hello [laughter] world.", language="English",
             instruct="a calm elderly man", ref_text=None, ref_len=0),
        dict(text="今天天气很好。", language="Chinese",
             instruct=None, ref_text="你好。", ref_len=37),
    ]

    for case in cases:
        ref_audio_tokens = (
            torch.randint(0, 1024, (NUM_CODEBOOK, case["ref_len"]))
            if case["ref_len"] else None
        )
        ours_ids, ours_mask = build_prefix(
            tokenizer=tokenizer, text=case["text"],
            num_audio_codebook=NUM_CODEBOOK, language=case["language"],
            instruct=case["instruct"], ref_text=case["ref_text"],
            ref_audio_tokens=ref_audio_tokens, denoise=True,
        )

        target_len = 5
        theirs = OmniVoice._prepare_inference_inputs(
            ref_model, text=case["text"], num_target_tokens=target_len,
            ref_text=case["ref_text"], ref_audio_tokens=ref_audio_tokens,
            lang=case["language"], instruct=case["instruct"], denoise=True,
        )
        assert torch.equal(ours_ids, theirs["input_ids"][0, :, :-target_len]), (
            f"prefix ids differ for {case['text']!r}"
        )
        assert torch.equal(ours_mask, theirs["audio_mask"][0, :-target_len]), (
            f"audio mask differs for {case['text']!r}"
        )


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def _load_patched():
    """The reference with its packed flashinfer path applied, on the GPU."""
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

    model = OmniVoice.from_pretrained(MODEL_PATH, dtype=torch.float16).eval().cuda()
    apply_flashinfer(model, enable_cuda_graph=False)
    return model


def _run_ours(backbone, config, specs):
    """Drive the port's packed loop over ``specs``; returns each finished canvas.

    ``specs`` is a list of ``(text, language, target_len)``.
    """
    items, schedules = [], []
    for idx, (text, language, target_len) in enumerate(specs):
        prefix_ids, prefix_audio_mask = build_prefix(
            tokenizer=backbone.model.text_tokenizer, text=text,
            num_audio_codebook=config.num_audio_codebook,
            language=language, denoise=True,
        )
        items.append(CanvasItem(
            request_id=f"r{idx}",
            prefix_ids=prefix_ids.cuda(),
            prefix_audio_mask=prefix_audio_mask.cuda(),
            tokens=torch.full(
                (1, config.num_audio_codebook, target_len),
                config.audio_mask_id, dtype=torch.long, device="cuda",
            ),
            guidance_scale=GREEDY["guidance_scale"],
        ))
        schedules.append(build_reveal_schedule(
            target_len=target_len, num_codebook=config.num_audio_codebook,
            num_step=GREEDY["num_step"], t_shift=GREEDY["t_shift"],
        ))

    for k in range(GREEDY["num_step"]):
        canvas = build_packed_canvas(items, config.audio_mask_id, torch.device("cuda"))
        logits = backbone(canvas).to(torch.float32)
        for item, schedule in zip(items, schedules, strict=True):
            c_logits, u_logits = canvas.slice_logits(logits, item)
            pred, scores = predict_tokens_with_scoring(
                c_logits, u_logits, config.audio_mask_id,
                guidance_scale=GREEDY["guidance_scale"],
                class_temperature=GREEDY["class_temperature"],
            )
            apply_reveal(
                tokens=item.tokens, pred_tokens=pred, scores=scores,
                reveal_count=schedule[k], audio_mask_id=config.audio_mask_id,
                layer_penalty_factor=GREEDY["layer_penalty_factor"],
                position_temperature=GREEDY["position_temperature"],
            )
    return [item.tokens[0].cpu() for item in items]


def _run_reference(model, specs):
    from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig

    task = GenerationTask(
        batch_size=len(specs), texts=[s[0] for s in specs],
        target_lens=[s[2] for s in specs], langs=[s[1] for s in specs],
        instructs=[None] * len(specs), ref_texts=[None] * len(specs),
        ref_audio_tokens=[None] * len(specs), ref_rms=[None] * len(specs),
    )
    out = model._generate_iterative(
        task, OmniVoiceGenerationConfig(**GREEDY, denoise=True)
    )
    return [t.cpu() for t in out]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_packed_parity_single():
    """Tier A: same kernels, one request, token-exact."""
    pytest.importorskip("omnivoice")
    from mstar.model.omnivoice.components.backbone import OmniVoiceBackbone

    config = OmniVoiceConfig()
    model = _load_patched()
    specs = [("Xin chào, đây là một câu thử.", "Vietnamese", 60)]

    theirs = _run_reference(model, specs)[0]
    ours = _run_ours(OmniVoiceBackbone(model, torch.float16).eval(), config, specs)[0]

    assert (ours != config.audio_mask_id).all(), "canvas left masked cells"
    assert torch.equal(ours, theirs), (
        f"{int((ours != theirs).sum())}/{ours.numel()} cells differ from the "
        "reference's packed path, which runs the identical kernels"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_packed_parity_batched():
    """Tier B: packing several requests must not change any of them.

    Cross-request packing is the thing the reference never does, so this is
    where a boundary bug would live -- a document attending past its own edge,
    or a position id continuing from the previous request.
    """
    pytest.importorskip("omnivoice")
    from mstar.model.omnivoice.components.backbone import OmniVoiceBackbone

    config = OmniVoiceConfig()
    backbone = OmniVoiceBackbone(_load_patched(), torch.float16).eval()

    specs = [
        ("Xin chào, đây là một câu thử.", "Vietnamese", 60),
        ("Hello world.", "English", 25),
        ("今天天气很好，我们出去走走吧。", "Chinese", 90),
    ]
    together = _run_ours(backbone, config, specs)
    alone = [_run_ours(backbone, config, [spec])[0] for spec in specs]

    for i, (packed, solo) in enumerate(zip(together, alone, strict=True)):
        assert torch.equal(packed, solo), (
            f"request {i} changed when packed with others: "
            f"{int((packed != solo).sum())}/{packed.numel()} cells differ"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_dense_agreement():
    """Tier C: against the unoptimised dense path, an agreement rate.

    Fused QKV, flashinfer RoPE and the ragged kernel are mathematically the
    same as the dense path but not bit-identical, and a greedy argmax turns a
    last-bit difference into a different token.  So this measures agreement
    rather than asserting equality -- the reference's own fast mode would fail
    an equality check here too.
    """
    pytest.importorskip("omnivoice")
    from omnivoice.models.omnivoice import OmniVoice

    from mstar.model.omnivoice.components.backbone import OmniVoiceBackbone

    config = OmniVoiceConfig()
    specs = [("Xin chào, đây là một câu thử.", "Vietnamese", 60)]

    dense = OmniVoice.from_pretrained(MODEL_PATH, dtype=torch.float16).eval().cuda()
    theirs = _run_reference(dense, specs)[0]
    del dense
    torch.cuda.empty_cache()

    backbone = OmniVoiceBackbone(_load_patched(), torch.float16).eval()
    ours = _run_ours(backbone, config, specs)[0]

    agreement = float((ours == theirs).float().mean())
    assert agreement >= 0.98, (
        f"only {agreement:.3f} of cells agree with the dense reference; "
        "below ~0.98 this is a porting bug, not kernel noise"
    )
