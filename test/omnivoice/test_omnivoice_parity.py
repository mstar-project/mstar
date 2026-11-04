"""Parity of the OmniVoice port against the reference implementation.

The port's packing, scoring and reveal loop run over the reference's own
fused-kernel packed forward (``_packed_logits``), so the per-step numerics
should be *identical* to its fastest mode and merely *close* to its
unoptimised dense one; ``encode`` + ``head``, mstar's layer loop over the same
fused weights, is close to that forward. The tiers say which is which:

``test_packed_layout_*`` / ``test_reveal_schedule_*`` / ``test_prefix_parity``
    No GPU, no weights (the prefix test needs the tokenizer).  Structure only.

``test_packed_parity_single``  — tier A
    ``apply_flashinfer``'d reference vs this port, one request, fully greedy.
    Same kernels, so: token-exact.

``test_packed_parity_batched`` — tier B
    Step-0 logits for one request, packed against different neighbours. This
    is the test that proves cross-request packing does not leak: it is the one
    thing M* does that the reference never does. It compares logits rather
    than finished canvases because the document count changes the ragged
    kernel's reduction order by about one fp16 ulp, and eight greedy reveals
    turn that into a visibly different canvas without anything being wrong.

``test_encode_head_matches_the_packed_reference``
    ``encode`` + ``head`` -- what the captured region and the eager fallback
    run -- against the packed forward through the same planned kernel: the
    same logits up to half-precision rounding.

``test_dense_agreement``       — tier C
    Against the reference's dense path, after a single step.  Greedy argmax
    flips on last-bit differences between kernels, so this asserts an
    agreement *rate*, not exactness.  Claiming bit-equality here would be
    claiming something untrue, and measuring it after all eight steps would
    measure how fast a one-ulp gap compounds rather than porting accuracy.

Run the structural tier anywhere::

    pytest test/omnivoice/test_omnivoice_parity.py -k "layout or schedule or prefix or mask_class"

and the rest on a box with a GPU and the checkpoint::

    OMNIVOICE_PATH=/path/to/OmniVoice pytest test/omnivoice/test_omnivoice_parity.py
"""

import os

import pytest
import torch
import torch.nn.functional as F

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.omnivoice.components.backbone import CanvasItem, build_packed_canvas
from mstar.model.omnivoice.components.text import build_prefix
from mstar.model.omnivoice.components.unmask import build_reveal_schedule, filter_top_k, unmask_step
from mstar.model.omnivoice.config import OmniVoiceConfig
from mstar.model.omnivoice.submodules import (
    UNMASK_LOOP_NAME,
    OmniVoiceBackboneSubmodule,
)

MODEL_PATH = os.environ.get("OMNIVOICE_PATH", "k2-fsa/OmniVoice")
NUM_CODEBOOK = 8
MASK_ID = 1024

GREEDY = dict(
    num_step=8, guidance_scale=2.0, t_shift=0.1, layer_penalty_factor=5.0,
    # Both zero: no Gumbel noise on the reveal order, no sampling on the token.
    # Nothing random is left, so two correct implementations must agree exactly.
    position_temperature=0.0, class_temperature=0.0,
)


class _DummyTokenizer:
    """One id per character. Enough to compare prefix lengths without weights."""

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(c) % 1000 for c in text]
        if return_tensors == "pt":
            return type("Out", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()
        return type("Out", (), {"input_ids": ids})()


def _item(prefix_len: int, target_len: int, guidance_scale: float = 2.0):
    return CanvasItem(
        prefix_ids=torch.randint(0, 1000, (NUM_CODEBOOK, prefix_len)),
        prefix_audio_mask=torch.zeros(prefix_len, dtype=torch.bool),
        tokens=torch.full((1, NUM_CODEBOOK, target_len), MASK_ID, dtype=torch.long),
        guidance_scale=guidance_scale,
    )


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------


def _slice_logits(canvas, logits: torch.Tensor, item: CanvasItem) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One item's conditional and unconditional target logits, read from the canvas
    layout: every item's conditional block, then the unconditional blocks of the items
    with CFG on (``None`` for an item without)."""
    start, length = item.flat_start, item.target_len
    c_logits = logits[:, :, start : start + length, :]
    if item.flat_u_start < 0:
        return c_logits, None
    u_start = canvas.flat_target_total + item.flat_u_start
    return c_logits, logits[:, :, u_start : u_start + length, :]


def test_packed_layout_doc_boundaries():
    """Two documents per item, laid out cond/uncond in request order."""
    items = [
        _item(prefix_len=40, target_len=100),
        _item(prefix_len=12, target_len=20),
    ]
    canvas = build_packed_canvas(items, torch.device("cpu"))

    assert canvas.doc_lens == [140, 100, 32, 20]
    assert canvas.packed_ids.shape == (1, NUM_CODEBOOK, 292)
    # No padding at all: that is the point of packing over a dense batch.
    assert canvas.packed_ids.shape[-1] == sum(canvas.doc_lens)
    assert canvas.flat_target_total == 120
    assert canvas.tgt_index.numel() == 240


def test_cfg_off_drops_the_unconditional_document():
    """guidance_scale 0 should not pay for logits scoring throws away."""
    with_cfg = _item(prefix_len=40, target_len=100, guidance_scale=2.0)
    without = _item(prefix_len=40, target_len=100, guidance_scale=0.0)

    c_on = build_packed_canvas([with_cfg], torch.device("cpu"))
    c_off = build_packed_canvas([without], torch.device("cpu"))

    assert c_on.doc_lens == [140, 100]
    assert c_off.doc_lens == [140]
    assert c_off.packed_ids.shape[-1] == 140, "no unconditional document"
    assert c_off.tgt_index.numel() == 100, "only the conditional block is gathered"
    assert c_off.flat_uncond_total == 0

    # Mixed batch: the layout is per item, not per step.
    mixed = build_packed_canvas(
        [
            _item(prefix_len=10, target_len=20, guidance_scale=2.0),
            _item(prefix_len=12, target_len=30, guidance_scale=0.0),
        ],
        torch.device("cpu"),
    )
    assert mixed.doc_lens == [30, 20, 42]
    assert mixed.flat_target_total == 50
    assert mixed.flat_uncond_total == 20
    logits = torch.randn(
        1, NUM_CODEBOOK, mixed.flat_target_total + mixed.flat_uncond_total, 1025
    )
    c_x, u_x = _slice_logits(mixed, logits, mixed.items[0])
    c_y, u_y = _slice_logits(mixed, logits, mixed.items[1])
    assert c_x.shape[2] == 20 and u_x is not None and u_x.shape[2] == 20
    assert c_y.shape[2] == 30 and u_y is None

    # The conditional blocks must still be the ones the canvas gathered.
    gathered = mixed.audio_mask[0][mixed.tgt_index]
    assert gathered.all()


def test_packed_layout_positions_restart_per_document():
    """RoPE must not read document n+1 as a continuation of document n."""
    items = [
        _item(prefix_len=5, target_len=7),
        _item(prefix_len=3, target_len=4),
    ]
    canvas = build_packed_canvas(items, torch.device("cpu"))
    pos = canvas.position_ids[0]

    cursor = 0
    for length in canvas.doc_lens:
        assert torch.equal(pos[cursor : cursor + length], torch.arange(length)), (
            "positions must restart at 0 in every packed document"
        )
        cursor += length


def test_packed_layout_target_gather_is_the_canvas():
    """tgt_index must point at the target region, not the prefix."""
    item = _item(prefix_len=9, target_len=6)
    canvas = build_packed_canvas([item], torch.device("cpu"))

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


def _reveal_all(canvas, logits, generator=None, class_temperature=0.0, count=None):
    """One ``unmask_step`` over a one-item canvas, revealing ``count`` cells (all by default)."""
    assert len(canvas.items) == 1
    item = canvas.items[0]
    count = NUM_CODEBOOK * item.target_len if count is None else count
    return unmask_step(
        logits, [item.tokens[0]], canvas.uncond_rows(), [count], [generator],
        [item.guidance_scale], [class_temperature], [0.1], [1.0], MASK_ID,
    )


def test_mask_class_is_never_predicted():
    item = _item(prefix_len=10, target_len=6)
    canvas = build_packed_canvas([item], torch.device("cpu"))
    logits = torch.randn(1, NUM_CODEBOOK, 2 * canvas.flat_target_total, 1025)
    logits[..., MASK_ID] += 100.0  # the MASK class would win every cell's argmax
    assert (_reveal_all(canvas, logits) != MASK_ID).all()


def test_sampling_is_reproducible_under_a_seed():
    """Same seed, same draw; different seed, different draw: the token
    choice and the reveal order both come from the request's generator."""
    item = _item(prefix_len=10, target_len=6)
    canvas = build_packed_canvas([item], torch.device("cpu"))
    logits = torch.randn(1, NUM_CODEBOOK, 2 * canvas.flat_target_total, 1025)

    def draw(seed):
        return _reveal_all(canvas, logits, torch.Generator().manual_seed(seed), class_temperature=1.0, count=5)

    assert torch.equal(draw(1234), draw(1234))
    assert not torch.equal(draw(1234), draw(5678))


# ---------------------------------------------------------------------------
# the reference's per-request scoring and reveal: the oracle for ``unmask_step``
# ---------------------------------------------------------------------------


def _gumbel(logits, temperature, generator=None):
    """``logits / temperature`` plus Gumbel noise; the caller takes the argmax."""
    scaled_logits = logits / temperature
    u = torch.rand(scaled_logits.shape, dtype=scaled_logits.dtype, device=scaled_logits.device, generator=generator)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _predict(c_logits, u_logits, guidance_scale, class_temperature, generator=None):
    """The reference's ``_predict_tokens_with_scoring`` for one request:
    ``(pred_tokens, scores)``, each ``[1, C, T]``."""
    if guidance_scale != 0:
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        u_log_probs = F.log_softmax(u_logits, dim=-1)
        log_probs = torch.log_softmax(c_log_probs + guidance_scale * (c_log_probs - u_log_probs), dim=-1)
    else:
        log_probs = F.log_softmax(c_logits, dim=-1)
    log_probs[..., MASK_ID] = -float("inf")
    if class_temperature > 0.0:
        pred_tokens = _gumbel(filter_top_k(log_probs, ratio=0.1), class_temperature, generator).argmax(dim=-1)
    else:
        pred_tokens = log_probs.argmax(dim=-1)
    return pred_tokens, log_probs.max(dim=-1)[0]


def _reveal(tokens, pred_tokens, scores, reveal_count, layer_penalty_factor, position_temperature, generator=None):
    """The reference's reveal for one request: its ``reveal_count`` best
    masked cells of ``tokens`` (``[1, C, T]``), written in place."""
    if reveal_count <= 0:
        return tokens
    layer_ids = torch.arange(tokens.shape[1], device=tokens.device).view(1, -1, 1)
    scores = scores - (layer_ids * layer_penalty_factor)
    if position_temperature > 0.0:
        scores = _gumbel(scores, position_temperature, generator)
    scores = scores.masked_fill(tokens != MASK_ID, -float("inf"))
    _, topk_idx = torch.topk(scores.flatten(), reveal_count)
    flat_tokens = tokens.flatten()
    flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
    tokens.copy_(flat_tokens.view_as(tokens))
    return tokens


@pytest.mark.parametrize("guidance", [(2.0, 2.0, 2.0), (2.0, 0.0, 3.0)])
@pytest.mark.parametrize("class_temperature", [0.0, 1.0, (0.0, 1.0, 0.0)])
def test_batched_unmask_step_matches_the_per_request_reveal(guidance, class_temperature):
    """``unmask_step`` over a step's requests is the reference's scoring and
    reveal per request (``_predict``, ``_reveal``), with each request's own
    generator, cell for cell; also with CFG off for one request, and with
    sampled tokens, for every request or only some."""
    temperatures = class_temperature if isinstance(class_temperature, tuple) else (class_temperature,) * 3
    gen = torch.Generator().manual_seed(0)
    items = [_item(prefix_len=4, target_len=n, guidance_scale=g) for n, g in zip((9, 5, 12), guidance, strict=True)]
    for item, done in zip(items, (0, 7, 30), strict=True):
        cells = torch.randperm(NUM_CODEBOOK * item.target_len, generator=gen)[:done]
        item.tokens.view(-1)[cells] = torch.randint(0, MASK_ID, (done,), generator=gen)
    canvas = build_packed_canvas(items, torch.device("cpu"))
    rows = canvas.flat_target_total + canvas.flat_uncond_total
    logits = torch.randn(1, NUM_CODEBOOK, rows, MASK_ID + 1, generator=gen).half()
    counts = [4, 0, 11]

    def gens():
        return [torch.Generator().manual_seed(100 + i) for i in range(len(items))]

    want = []
    for item, count, temperature, g in zip(items, counts, temperatures, gens(), strict=True):
        c_logits, u_logits = _slice_logits(canvas, logits, item)
        u_logits = c_logits if u_logits is None else u_logits
        pred, scores = _predict(c_logits.float(), u_logits.float(), item.guidance_scale, temperature, g)
        want.append(_reveal(
            item.tokens.clone(), pred, scores, count, layer_penalty_factor=5.0, position_temperature=5.0, generator=g,
        )[0])
    before = [item.tokens.clone() for item in items]
    got = unmask_step(
        logits, [item.tokens[0] for item in items], canvas.uncond_rows(),
        counts, gens(), [item.guidance_scale for item in items], list(temperatures), [5.0] * 3, [5.0] * 3,
        MASK_ID,
    )
    assert torch.equal(got, torch.cat(want, dim=-1))
    assert all(torch.equal(item.tokens, b) for item, b in zip(items, before, strict=True)), (
        "the step wrote its input canvas"
    )


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def test_denoise_survives_a_reference_that_is_not_encoded_yet():
    """The style span must carry <|denoise|> for a cloning request.

    The data worker builds the prefix before the reference is encoded, so
    ``ref_audio_tokens`` is None there even when cloning. Inferring the flag
    from the tokens dropped the token from every clone.
    """
    from mstar.model.omnivoice.components.text import build_style_text

    assert "<|denoise|>" in build_style_text(
        language="en", instruct=None, denoise=True, has_reference=True
    )
    assert "<|denoise|>" not in build_style_text(
        language="en", instruct=None, denoise=True, has_reference=False
    )

    tokenizer = _DummyTokenizer()
    with_ref, _ = build_prefix(
        tokenizer=tokenizer, text="hello", num_audio_codebook=NUM_CODEBOOK,
        language="en", ref_audio_tokens=None, has_reference=True, denoise=True,
    )
    without_ref, _ = build_prefix(
        tokenizer=tokenizer, text="hello", num_audio_codebook=NUM_CODEBOOK,
        language="en", ref_audio_tokens=None, has_reference=False, denoise=True,
    )
    assert with_ref.shape[-1] > without_ref.shape[-1]


def test_language_names_resolve_to_ids():
    """A name reaching the style span as prose produces noise, not speech."""
    # the reference's submodule: mstar/model/omnivoice also answers to `omnivoice` with mstar/model on sys.path
    pytest.importorskip("omnivoice.models.omnivoice")
    from mstar.model.omnivoice.components.text import resolve_language

    assert resolve_language("Chinese") == "zh"
    assert resolve_language("English") == "en"
    assert resolve_language("zh") == "zh"
    assert resolve_language(None) is None
    assert resolve_language("None") is None
    assert resolve_language("Klingon") is None


def test_prefix_parity():
    """The canvas prefix must match the reference token for token."""
    pytest.importorskip("omnivoice.models.omnivoice")
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


@torch.inference_mode()
def _packed_logits(backbone, canvas):
    """The reference's own packed forward over ``canvas``:
    ``[1, C, 2 * flat_target_total, V]``, the yardstick for mstar's layer loop."""
    from omnivoice.models import omnivoice_flashinfer as fi

    model = backbone.model
    model._fi_runner.plan(canvas.doc_lens, next(model.llm.parameters()).dtype)
    fi._CTX["wrapper"] = model._fi_runner.wrapper
    fi._CTX["pos_ids"] = canvas.position_ids[0].to(torch.int32)
    fi._CTX["doc_slots"] = None
    return fi._forward_logits(
        model, canvas.packed_ids, canvas.audio_mask, canvas.position_ids, canvas.tgt_index,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_encode_head_matches_the_packed_reference():
    pytest.importorskip("omnivoice.models.omnivoice")
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

    from mstar.model.omnivoice.components.backbone import OmniVoiceBackbone

    model = OmniVoice.from_pretrained(MODEL_PATH, dtype=torch.float16).eval().cuda()
    apply_flashinfer(model, enable_cuda_graph=False)
    backbone = OmniVoiceBackbone(model).eval()

    gen = torch.Generator().manual_seed(0)
    items = []
    for prefix_len, target_len, guidance in [(23, 40, 2.0), (31, 25, 0.0), (17, 52, 2.0)]:
        tokens = torch.full((1, NUM_CODEBOOK, target_len), MASK_ID, dtype=torch.long)
        tokens[..., ::3] = torch.randint(0, MASK_ID, tokens[..., ::3].shape, generator=gen)
        items.append(CanvasItem(
            prefix_ids=torch.randint(0, 1000, (NUM_CODEBOOK, prefix_len), generator=gen),
            prefix_audio_mask=torch.zeros(prefix_len, dtype=torch.bool),
            tokens=tokens,
            guidance_scale=guidance,
        ))
    canvas = build_packed_canvas(items, torch.device("cuda"))

    with torch.inference_mode():
        want = _packed_logits(backbone, canvas)
        hidden = backbone.encode(
            canvas.packed_ids[0].t(), canvas.audio_mask[0], canvas.position_ids[0].to(torch.int32),
            attn=model._fi_runner.wrapper,  # planned for this canvas by _packed_logits
        )
        got = backbone.head(hidden, canvas.tgt_index)
    # the residual adds fold into the next RMSNorm (a float sum, not a rounded
    # half-precision one), so the logits agree to rounding, not bit for bit
    assert torch.allclose(got.float(), want.float(), atol=0.25, rtol=0.02)
    assert (got.argmax(-1) == want.argmax(-1)).float().mean() > 0.95


def _run_ours(backbone, config, specs, num_step=None):
    """Drive the port's packed loop over ``specs``; returns each finished canvas.

    ``specs`` is a list of ``(text, language, target_len)``. ``num_step``
    overrides the greedy default, so a caller can stop after one step and
    compare before greedy reveals start compounding.
    """
    num_step = GREEDY["num_step"] if num_step is None else num_step
    items, schedules = [], []
    for text, language, target_len in specs:
        prefix_ids, prefix_audio_mask = build_prefix(
            tokenizer=backbone.model.text_tokenizer, text=text,
            num_audio_codebook=config.num_audio_codebook,
            language=language, denoise=True,
        )
        items.append(CanvasItem(
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
            num_step=num_step, t_shift=GREEDY["t_shift"],
        ))

    knobs = [[GREEDY[name]] * len(items) for name in (
        "guidance_scale", "class_temperature", "layer_penalty_factor", "position_temperature",
    )]
    for k in range(num_step):
        canvas = build_packed_canvas(items, torch.device("cuda"))
        # the served step: one batched scoring and reveal over the packed logits
        tokens = unmask_step(
            _packed_logits(backbone, canvas), [item.tokens[0] for item in items],
            canvas.uncond_rows(),
            [schedule[k] for schedule in schedules], [None] * len(items), *knobs, config.audio_mask_id,
        )
        for item in items:
            item.tokens = tokens[:, item.flat_start:item.flat_start + item.target_len].unsqueeze(0)
    return [item.tokens[0].cpu() for item in items]


def _step0_logits(backbone, config, specs, which=0):
    """``(cond, uncond)`` logits for one request on a fresh all-MASK canvas.

    One forward, before any reveal, so nothing downstream can amplify a
    difference into something that looks bigger than it is.
    """
    items = []
    for text, language, target_len in specs:
        prefix_ids, prefix_audio_mask = build_prefix(
            tokenizer=backbone.model.text_tokenizer, text=text,
            num_audio_codebook=config.num_audio_codebook,
            language=language, denoise=True,
        )
        items.append(CanvasItem(
            prefix_ids=prefix_ids.cuda(),
            prefix_audio_mask=prefix_audio_mask.cuda(),
            tokens=torch.full(
                (1, config.num_audio_codebook, target_len),
                config.audio_mask_id, dtype=torch.long, device="cuda",
            ),
            guidance_scale=GREEDY["guidance_scale"],
        ))
    canvas = build_packed_canvas(items, torch.device("cuda"))
    logits = _packed_logits(backbone, canvas).to(torch.float32)
    return _slice_logits(canvas, logits, items[which])


def _run_reference(model, specs, num_step=None):
    from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig

    knobs = dict(GREEDY)
    if num_step is not None:
        knobs["num_step"] = num_step

    task = GenerationTask(
        batch_size=len(specs), texts=[s[0] for s in specs],
        target_lens=[s[2] for s in specs], langs=[s[1] for s in specs],
        instructs=[None] * len(specs), ref_texts=[None] * len(specs),
        ref_audio_tokens=[None] * len(specs), ref_rms=[None] * len(specs),
    )
    out = model._generate_iterative(
        task, OmniVoiceGenerationConfig(**knobs, denoise=True)
    )
    return [t.cpu() for t in out]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_packed_parity_single():
    """Tier A: same kernels, one request, token-exact."""
    pytest.importorskip("omnivoice.models.omnivoice")
    from mstar.model.omnivoice.components.backbone import OmniVoiceBackbone

    config = OmniVoiceConfig()
    model = _load_patched()
    specs = [("Xin chào, đây là một câu thử.", "Vietnamese", 60)]

    theirs = _run_reference(model, specs)[0]
    ours = _run_ours(OmniVoiceBackbone(model).eval(), config, specs)[0]

    assert (ours != config.audio_mask_id).all(), "canvas left masked cells"
    assert torch.equal(ours, theirs), (
        f"{int((ours != theirs).sum())}/{ours.numel()} cells differ from the "
        "reference's packed path, which runs the identical kernels"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_packed_parity_batched():
    """Tier B: packing several requests must not leak between them.

    Cross-request packing is the thing the reference never does, so this is
    where a boundary bug would live -- a document attending past its own edge,
    or a position id continuing from the previous request.

    Measured on the step-0 logits, not on the finished canvas. Changing the
    number of documents changes how the ragged kernel splits its reductions,
    so packed and solo logits differ by about one fp16 ulp; over eight greedy
    reveals that lands on a different canvas (measured: 0 cells differ for
    four steps, then 2, 22, 69, 274). Asserting equality on the canvas would
    be asserting bit-exactness across batch shapes, which no GPU kernel
    offers. A leak is a different signature entirely -- it scales with the
    neighbour's content -- so this checks the logits are within tolerance and
    that WHO the neighbour is does not matter.
    """
    pytest.importorskip("omnivoice.models.omnivoice")
    from mstar.model.omnivoice.components.backbone import OmniVoiceBackbone

    config = OmniVoiceConfig()
    backbone = OmniVoiceBackbone(_load_patched()).eval()

    a = ("Xin chào, đây là một câu thử.", "Vietnamese", 60)
    b = ("Hello world.", "English", 25)
    c = ("今天天气很好，我们出去走走吧。", "Chinese", 90)

    solo = _step0_logits(backbone, config, [a])
    neighbours = {
        "a copy of itself": [a, a],
        "one other sentence": [a, b],
        "two other sentences": [a, b, c],
    }
    packed = {k: _step0_logits(backbone, config, v) for k, v in neighbours.items()}

    for label, (c_logits, u_logits) in packed.items():
        for half, got, want in (("cond", c_logits, solo[0]), ("uncond", u_logits, solo[1])):
            gap = (got - want).abs().max().item()
            assert gap <= 0.5, (
                f"packed with {label}, {half} logits moved by {gap:.4f}; "
                "an fp16 ulp at this scale is 0.125, so this is a leak, "
                "not reduction order"
            )
            agree = float((got.argmax(-1) == want.argmax(-1)).float().mean())
            assert agree >= 0.95, (
                f"packed with {label}, {half} argmax agrees only {agree:.3f}"
            )

    # The real leak detector: a neighbour's *content* must not reach us. If it
    # did, packing with a copy of ourselves and packing with Chinese text would
    # perturb us differently.
    ref = packed["a copy of itself"][0]
    for label in ("one other sentence", "two other sentences"):
        assert torch.equal(packed[label][0], ref), (
            f"request 0 depends on who it is packed with: {label} differs from "
            "a copy of itself, which means content crossed a document boundary"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_dense_agreement():
    """Tier C: against the unoptimised dense path, an agreement rate.

    Fused QKV, flashinfer RoPE and the ragged kernel are mathematically the
    same as the dense path but not bit-identical, and a greedy argmax turns a
    last-bit difference into a different token.  So this measures agreement
    rather than asserting equality -- the reference's own fast mode would fail
    an equality check here too.

    Measured after a single unmask step, deliberately. Over the full eight,
    each flipped cell changes the canvas the next step reads, so the rate
    compounds and lands near 0.77 for two implementations that agree to one
    ulp everywhere. That number would say nothing about porting accuracy; the
    first step, where both sides read an identical all-MASK canvas, says
    everything.

    The 0.90 floor is measured, not guessed. The scored distribution on this
    checkpoint is very flat: the median top1-top2 margin is 0.25 and 76% of
    cells sit below 0.5, while one fp16 ulp at the logit scale here
    (max |logit| ~ 137) is about 0.134. Injecting exactly one ulp of noise
    into the logits drops argmax agreement to 0.70 by itself, so two kernels
    that are numerically equivalent cannot be held to 0.98. The port measures
    0.927, well clear of that noise floor; a porting bug would land at or
    below it.
    """
    pytest.importorskip("omnivoice.models.omnivoice")
    from omnivoice.models.omnivoice import OmniVoice

    from mstar.model.omnivoice.components.backbone import OmniVoiceBackbone

    config = OmniVoiceConfig()
    specs = [("Xin chào, đây là một câu thử.", "Vietnamese", 60)]

    dense = OmniVoice.from_pretrained(MODEL_PATH, dtype=torch.float16).eval().cuda()
    theirs = _run_reference(dense, specs, num_step=1)[0]
    del dense
    torch.cuda.empty_cache()

    backbone = OmniVoiceBackbone(_load_patched()).eval()
    ours = _run_ours(backbone, config, specs, num_step=1)[0]

    agreement = float((ours == theirs).float().mean())
    assert agreement >= 0.90, (
        f"after one step only {agreement:.3f} of cells agree with the dense "
        "reference; one ulp of fp16 noise alone costs ~0.30 here, so below "
        "0.90 is a porting bug rather than kernel noise"
    )


# ---------------------------------------------------------------------------
# the unmask step
# ---------------------------------------------------------------------------


def _fwd_info(request_id: str, iter_idx: int):
    info = CurrentForwardPassInfo(
        request_id=request_id, graph_walk="unmask", fwd_index=0,
        random_seed=0, max_tokens=0,
    )
    info.step_metadata = dict(GREEDY)
    info.dynamic_loop_iter_counts = {UNMASK_LOOP_NAME: iter_idx}
    return info


def test_unmask_step_reveals_without_touching_the_loop_edge():
    """The forward reveals into a new canvas, and the edge it reads stays intact.

    The canvas the engine routes in is the previous iteration's output edge;
    if the step wrote it in place, the engine's own copy of that step's state
    would change under it.
    """
    from types import SimpleNamespace

    target_len, vocab = 16, MASK_ID + 1
    sub = OmniVoiceBackboneSubmodule(backbone=None, config=OmniVoiceConfig())
    sub.get_device = lambda: torch.device("cpu")
    edge = torch.full((NUM_CODEBOOK, target_len), MASK_ID, dtype=torch.long)
    info = _fwd_info("a", 1)
    info.step_metadata["target_len"] = target_len
    row = sub.prepare_inputs("unmask", info, {
        "prefix_ids": [torch.zeros(NUM_CODEBOOK, 3, dtype=torch.long)],
        "prefix_audio_mask": [torch.zeros(3, dtype=torch.bool)],
        "audio_tokens": [edge],
        "step_index": [torch.ones(1, dtype=torch.int64)],
    })
    engine_inputs = SimpleNamespace(request_ids=[0], per_request_info={0: info})
    pre = sub.preprocess("unmask", engine_inputs, [row])

    # Logits that make the argmax a fixed, non-mask token everywhere.
    logits = torch.zeros(1, NUM_CODEBOOK, len(pre["canvas"].tgt_index), vocab)
    logits[..., 7] = 10.0
    sub._logits = lambda _inputs, _canvas: logits
    outputs = sub.forward_batched("unmask", engine_inputs, **pre)[0]

    assert bool((edge == MASK_ID).all()), "the unmask step wrote through the input edge"
    tokens = outputs["audio_tokens"][0]
    assert tokens.shape == (NUM_CODEBOOK, target_len)
    assert int(outputs["step_index"][0].reshape(-1)[0]) == 2
    assert set(tokens[tokens != MASK_ID].tolist()) == {7}

    revealed = int((tokens != MASK_ID).sum())
    expected = build_reveal_schedule(
        target_len=target_len, num_codebook=NUM_CODEBOOK,
        num_step=GREEDY["num_step"], t_shift=GREEDY["t_shift"],
    )[1]
    assert revealed == expected, f"revealed {revealed}, schedule says {expected}"

