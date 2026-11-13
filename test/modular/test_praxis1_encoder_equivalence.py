"""Numerical equivalence of the Praxis-1 encoder tower with the reference.

Companion to ``test_praxis1_reference_equivalence`` and its siblings: the
comparison is against the imported reference rather than against a second
copy of our own reading of it, and every comparison is bit-exact.

The reference is a checkout rather than a published package, so these tests
skip unless ``praxis`` is importable. Put the clone on ``PYTHONPATH``::

    PYTHONPATH=/path/to/praxis-1-oss pytest test/modular/test_praxis1_encoder_equivalence.py

The reference tower runs end to end under M*'s CPU-only torch, flex attention
and all, so these are whole-tower comparisons rather than layer-by-layer
ones. The attention core is taken *from the reference* (its compiled
``flex_attention`` over the reference's own ``BlockMask``) and threaded in as
the port's ``attend`` callable. That is deliberate: the engine kernel that
will serve this does not exist yet, so a comparison that also swapped the
attention core would be measuring two unported things at once. What is under
test here is the composition: the projections, the timestep modulation, the
packing order, the per-block scale, the dense prefix and the taps.

The negative controls carry the weight. Every one of the tower's conventions
produces a running model with plausible activations when inverted, so parity
alone cannot say we read the right one. Each control below inverts exactly
one convention and asserts the result is a different number.

Two measured notes, recorded as tests rather than as comments:

* the ``+ 0.0`` of ``positive_zero_residual_add`` turns out to be
  **unobservable in this slice**. Nothing reads the stack output directly:
  it goes through a linear projection, and a sum cannot see the sign of a
  zero addend. The semantics it exists for are asserted directly, and the
  flag's inertness on the tower output is recorded so nobody has to rerun
  the experiment,

* the dense boolean mask with plain SDPA, which is the tower's default when
  no mask and no ``attend`` are supplied, does **not** reproduce the
  reference's flex attention bit for bit. The difference is measured in a
  test rather than hidden behind a tolerance.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip(
    "praxis",
    reason="the Praxis-1 reference checkout is not on PYTHONPATH",
)

import praxis  # noqa: E402
from praxis import backbone as ref_backbone  # noqa: E402
from praxis import layers as ref_layers  # noqa: E402

from mstar.model.praxis1.components import encoder as port_enc  # noqa: E402

# Small, but every shape the tower cares about stays distinct: the head
# dimension does not divide the model width evenly into one head, there are
# fewer key/value heads than query heads, the dense hidden width is not the
# MoE one, and the anchor grid is not the video grid.
DIM, HEAD_DIM, NUM_KV_HEADS = 64, 32, 1
DEPTH, NUM_DENSE = 4, 2
DENSE_HIDDEN, MOE_HIDDEN = 32, 16
NUM_EXPERTS, TOP_K, SCALE = 8, 4, 2.5
TEXT_DIM, VIDEO_DIM, TIME_DIM = 16, 8, 8
NUM_REGISTERS, NUM_TEXT = 4, 5
ANCHOR_HW = (2, 2)
FRAMES_PER_CHUNK, HEIGHT, WIDTH = 2, 2, 3
# Block 3 is the last block and block 0 and 2 are taps, so every case of
# may_defer is exercised and only block 1 is ever deferred.
TAP_BLOCKS = (0, 2, 3)
MASK_BLOCK_SIZE = 16

# Flex attention on CPU is a compiled kernel; keep the parametrization small.
DTYPES = [torch.float32, torch.bfloat16]
CHUNK_COUNTS = [1, 2, 3]


@pytest.fixture(autouse=True)
def _one_thread():
    """Pin the intra-op pool.

    Parity is safe either way, since both sides run in this process under the
    same pool. The controls are not: they assert that a reassociated float32
    sum does change, which is a property of the reduction torch picks, and
    that depends on the thread count.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _static_shapes():
    """Compile a static kernel per sequence length.

    These tests run several window sizes in one process. On the second one,
    dynamo would mark the sequence length dynamic and recompile, and the
    dynamic CPU codegen of ``flex_attention`` does not compile at all in
    torch 2.12 (``'cur_qSplitSize8' was not declared in this scope``). That is
    an inductor bug on a path the served model never takes, so the tests pin
    static shapes and raise the recompile limit instead of working around it:
    falling back to eager flex attention would silently compare against a
    different kernel from the reference's.
    """
    import torch._dynamo.config as dynamo_config

    previous = (dynamo_config.automatic_dynamic_shapes, dynamo_config.recompile_limit)
    dynamo_config.automatic_dynamic_shapes = False
    dynamo_config.recompile_limit = max(previous[1], 64)
    yield
    dynamo_config.automatic_dynamic_shapes, dynamo_config.recompile_limit = previous


def _fill(module: torch.nn.Module, seed: int) -> None:
    """Deterministic values for every parameter and for the Fourier buffers.

    The timestep embedding's ``freqs`` and ``phases`` are checkpoint buffers
    that initialize to zero, which makes the embedding the constant 0.5 and
    every chunk's modulation identical. Filling them is what makes the
    per-chunk timestep visible at all.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
        for name, buf in module.named_buffers():
            if name.endswith(("freqs", "phases")):
                buf.copy_(torch.randn(buf.shape, generator=gen))


def _ref_core(depth: int = DEPTH, num_dense: int = NUM_DENSE) -> torch.nn.Module:
    return ref_layers.Transformer(
        depth=depth,
        num_dense=num_dense,
        dim=DIM,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        dense_hidden=DENSE_HIDDEN,
        moe_hidden=MOE_HIDDEN,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        routed_scaling_factor=SCALE,
        mod_dim=None,
        rope_effective_dim=HEAD_DIM // 2,
        attention_backend="triton",
    )


def _port_core(depth: int = DEPTH, num_dense: int = NUM_DENSE) -> port_enc.TrunkStack:
    return port_enc.TrunkStack(
        depth=depth,
        num_dense=num_dense,
        dim=DIM,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        dense_hidden=DENSE_HIDDEN,
        moe_hidden=MOE_HIDDEN,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        routed_scaling_factor=SCALE,
        rope_effective_dim=HEAD_DIM // 2,
    )


def _dense_remap(state: dict, num_dense: int = NUM_DENSE, prefix: str = "") -> dict:
    """The reference's dense feed-forward is an ``nn.Sequential``.

    It therefore stores index names (``ffn.0.linear.a_weight``,
    ``ffn.1.weight``) where the port stores names. This is the only remap
    between the two state dicts: if anything else needs one, a key drifted.
    """
    out = {}
    for key, value in state.items():
        for i in range(num_dense):
            head = f"{prefix}blocks.{i}.ffn."
            if key.startswith(head):
                tail = key[len(head) :]
                tail = tail.replace("0.linear.", "linear.").replace("1.weight", "down.weight")
                key = head + tail
                break
        out[key] = value
    return out


def _tower_pair(dtype: torch.dtype = torch.float32, seed: int = 0, tap_blocks=TAP_BLOCKS):
    """A reference encoder tower and our port, holding the same weights."""
    ref = ref_backbone.EncoderTower(
        core=_ref_core(),
        dim=DIM,
        text_dim=TEXT_DIM,
        video_dim=VIDEO_DIM,
        time_dim=TIME_DIM,
        num_registers=NUM_REGISTERS,
        tap_blocks=tap_blocks,
        positive_zero_residual_add=True,
        mask_block_size=MASK_BLOCK_SIZE,
    )
    _fill(ref, seed)
    port = port_enc.EncoderTower(
        core=_port_core(),
        dim=DIM,
        text_dim=TEXT_DIM,
        video_dim=VIDEO_DIM,
        time_dim=TIME_DIM,
        num_registers=NUM_REGISTERS,
        tap_blocks=tap_blocks,
        positive_zero_residual_add=True,
    )
    port.load_state_dict(_dense_remap(ref.state_dict(), prefix="core."))
    return ref.to(dtype).eval(), port.to(dtype).eval()


def _window(n_chunks: int, dtype: torch.dtype, seed: int = 3):
    """One chunk-causal window: ``(text, anchor, clean, noised, timesteps)``."""
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return (torch.randn(shape, generator=gen) * 0.5).to(dtype)

    frames = n_chunks * FRAMES_PER_CHUNK
    clean_frames = (n_chunks - 1) * FRAMES_PER_CHUNK
    text = rand(1, NUM_TEXT, TEXT_DIM)
    anchor = rand(1, VIDEO_DIM, 1, *ANCHOR_HW)
    noised = rand(1, VIDEO_DIM, frames, HEIGHT, WIDTH)
    clean = rand(1, VIDEO_DIM, clean_frames, HEIGHT, WIDTH) if clean_frames else None
    timesteps = torch.rand(1, n_chunks, generator=gen).to(dtype)
    return text, anchor, clean, noised, timesteps


def _ref_mask(n_chunks: int, core: torch.nn.Module):
    """The reference's own ``BlockMask`` for this window layout.

    Built with the slots the tower builds, so this is the mask the reference
    call uses, not a second reading of the rule. ``full_blocks`` is False on
    CPU because the effective backend is the Triton one, and that is asserted
    rather than assumed.
    """
    device = torch.device("cpu")
    assert ref_backbone._full_blocks(core, device, MASK_BLOCK_SIZE) is False
    frames = n_chunks * FRAMES_PER_CHUNK
    clean_frames = (n_chunks - 1) * FRAMES_PER_CHUNK
    tpf = HEIGHT * WIDTH
    slots = [
        ("global", 1, NUM_REGISTERS),
        ("global", 1, ANCHOR_HW[0] * ANCHOR_HW[1]),
        ("global", 1, NUM_TEXT),
        ("noised", frames, tpf),
    ]
    if clean_frames:
        slots.append(("clean", clean_frames, tpf))
    return ref_backbone.teacher_forcing_mask(
        slots, FRAMES_PER_CHUNK, True, MASK_BLOCK_SIZE, device, False
    )


def _flex_attend(q, k, v, mask):
    """The reference's attention core, as the port's ``attend`` callable.

    Exactly the call ``praxis.layers.Attention`` makes for the Triton
    backend, including ``kernel_options=None``: a different option set is a
    different kernel and a different set of roundings.
    """
    return ref_layers._compiled_flex_attention(
        q,
        k,
        v,
        block_mask=mask,
        scale=None,
        enable_gqa=k.shape[1] != q.shape[1],
        kernel_options=None,
    )


def _run_pair(n_chunks: int, dtype: torch.dtype, **port_kwargs):
    """Run both towers over one window and return ``(ref_out, port_out)``."""
    ref, port = _tower_pair(dtype)
    text, anchor, clean, noised, timesteps = _window(n_chunks, dtype)
    mask = _ref_mask(n_chunks, ref.core)
    with torch.no_grad():
        got_ref = ref.forward_chunk_causal(text, anchor, clean, noised, timesteps)
        got_port = port.forward_chunk_causal(
            text, anchor, clean, noised, timesteps, mask=mask, attend=_flex_attend, **port_kwargs
        )
    return got_ref, got_port, port, (text, anchor, clean, noised, timesteps), mask


# ---------------------------------------------------------------------------
# Whole-tower parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_chunks", CHUNK_COUNTS)
def test_tower_cond_matches_reference(dtype, n_chunks):
    (ref_cond, _), (port_cond, _), *_ = _run_pair(n_chunks, dtype)
    assert port_cond.shape == ref_cond.shape
    assert torch.equal(port_cond, ref_cond)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_chunks", CHUNK_COUNTS)
def test_tower_taps_match_reference(dtype, n_chunks):
    (_, ref_feats), (_, port_feats), *_ = _run_pair(n_chunks, dtype)
    assert len(port_feats.taps) == len(TAP_BLOCKS)
    for i, (got, want) in enumerate(zip(port_feats.taps, ref_feats.taps, strict=True)):
        assert torch.equal(got, want), f"tap {i} (block {TAP_BLOCKS[i]}) differs"


@pytest.mark.parametrize("n_chunks", CHUNK_COUNTS)
def test_tower_features_match_reference(n_chunks):
    """Everything in ``TapFeatures`` that is not a tap."""
    (_, ref_feats), (_, port_feats), *_ = _run_pair(n_chunks, torch.float32)
    assert torch.equal(port_feats.embeddings, ref_feats.embeddings)
    assert torch.equal(port_feats.rope_ids, ref_feats.rope_ids)
    assert port_feats.rope_ids.dtype == torch.int64
    assert port_feats.global_spans == ref_feats.global_spans
    assert port_feats.noised_span == ref_feats.noised_span
    assert port_feats.clean_span == ref_feats.clean_span
    assert port_feats.n_chunks == ref_feats.n_chunks
    assert port_feats.frames_per_chunk == ref_feats.frames_per_chunk
    assert port_feats.tokens_per_frame == ref_feats.tokens_per_frame


def test_tower_taps_follow_tap_block_order_not_sorted_order():
    """``tap_blocks`` is read positionally, so its order is the taps' order."""
    dtype = torch.float32
    order = (3, 0, 2)
    ref, port = _tower_pair(dtype, tap_blocks=order)
    _, sorted_port = _tower_pair(dtype, tap_blocks=tuple(sorted(order)))
    text, anchor, clean, noised, timesteps = _window(2, dtype)
    mask = _ref_mask(2, ref.core)
    with torch.no_grad():
        _, feats = port.forward_chunk_causal(
            text, anchor, clean, noised, timesteps, mask=mask, attend=_flex_attend
        )
        _, ref_feats = ref.forward_chunk_causal(text, anchor, clean, noised, timesteps)
        _, sorted_feats = sorted_port.forward_chunk_causal(
            text, anchor, clean, noised, timesteps, mask=mask, attend=_flex_attend
        )
    for got, want in zip(feats.taps, ref_feats.taps, strict=True):
        assert torch.equal(got, want)
    # The same three tensors, permuted: not a vacuous order assertion.
    assert torch.equal(feats.taps[0], sorted_feats.taps[2])
    assert not torch.equal(feats.taps[0], sorted_feats.taps[0])


def test_stack_matches_reference_transformer_directly():
    """The stack on its own, without the tower's packing around it."""
    torch.manual_seed(0)
    ref, port = _ref_core(), _port_core()
    _fill(ref, 11)
    port.load_state_dict(_dense_remap(ref.state_dict()))
    ref.eval()
    port.eval()
    positions = port_enc.chunk_causal_positions(
        NUM_REGISTERS, ANCHOR_HW, NUM_TEXT, (4, HEIGHT, WIDTH), 2, (0, 0, 0)
    ).unsqueeze(0)
    seq = positions.shape[1]
    x = torch.randn(1, seq, DIM) * 0.3
    mask = _ref_mask(2, ref)
    with torch.no_grad():
        want_out, want_taps = ref(x, positions, None, mask, TAP_BLOCKS)
        got_out, got_taps = port(x, positions, mask, TAP_BLOCKS, _flex_attend)
    assert torch.equal(got_out, want_out)
    for got, want in zip(got_taps, want_taps, strict=True):
        assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# The pieces: projections, positions, modulation
# ---------------------------------------------------------------------------


def test_text_projection_matches_reference():
    torch.manual_seed(0)
    ref = ref_backbone.TextProjection(TEXT_DIM, DIM)
    _fill(ref, 1)
    port = port_enc.TextProjection(TEXT_DIM, DIM)
    port.load_state_dict(ref.state_dict())
    x = torch.randn(1, NUM_TEXT, TEXT_DIM)
    with torch.no_grad():
        assert torch.equal(port(x), ref(x))


def test_video_input_projection_matches_reference():
    torch.manual_seed(0)
    ref = ref_backbone.VideoInputProjection(VIDEO_DIM, DIM)
    _fill(ref, 2)
    port = port_enc.VideoInputProjection(VIDEO_DIM, DIM)
    port.load_state_dict(ref.state_dict())
    v = torch.randn(1, VIDEO_DIM, 3, HEIGHT, WIDTH)
    with torch.no_grad():
        got, want = port(v), ref(v)
    assert got.shape == (1, 3, HEIGHT, WIDTH, DIM)
    assert torch.equal(got, want)


def test_encoder_output_projection_matches_reference():
    torch.manual_seed(0)
    ref = ref_backbone.EncoderOutputProjection(DIM)
    _fill(ref, 3)
    port = port_enc.EncoderOutputProjection(DIM)
    port.load_state_dict(ref.state_dict())
    x = torch.randn(1, 11, DIM)
    with torch.no_grad():
        assert torch.equal(port(x), ref(x))


@pytest.mark.parametrize("n_chunks", CHUNK_COUNTS)
@pytest.mark.parametrize("shifts", [(0, 0, 0), (3, 5, 7)])
def test_chunk_causal_positions_match_reference(n_chunks, shifts):
    frames = n_chunks * FRAMES_PER_CHUNK
    clean_frames = (n_chunks - 1) * FRAMES_PER_CHUNK
    args = (NUM_REGISTERS, ANCHOR_HW, NUM_TEXT, (frames, HEIGHT, WIDTH), clean_frames, shifts)
    assert torch.equal(
        port_enc.chunk_causal_positions(*args), ref_backbone.chunk_causal_positions(*args)
    )


def test_clean_frames_share_the_noised_frames_positions():
    """Stated in the docstring, read off the tensor: the mask is the only
    thing that separates a clean token from its noised counterpart."""
    frames = 3 * FRAMES_PER_CHUNK
    clean_frames = 2 * FRAMES_PER_CHUNK
    pos = port_enc.chunk_causal_positions(
        NUM_REGISTERS, ANCHOR_HW, NUM_TEXT, (frames, HEIGHT, WIDTH), clean_frames, (0, 0, 0)
    )
    n_global = NUM_REGISTERS + ANCHOR_HW[0] * ANCHOR_HW[1] + NUM_TEXT
    tpf = HEIGHT * WIDTH
    noised = pos[n_global : n_global + frames * tpf]
    clean = pos[n_global + frames * tpf :]
    assert torch.equal(clean, noised[: clean_frames * tpf])


@pytest.mark.parametrize("dtype", DTYPES)
def test_modulate_matches_reference(dtype):
    gen = torch.Generator().manual_seed(5)
    x = (torch.randn(1, 9, DIM, generator=gen)).to(dtype)
    shift = (torch.randn(1, 9, DIM, generator=gen)).to(dtype)
    scale = (torch.randn(1, 9, DIM, generator=gen)).to(dtype)
    want = ref_backbone.modulate(x, shift, scale)
    assert torch.equal(port_enc.modulate(x, shift, scale), want)


def test_chunk_rows_matches_reference():
    gen = torch.Generator().manual_seed(6)
    rows = torch.randn(1, 3, DIM, generator=gen)
    counts = [4, 1, 7]
    assert torch.equal(
        port_enc.chunk_rows(rows, counts), ref_backbone._chunk_rows(rows, counts)
    )


def test_timestep_modulation_is_only_on_the_video_tokens():
    """Registers, anchor and text go into the stack unmodulated.

    Read off the packed sequence: the register rows of ``embeddings`` are the
    register parameter itself, bit for bit.
    """
    dtype = torch.float32
    _, port = _tower_pair(dtype)
    text, anchor, clean, noised, timesteps = _window(2, dtype)
    mask = _ref_mask(2, port.core)
    with torch.no_grad():
        _, feats = port.forward_chunk_causal(
            text, anchor, clean, noised, timesteps, mask=mask, attend=_flex_attend
        )
        plain_text = port.input_projections[port_enc.TEXT](text)
        plain_anchor = port.input_projections[port_enc.VIDEO](anchor).flatten(1, -2)
    reg = feats.global_spans["register"]
    anc = feats.global_spans["anchor"]
    txt = feats.global_spans["text"]
    assert torch.equal(feats.embeddings[:, reg[0] : reg[0] + reg[1]], port.registers)
    assert torch.equal(feats.embeddings[:, anc[0] : anc[0] + anc[1]], plain_anchor)
    assert torch.equal(feats.embeddings[:, txt[0] : txt[0] + txt[1]], plain_text)
    # And the noised tokens are NOT the plain projection, or the assertions
    # above would hold whether or not anything is modulated.
    with torch.no_grad():
        plain_noised = port.input_projections[port_enc.VIDEO](noised).flatten(1, -2)
    start, length = feats.noised_span
    assert not torch.equal(feats.embeddings[:, start : start + length], plain_noised)


# ---------------------------------------------------------------------------
# The layer scale
# ---------------------------------------------------------------------------


def test_layer_scale_is_one_over_sqrt_index_plus_one():
    core = _port_core(depth=DEPTH)
    for i, block in enumerate(core.blocks):
        # The buffer is float32, so the comparison rounds the same way.
        assert torch.equal(block.s, torch.tensor(1 / math.sqrt(i + 1), dtype=torch.float32))
    assert float(core.blocks[0].s) == 1.0
    ref = _ref_core(depth=DEPTH)
    for got, want in zip(core.blocks, ref.blocks, strict=True):
        assert torch.equal(got.s, want.s)


def test_layer_scale_is_a_persistent_buffer():
    """The checkpoint's bf16-rounded value has to win over the float32 one.

    A non-persistent buffer is absent from the state dict, so loading leaves
    the exact float32 value in place and every block below the first runs at
    a slightly different scale.
    """
    core = _port_core(depth=DEPTH)
    assert "blocks.1.s" in core.state_dict()
    rounded = torch.tensor(1 / math.sqrt(2), dtype=torch.bfloat16).to(torch.float32)
    state = core.state_dict()
    state["blocks.1.s"] = rounded
    core.load_state_dict(state)
    assert torch.equal(core.blocks[1].s, rounded)
    assert not torch.equal(rounded, torch.tensor(1 / math.sqrt(2)))


def test_bf16_rounded_layer_scale_still_matches_the_reference():
    """Parity under the scale the checkpoint actually carries."""
    dtype = torch.float32
    ref, port = _tower_pair(dtype)
    with torch.no_grad():
        for block in list(ref.core.blocks) + list(port.core.blocks):
            block.s.copy_(block.s.to(torch.bfloat16).to(block.s.dtype))
    text, anchor, clean, noised, timesteps = _window(2, dtype)
    mask = _ref_mask(2, ref.core)
    with torch.no_grad():
        want, want_feats = ref.forward_chunk_causal(text, anchor, clean, noised, timesteps)
        got, got_feats = port.forward_chunk_causal(
            text, anchor, clean, noised, timesteps, mask=mask, attend=_flex_attend
        )
    assert torch.equal(got, want)
    for a, b in zip(got_feats.taps, want_feats.taps, strict=True):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# The deferred residual add
# ---------------------------------------------------------------------------


def test_may_defer_is_suppressed_at_taps_and_at_the_last_block():
    last = DEPTH - 1
    allowed = [i for i in range(DEPTH) if port_enc.may_defer(i, last, TAP_BLOCKS)]
    assert allowed == [1]
    for i in TAP_BLOCKS:
        assert not port_enc.may_defer(i, last, TAP_BLOCKS)
    assert not port_enc.may_defer(last, last, ())


def test_fusing_the_residual_add_changes_nothing():
    """Deferring is a refactor in eager mode, which is why it is safe to run.

    What it changes is which intermediate tensors exist, not their values, so
    the rule it has to obey is about the taps rather than about arithmetic.
    """
    (ref_cond, ref_feats), (port_cond, _), *_ = _run_pair(2, torch.float32)
    (_, _), (fused_cond, fused_feats), *_ = _run_pair(2, torch.float32, fuse_residual=True)
    assert torch.equal(fused_cond, ref_cond) and torch.equal(fused_cond, port_cond)
    for a, b in zip(fused_feats.taps, ref_feats.taps, strict=True):
        assert torch.equal(a, b)


def test_control_deferring_at_a_tap_block_breaks_the_tap(monkeypatch):
    """Control: the taps are the output AFTER the second residual add.

    With deferring allowed everywhere, a tap is the block's residual stream
    before its feed-forward output is added back. It has the same shape and
    a similar magnitude, and it is a tensor one feed-forward short of the
    block's output.
    """
    (_, want_feats), (_, _), *_ = _run_pair(2, torch.float32)
    monkeypatch.setattr(port_enc, "may_defer", lambda index, last, tap_blocks: True)
    (_, _), (_, bad_feats), *_ = _run_pair(2, torch.float32, fuse_residual=True)
    for i, (bad, want) in enumerate(zip(bad_feats.taps, want_feats.taps, strict=True)):
        assert bad.shape == want.shape
        assert not torch.equal(bad, want), f"tap {i} survived being deferred"


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def test_control_layer_scale_of_one_differs():
    """Control: the ``1 / sqrt(i + 1)`` has to be doing something.

    Block 0's scale is 1 already, so only blocks 1 and up move. If this
    passed, the scale could be dropped and every parity test above would
    still hold.
    """
    dtype = torch.float32
    ref, port = _tower_pair(dtype)
    text, anchor, clean, noised, timesteps = _window(2, dtype)
    mask = _ref_mask(2, ref.core)
    with torch.no_grad():
        want, _ = ref.forward_chunk_causal(text, anchor, clean, noised, timesteps)
        for block in port.core.blocks:
            block.s.fill_(1.0)
        got, _ = port.forward_chunk_causal(
            text, anchor, clean, noised, timesteps, mask=mask, attend=_flex_attend
        )
    assert got.shape == want.shape
    assert not torch.equal(got, want)


def test_control_embeddings_are_not_the_first_tap():
    """Control: ``embeddings`` is block 0's INPUT, and block 0 is also a tap.

    Both tensors are ``(1, S, dim)`` and both exist, so handing the expert
    the wrong one costs nothing but a whole block. The relationship is pinned
    from the other side too: running block 0 on the embeddings reproduces the
    first tap exactly.
    """
    dtype = torch.float32
    (_, _), (_, feats), port, _, mask = _run_pair(2, dtype)
    assert TAP_BLOCKS[0] == 0
    assert feats.embeddings.shape == feats.taps[0].shape
    assert not torch.equal(feats.embeddings, feats.taps[0])
    with torch.no_grad():
        rope = port.core.rope_table(feats.rope_ids, feats.embeddings.size(-2))
        again = port.core.blocks[0](feats.embeddings, rope, mask, _flex_attend)
    assert torch.equal(again, feats.taps[0])


def test_control_clean_tokens_are_modulated_at_timestep_zero():
    """Control: the clean copy's timestep is 0, not the chunk's own.

    Asserted from both directions: the clean rows of the packed sequence are
    what modulating at 0 gives, and modulating at ``timesteps`` instead gives
    something else. Without the second half the first is vacuous whenever the
    timestep embedding is flat.
    """
    dtype = torch.float32
    n_chunks = 3
    _, port = _tower_pair(dtype)
    text, anchor, clean, noised, timesteps = _window(n_chunks, dtype)
    mask = _ref_mask(n_chunks, port.core)
    tokens_per_chunk = FRAMES_PER_CHUNK * HEIGHT * WIDTH
    counts = [tokens_per_chunk] * (n_chunks - 1)
    with torch.no_grad():
        _, feats = port.forward_chunk_causal(
            text, anchor, clean, noised, timesteps, mask=mask, attend=_flex_attend
        )
        clean_tok = port.input_projections[port_enc.VIDEO](clean).flatten(1, -2)
        at_zero = port.chunk_modulate(
            clean_tok, port.t_embedding(torch.zeros_like(timesteps)), counts
        )
        at_t = port.chunk_modulate(
            clean_tok, port.t_embedding(timesteps[:, : n_chunks - 1]), counts
        )
    start, length = feats.clean_span
    assert torch.equal(feats.embeddings[:, start : start + length], at_zero)
    assert not torch.equal(at_zero, at_t)


def test_control_dense_blocks_are_a_prefix_not_an_interleave():
    """Control: where the dense block sits changes the model.

    The two stacks hold the *same* feed-forward modules, swapped between
    block 0 and block 1, so nothing here is a weight mismatch: only the
    composition differs.
    """
    torch.manual_seed(0)
    prefix = _port_core(depth=2, num_dense=1)
    _fill(prefix, 21)
    prefix.eval()
    swapped = copy.deepcopy(prefix)
    swapped.blocks[0].ffn, swapped.blocks[1].ffn = swapped.blocks[1].ffn, swapped.blocks[0].ffn
    positions = port_enc.chunk_causal_positions(
        NUM_REGISTERS, ANCHOR_HW, NUM_TEXT, (4, HEIGHT, WIDTH), 2, (0, 0, 0)
    ).unsqueeze(0)
    x = torch.randn(1, positions.shape[1], DIM) * 0.3
    mask = _ref_mask(2, prefix)
    with torch.no_grad():
        a, _ = prefix(x, positions, mask, (), _flex_attend)
        b, _ = swapped(x, positions, mask, (), _flex_attend)
    assert not torch.equal(a, b)


def test_dense_prefix_composition_matches_reference():
    """The port's per-block feed-forward kinds, against the reference's."""
    ref, port = _ref_core(), _port_core()
    kinds = []
    for got, want in zip(port.blocks, ref.blocks, strict=True):
        dense = isinstance(want.ffn, ref_layers.SwiGLUFeedForward)
        assert isinstance(got.ffn, port_enc.SwiGLUFeedForward) is dense
        kinds.append(dense)
    assert kinds == [True] * NUM_DENSE + [False] * (DEPTH - NUM_DENSE)


# ---------------------------------------------------------------------------
# The shipped checkpoint's own numbers
# ---------------------------------------------------------------------------


def _shipped_encoder_config():
    root = Path(praxis.__file__).resolve().parent.parent
    path = root / "tests" / "data" / "praxis-1-droid.json"
    if not path.exists():
        pytest.skip(f"the reference checkout has no {path.name}")
    return json.loads(path.read_text())["backbone"]


def test_shipped_config_matches_the_ported_defaults():
    """The facts the port's defaults encode, read off a shipped config.

    Including the one that is an assumption rather than a reading: the
    encoder's ``dense_blocks`` list happens to be a prefix. A port that reads
    the list as a set of indices loads the same checkpoint and runs a
    different model, so the prefix is worth asserting where it is true.
    """
    backbone = _shipped_encoder_config()
    encoder = backbone["encoder"]
    assert encoder["depth"] == 14
    assert tuple(encoder["dense_blocks"]) == (0, 1)
    assert len(encoder["dense_blocks"]) == 2
    assert tuple(encoder["dense_blocks"]) == tuple(range(len(encoder["dense_blocks"])))
    assert tuple(encoder["tap_blocks"]) == port_enc.DEFAULT_TAP_BLOCKS == (0, 2, 4, 6, 7, 9, 11, 13)
    assert encoder["positive_zero_residual_add"] is True
    # The encoder stack carries no per-block modulator; the decoder's does.
    assert "modulation_dim" not in encoder
    assert backbone["decoder"]["modulation_dim"] == 3072
    assert backbone["dim"] == 3072
    assert backbone["timestep_dim"] == 128
    assert backbone["num_registers"] == 4
    attention = backbone["attention"]
    assert attention["head_dim"] == 128
    assert attention["num_kv_heads"] == 8
    assert attention["num_heads"] == backbone["dim"] // attention["head_dim"] == 24
    assert encoder["input_projections"]["TextEncodingTensor"]["c_in"] == 2048
    assert encoder["input_projections"]["VideoTensor"]["c_in"] == 192
    assert encoder["input_projections"]["TextEncodingTensor"]["timestep_modulated"] is False
    assert encoder["input_projections"]["VideoTensor"]["timestep_modulated"] is True


def test_default_tower_has_the_shipped_shape():
    """Constructed at the defaults, with no checkpoint: block kinds and count."""
    core = port_enc.TrunkStack()
    assert len(core.blocks) == 14
    dense = [isinstance(b.ffn, port_enc.SwiGLUFeedForward) for b in core.blocks]
    assert dense == [True, True] + [False] * 12
    attn = core.blocks[0].mha
    assert (attn.num_heads, attn.num_kv_heads, attn.head_dim) == (24, 8, 128)


# ---------------------------------------------------------------------------
# Measured notes
# ---------------------------------------------------------------------------


def test_positive_zero_residual_add_preserves_signed_zeros():
    """What the ``+ 0.0`` is for, asserted on the operation itself.

    ``torch.equal`` cannot see this: it compares values, and ``-0.0 == 0.0``.
    Only ``signbit`` does.
    """
    z = torch.tensor([-0.0, 0.0, -1.5])
    assert torch.equal(z + 0.0, z)
    assert torch.signbit(z).tolist() == [True, False, True]
    assert torch.signbit(z + 0.0).tolist() == [False, False, True]


def test_positive_zero_residual_add_is_inert_on_this_towers_output():
    """Measured: the flag changes nothing the encoder tower returns.

    Nothing reads the stack output directly. It goes through
    ``EncoderOutputProjection``, and a dot product cannot see the sign of a
    zero addend, so the add is unobservable from here, signbit included. The
    flag is still read from the config and still applied, because the tensor
    it guards is what a later slice (or a kernel that writes exact zeros into
    the conditioning) would consume.
    """
    dtype = torch.float32
    (ref_cond, _), (port_cond, port_feats), port, window, mask = _run_pair(2, dtype)
    port.positive_zero_residual_add = False
    with torch.no_grad():
        off_cond, off_feats = port.forward_chunk_causal(
            *window, mask=mask, attend=_flex_attend
        )
    assert torch.equal(off_cond, ref_cond) and torch.equal(off_cond, port_cond)
    assert torch.equal(torch.signbit(off_cond), torch.signbit(ref_cond))
    for a, b in zip(off_feats.taps, port_feats.taps, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("dtype,ceiling", [(torch.float32, 1e-5), (torch.bfloat16, 0.1)])
def test_dense_sdpa_fallback_is_not_bit_exact_against_flex(dtype, ceiling):
    """Measured: the tower's default attention path is not the kernel's.

    With no mask and no ``attend``, the tower builds the dense boolean mask
    and runs plain SDPA. That is the parity driver for everything *around*
    attention, and it is close to but never bit-equal with the reference's
    compiled flex attention: a different kernel accumulates a softmax in a
    different order. Measured over this four-block stack, relative to the
    largest output magnitude, the gap is about ``6e-7`` in float32 and about
    ``4e-2`` in bfloat16. The bfloat16 number is not a bug: two kernels
    disagreeing in the last bit of each block's attention output compounds
    through four residual adds at seven decimal digits of headroom.

    This is why every parity test above drives the reference's own kernel.
    """
    ref, port = _tower_pair(dtype)
    text, anchor, clean, noised, timesteps = _window(2, dtype)
    with torch.no_grad():
        want, _ = ref.forward_chunk_causal(text, anchor, clean, noised, timesteps)
        got, _ = port.forward_chunk_causal(text, anchor, clean, noised, timesteps)
    assert got.shape == want.shape
    assert not torch.equal(got, want)
    gap = (got.float() - want.float()).abs().max() / want.float().abs().max()
    # A ceiling on a measured rounding difference, not a tolerance anything
    # is allowed to drift to.
    assert gap.item() < ceiling, f"the dense SDPA fallback drifted by {gap.item()}"
