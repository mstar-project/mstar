"""Fish S2-Pro codec streaming: the stitched chunks are one decode of the stream.

Frames go through the edge's real ``ScheduledLeftContextChunkPolicy`` and
``StreamBuffer``, then through ``CodecSubmodule``'s ``prepare_inputs`` /
``preprocess`` / ``forward_batched`` / ``postprocess``, and the concatenated
PCM is compared with one decode of the whole stream.

The decoder is the real modded-DAC architecture, shrunk and with random
weights, in float64. Its quantizer transformer (2 layers, 4-frame window)
looks 6 frames back and its causal conv stack 22 more, so a window whose
context covers both decodes its new frames exactly as the whole stream does:
the frames keep their stream positions, so even the bfloat16 RoPE table
rotates them the same way.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.model.fish_s2_pro.components.dac import (
    FishDACDecoder,
    FishDACEncoder,
    _fold_weight_norm,
    load_codec_state,
    precompute_freqs_cis,
)
from mstar.model.fish_s2_pro.config import (
    FishDACConfig,
    FishFastARConfig,
    FishGenerationConfig,
    FishS2ProConfig,
    FishSlowARConfig,
)
from mstar.model.fish_s2_pro.fish_s2_pro_model import clip_digest
from mstar.model.fish_s2_pro.submodules import ARSubmodule, CodecSubmodule, RefEncoderSubmodule
from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine, StopCheck
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer, StreamChunkInfo

HOP = 16
NUM_CODEBOOKS = 4
SEMANTIC_CODES, RESIDUAL_CODES = 32, 8


def tiny_codec(**overrides) -> FishDACConfig:
    values = dict(
        encoder_dim=16,
        encoder_rates=(2, 2),
        encoder_transformer_layers=(0, 1),
        encoder_window=4,
        encoder_block_size=256,
        decoder_dim=32,
        decoder_rates=(2, 2),
        latent_dim=32,
        downsample_factor=(2, 2),
        semantic_codebook_size=SEMANTIC_CODES,
        codebook_size=RESIDUAL_CODES,
        codebook_dim=4,
        n_residual_codebooks=NUM_CODEBOOKS - 1,
        post_layers=2,
        post_heads=2,
        post_head_dim=16,
        post_intermediate_size=48,
        post_window=4,
        post_block_size=256,
        chunk_schedule=(2, 3),
        chunk_frames=5,
        conv_context_frames=24,
    )
    values.update(overrides)
    return FishDACConfig(**values)


def tiny_config(**codec) -> FishS2ProConfig:
    return FishS2ProConfig(
        semantic_start_token_id=100,
        semantic_end_token_id=100 + SEMANTIC_CODES - 1,
        eos_token_id=99,
        slow_ar=FishSlowARConfig(
            dim=32,
            n_layer=2,
            n_head=4,
            n_local_heads=2,
            head_dim=8,
            intermediate_size=48,
            vocab_size=140,
            max_seq_len=512,
        ),
        fast_ar=FishFastARConfig(
            dim=32,
            text_dim=32,
            n_layer=2,
            n_head=4,
            n_local_heads=2,
            head_dim=8,
            intermediate_size=48,
            vocab_size=SEMANTIC_CODES,
            num_codebooks=NUM_CODEBOOKS,
        ),
        codec=tiny_codec(**codec),
        generation=FishGenerationConfig(max_new_tokens=64),
    )


def random_decoder(config: FishS2ProConfig, seed: int = 0) -> FishDACDecoder:
    torch.manual_seed(seed)
    decoder = FishDACDecoder(config.codec)
    with torch.no_grad():
        for name, param in decoder.named_parameters():
            if name.endswith(("alpha", "gamma")) or "norm" in name:
                param.uniform_(0.5, 1.5)
    return decoder.double().eval()


def stream_codes(num_frames: int, seed: int = 1, eos: bool = True) -> torch.Tensor:
    """``[T (+1), 4]`` frames as the AR samples them; the last one is <|im_end|>."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.cat(
        [
            torch.randint(0, SEMANTIC_CODES, (num_frames, 1), generator=g),
            torch.randint(0, RESIDUAL_CODES, (num_frames, NUM_CODEBOOKS - 1), generator=g),
        ],
        dim=1,
    )
    if eos:
        codes = torch.cat([codes, torch.tensor([[SEMANTIC_CODES, 0, 0, 0]])])
    return codes


def whole_decode(decoder: FishDACDecoder, codes: torch.Tensor) -> torch.Tensor:
    codes = codes[codes[:, 0] < SEMANTIC_CODES]
    with torch.no_grad():
        wav = decoder(codes.t()[None])[0]
    return (wav.clamp(-1, 1) * 32767).to(torch.int16)


def ar_streamed(config: FishS2ProConfig, frames: torch.Tensor, rid: str = "r") -> list[torch.Tensor]:
    """The frames the AR puts on the codec stream: each step's stop check
    withholds the <|im_end|> frame (head index in codebook 0)."""
    ar = ARSubmodule.__new__(ARSubmodule)
    ar.eos_code, ar.request_states = config.eos_code, {}
    info = SimpleNamespace(max_tokens=frames.shape[0] + 1, resource_configs={})
    kept = []
    for frame in frames:
        result = ar.check_stop(rid, info, {"head_index": [frame[:1]], "codec_tokens": [frame]})
        if not (isinstance(result, StopCheck) and "codec_tokens" in result.withheld):
            kept.append(frame)
    return kept


def run_streams(submodule: CodecSubmodule, frames: dict[str, torch.Tensor]):
    """Stream each request's frames, as the AR samples them, through its own
    buffer; every step batches the requests that have a chunk ready, as the
    scheduler batches the codec. Returns, per request, ``(chunk_info, audio)``
    per step (a declined chunk emits no audio) and the decode count."""
    cfg = submodule.codec
    buffers = {}
    for rid, request_frames in frames.items():
        buffers[rid] = StreamBuffer(
            request_id=rid,
            edge_name="codec_tokens",
            from_partition="AR",
            policy=ScheduledLeftContextChunkPolicy(cfg.chunk_schedule, cfg.chunk_frames, cfg.left_context_frames),
        )
        streamed_frames = ar_streamed(submodule.config, request_frames, rid)
        for i, frame in enumerate(streamed_frames):
            buffers[rid].pre_read_register(f"t{i}")
            buffers[rid].put(f"t{i}", frame)
        buffers[rid].signal_done(len(streamed_frames))
    infos = {rid: SimpleNamespace(request_id=rid, rid_handle=rid) for rid in frames}
    steps = {rid: [] for rid in frames}
    decodes = 0
    while rids := [rid for rid, buffer in buffers.items() if buffer.has_chunk_ready()]:
        chunks = {rid: buffers[rid].pop_chunk() for rid in rids}
        meta = {rid: InputMetadata(stream_chunks={"codec_tokens": chunk.info}) for rid, chunk in chunks.items()}
        # an empty chunk reaches the node as an edge with no tensors
        prepared = {
            rid: submodule.prepare_inputs(
                "codec_chunk", infos[rid],
                {"codec_tokens": [] if chunks[rid].data["data"] is None else [chunks[rid].data["data"]]},
                input_metadata=meta[rid],
            )
            for rid in rids
        }
        # as the engine does: a declined chunk runs no forward
        for rid in rids:
            if prepared[rid] is None:
                assert chunks[rid].is_final
                steps[rid].append((chunks[rid].info, torch.empty(0, dtype=torch.int16)))
        run = [rid for rid in rids if prepared[rid] is not None]
        if not run:
            continue
        items = [prepared[rid] for rid in run]
        engine_inputs = ModelInputsFromEngine(request_ids=run, per_request_info={})
        assert submodule.can_batch(None, items)
        packed = submodule.preprocess("codec_chunk", engine_inputs, items)
        decodes += 1
        with torch.no_grad():
            outputs = submodule.forward_batched("codec_chunk", engine_inputs, **packed)
        for rid, item in zip(run, items, strict=True):
            submodule.postprocess(rid, infos[rid], outputs[rid], inputs=item)
            steps[rid].append((chunks[rid].info, outputs[rid]["audio_chunk"][0]))
    return steps, decodes


def _no_device():
    raise AssertionError("the codec built a device tensor")


def streamed(steps) -> torch.Tensor:
    return torch.cat([audio for _, audio in steps])


def assert_same_audio(out: torch.Tensor, full: torch.Tensor) -> None:
    """Same samples, up to float rounding: the window and the whole stream run the
    transformer's float32 norms and reductions over different lengths, which can
    move a sample across an int16 step."""
    assert out.shape == full.shape
    diff = (out.int() - full.int()).abs()
    assert int(diff.max()) <= 1
    assert float((diff > 0).float().mean()) < 1e-2


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    """Tiny ops: intra-op threads only add overhead (and contend on a loaded host)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def config() -> FishS2ProConfig:
    return tiny_config()


@pytest.fixture(scope="module")
def decoder(config) -> FishDACDecoder:
    return random_decoder(config)


def test_default_context_is_the_decoders_receptive_field(config, decoder):
    assert decoder.conv_receptive_field() <= config.codec.conv_context_frames
    assert config.codec.transformer_context_frames == decoder.quantizer.post_module.receptive_field == 6
    assert config.codec.left_context_frames == 6 + config.codec.conv_context_frames
    assert config.codec.exact_streaming
    full = FishDACConfig()
    assert (full.transformer_context_frames, full.left_context_frames, full.hop_length) == (1016, 1026, 2048)
    assert FishDACDecoder(full).conv_receptive_field() == full.conv_context_frames == 10


def test_conv_receptive_field_matches_autograd(decoder):
    """The earliest latent frame a frame's first sample depends on is ``frame - rf``."""
    frames, target = 40, 32
    latents = torch.randn(1, decoder.config.latent_dim, frames, dtype=torch.float64, requires_grad=True)
    audio = decoder.run_conv_stack(latents)
    audio[0, target * HOP].backward()
    touched = latents.grad.abs().sum(1)[0].nonzero().flatten()
    assert int(touched.min()) == target - decoder.conv_receptive_field()


def test_transformer_receptive_field_matches_autograd(decoder):
    """A post-transformer frame depends on exactly ``layers * (window - 1)`` earlier frames."""
    module = decoder.quantizer.post_module
    frames, target = 20, 15
    x = torch.randn(1, decoder.config.latent_dim, frames, dtype=torch.float64, requires_grad=True)
    module(x)[0, :, target].sum().backward()
    touched = x.grad.abs().sum(1)[0].nonzero().flatten()
    assert int(touched.min()) == target - module.receptive_field
    assert int(touched.max()) == target


@pytest.mark.parametrize("num_frames", [1, 2, 4, 5, 7, 10, 15, 16, 17, 26, 31, 41, 64, 90])
def test_streamed_chunks_stitch_into_the_whole_decode(config, decoder, num_frames):
    """Ramp chunks, steady chunks and the terminal flush, around every boundary of the
    schedule and past the point where the context stops growing."""
    submodule = CodecSubmodule(decoder, config)
    codes = stream_codes(num_frames)

    steps = run_streams(submodule, {"r": codes})[0]["r"]

    full = whole_decode(decoder, codes)
    assert full.numel() == num_frames * HOP
    for info, audio in steps:
        # every item is a frame: the AR withheld <|im_end|>
        assert info.start_offset + info.num_items <= num_frames
        assert audio.numel() == (info.num_items - info.context_items) * HOP
    assert_same_audio(streamed(steps), full)


def test_short_context_changes_the_audio(decoder):
    """The test has teeth: windows with half the conv stack's context drift audibly."""
    conv = decoder.conv_receptive_field()
    config = tiny_config(conv_context_frames=conv, left_context_frames=conv // 2)
    codes = stream_codes(90)

    steps = run_streams(CodecSubmodule(decoder, config), {"r": codes})[0]["r"]

    out, full = streamed(steps), whole_decode(decoder, codes)
    assert out.shape == full.shape
    assert int((out.int() - full.int()).abs().max()) > 64


def test_streams_batched_at_different_offsets_each_stitch_into_their_own_decode(config, decoder):
    submodule = CodecSubmodule(decoder, config)
    codes = {"a": stream_codes(23, seed=3), "b": stream_codes(77, seed=4), "c": stream_codes(5, seed=5)}

    steps, _ = run_streams(submodule, codes)

    for rid, request_codes in codes.items():
        assert_same_audio(streamed(steps[rid]), whole_decode(decoder, request_codes))


def test_context_only_terminal_flush_decodes_nothing(config, decoder):
    """A stream whose last window holds only context declines it: it emits
    nothing and runs no decode."""
    submodule = CodecSubmodule(decoder, config)
    # ramp 2 + 3, then chunks of 5: 15 frames end exactly on a window
    steps, decodes = run_streams(submodule, {"r": stream_codes(15)})
    steps = steps["r"]

    info, audio = steps[-1]
    assert info.is_final and info.num_items == info.context_items
    assert audio.numel() == 0
    assert decodes == len(steps) - 1


@pytest.mark.parametrize("num_frames", [1, 3, 4, 6, 14, 16])
def test_the_ar_withholds_im_end_and_the_final_window_keeps_the_last_frames(config, decoder, num_frames):
    """Off a window boundary, the stream's final chunk carries its last frames
    and emits their audio, rather than a flush around the <|im_end|> frame."""
    submodule = CodecSubmodule(decoder, config)
    codes = stream_codes(num_frames)
    assert len(ar_streamed(config, codes)) == num_frames

    steps, _ = run_streams(submodule, {"r": codes})
    info, audio = steps["r"][-1]

    assert info.is_final and info.num_items > info.context_items
    assert audio.numel() == (info.num_items - info.context_items) * HOP
    assert_same_audio(streamed(steps["r"]), whole_decode(decoder, codes))


def test_a_stream_of_only_im_end_ends_with_an_empty_chunk(config, decoder, monkeypatch):
    """The AR's first frame was <|im_end|>: the stream gets no item, and the
    codec declines its final chunk, which reaches it with no tensor, before
    building any: it decodes nothing and emits no audio."""
    submodule = CodecSubmodule(decoder, config)
    monkeypatch.setattr(submodule, "get_device", _no_device)
    steps, decodes = run_streams(submodule, {"r": stream_codes(0)})

    ((info, audio),) = steps["r"]
    assert info == StreamChunkInfo(0, 0, 0, True)
    assert audio.numel() == 0 and decodes == 0


def test_codec_takes_every_item_as_a_frame(config, decoder):
    submodule = CodecSubmodule(decoder, config)
    frames = stream_codes(3, eos=False)
    item = submodule.prepare_inputs(
        "codec_chunk",
        SimpleNamespace(request_id="r", rid_handle="r"),
        {"codec_tokens": [frames]},
    )
    assert item.tensor_inputs["codec_tokens"].shape == (NUM_CODEBOOKS, 3)
    assert item.kwargs == {"frames": 3, "context": 0, "conv_start": 0, "start_offset": 0}


def test_codec_preprocess_stages_window_geometry_as_the_per_column_copies_did(config, decoder):
    submodule = CodecSubmodule(decoder, config)
    windows = [(0, 0, 4), (4, 4, 12), (40, 30, 62), (100, 30, 30)]  # (start_offset, context, frames)
    items = [
        submodule.prepare_inputs(
            "codec_chunk",
            SimpleNamespace(request_id=str(i), rid_handle=i),
            {"codec_tokens": [stream_codes(frames, eos=False)]},
            input_metadata=InputMetadata(
                stream_chunks={"codec_tokens": StreamChunkInfo(start, context, frames, is_final=False)}
            ),
        )
        for i, (start, context, frames) in enumerate(windows)
    ]
    packed = submodule.preprocess("codec_chunk", ModelInputsFromEngine(request_ids=[], per_request_info={}), items)
    for name in ("start_offset", "conv_start"):
        old = torch.tensor([item.kwargs[name] for item in items], dtype=torch.long)
        assert packed[name].dtype == old.dtype and packed[name].shape == old.shape and torch.equal(packed[name], old)
    assert packed["start_offset"].tolist() == [0, 4, 40, 100]


@pytest.mark.parametrize("schedule", [(), (2, 3)])
def test_codec_sizes_its_window_with_or_without_a_schedule(schedule):
    config = tiny_config(chunk_schedule=schedule)
    submodule = CodecSubmodule(random_decoder(config), config)
    assert submodule.max_new_frames == max([*schedule, config.codec.chunk_frames])


def test_codec_rejects_conv_context_below_the_receptive_field():
    config = tiny_config(conv_context_frames=1)
    with pytest.raises(ValueError, match="receptive field"):
        CodecSubmodule(random_decoder(config), config)


# ---------------------------------------------------------------------------
# codec.pth loading and the reference encoder
# ---------------------------------------------------------------------------


def test_weight_norm_folds_to_the_parametrized_weight():
    torch.manual_seed(0)
    modern = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 5, 4))
    legacy = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 5, 4))
    for module in (modern, legacy):
        folded = dict(_fold_weight_norm(module.state_dict()))
        assert set(folded) == {"weight", "bias"}
        assert torch.equal(folded["weight"], module.weight)


def test_codec_state_loads_completely_and_rejects_gaps(config, decoder):
    state = {name: tensor.float() for name, tensor in decoder.state_dict().items()}
    state["quantizer.post_module.freqs_cis"] = precompute_freqs_cis(256, 16, 10000.0)
    state["quantizer.post_module.causal_mask"] = torch.ones(2, 2, dtype=torch.bool)
    fresh = FishDACDecoder(config.codec)
    prefixes = ("quantizer.", "decoder.")
    load_codec_state(fresh, state, prefixes)
    for name, param in fresh.named_parameters():
        assert torch.equal(param, state[name]), name

    with pytest.raises(RuntimeError, match="did not initialize"):
        load_codec_state(
            FishDACDecoder(config.codec), {k: v for k, v in state.items() if "decoder.model.0" not in k}, prefixes
        )
    with pytest.raises(RuntimeError, match="no place"):
        load_codec_state(FishDACDecoder(config.codec), {**state, "decoder.extra.weight": torch.zeros(1)}, prefixes)
    bad_rope = dict(state, **{"quantizer.post_module.freqs_cis": torch.zeros(256, 8, 2, dtype=torch.bfloat16)})
    with pytest.raises(RuntimeError, match=r"by up to 1; the tolerance is 0\.015625$"):
        load_codec_state(FishDACDecoder(config.codec), bad_rope, prefixes)
    short_rope = dict(state, **{"quantizer.post_module.freqs_cis": precompute_freqs_cis(128, 16, 10000.0)})
    with pytest.raises(RuntimeError, match=r"freqs_cis: checkpoint \(128, 8, 2\) vs \(256, 8, 2\)"):
        load_codec_state(FishDACDecoder(config.codec), short_rope, prefixes)


def test_codec_runs_the_checkpoints_rope_table_even_one_bf16_step_off(config, decoder):
    """Some hosts compute one entry of a table one bf16 step away from the one
    stored in ``codec.pth``; the stored table loads, so every host runs the same
    values. A table from another rope base is still refused."""
    state = {name: tensor.float() for name, tensor in decoder.state_dict().items()}
    key = "quantizer.post_module.freqs_cis"
    stored = precompute_freqs_cis(256, 16, 10000.0)
    stored.view(torch.int16).view(-1)[1001] += 1
    fresh = FishDACDecoder(config.codec)
    assert not torch.equal(fresh.quantizer.post_module.freqs_cis, stored)
    load_codec_state(fresh, {**state, key: stored}, ("quantizer.", "decoder."))
    assert torch.equal(fresh.quantizer.post_module.freqs_cis, stored)

    other_base = precompute_freqs_cis(256, 16, 5000.0)
    target = FishDACDecoder(config.codec)
    gap = (other_base.float() - target.quantizer.post_module.freqs_cis.float()).abs().max().item()
    assert gap > 2.0**-6
    with pytest.raises(RuntimeError, match=rf"RoPE table {key} .* by up to {gap:.3g}; the tolerance is 0\.015625"):
        load_codec_state(target, {**state, key: other_base}, ("quantizer.", "decoder."))


def test_ref_encoder_emits_frames_and_memoises_by_clip(config):
    torch.manual_seed(0)
    encoder = FishDACEncoder(config.codec).eval()
    submodule = RefEncoderSubmodule(encoder, config)
    clip = torch.randn(5 * HOP + 3)
    calls = []
    original = submodule.encode
    submodule.encode = lambda waveform: calls.append(1) or original(waveform)
    info = SimpleNamespace(request_id="r", rid_handle="r", step_metadata={"ref_clip_key": clip_digest(clip)})

    item = submodule.prepare_inputs("ar_prefill_clone", info, {"audio_inputs": [clip]})
    assert item.kwargs == {"clip_key": clip_digest(clip)}
    first = submodule.forward("ar_prefill_clone", None, **item.tensor_inputs, **item.kwargs)["ref_codes"][0]
    again = submodule.forward("ar_prefill_clone", None, waveform=clip, clip_key=clip_digest(clip))

    assert first.shape == (config.codec.frames_for_samples(clip.numel()), NUM_CODEBOOKS) == (6, NUM_CODEBOOKS)
    assert first.dtype == torch.long
    assert int(first[:, 0].max()) < SEMANTIC_CODES and int(first[:, 1:].max()) < RESIDUAL_CODES
    assert torch.equal(again["ref_codes"][0], first) and len(calls) == 1
