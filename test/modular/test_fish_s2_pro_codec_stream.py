"""Fish S2-Pro codec streaming: the stitched chunks are one decode of the stream.

Frames go through the edge's real ``ScheduledLeftContextChunkPolicy`` and
``StreamBuffer``, then through ``CodecSubmodule``'s ``prepare_inputs`` /
``declare_step`` / ``preprocess`` / ``forward_batched`` / ``postprocess``
against a real ``CODEC_STATE`` pool, and the concatenated PCM is compared with
one decode of the whole stream.

The decoder is the real modded-DAC architecture, shrunk and with random
weights, in float64. Each chunk decodes its new frames from the state the
chunks before it left (the transformer's cached keys, each causal conv's last
inputs, the frame count), and the frames keep their stream positions, so even
the bfloat16 RoPE table rotates them as one decode of the whole stream does.
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
    CODEC_STATE,
    FishDACConfig,
    FishFastARConfig,
    FishGenerationConfig,
    FishS2ProConfig,
    FishSlowARConfig,
)
from mstar.model.fish_s2_pro.fish_s2_pro_model import clip_digest
from mstar.model.fish_s2_pro.submodules import CodecSubmodule, RefEncoderSubmodule
from mstar.model.submodule_base import ARNodeInputs, InputMetadata, ModelInputsFromEngine
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


def stream_codes(num_frames: int, seed: int = 1) -> torch.Tensor:
    """``[T, 4]`` frames as the AR streams them (it withholds the <|im_end|> frame)."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.cat(
        [
            torch.randint(0, SEMANTIC_CODES, (num_frames, 1), generator=g),
            torch.randint(0, RESIDUAL_CODES, (num_frames, NUM_CODEBOOKS - 1), generator=g),
        ],
        dim=1,
    )
    return codes


def whole_decode(decoder: FishDACDecoder, codes: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        wav = decoder(codes.t()[None])[0]
    return (wav.clamp(-1, 1) * 32767).to(torch.int16)


class CodecDriver:
    """The codec node with its state pool, driven as the engine drives a step:
    declare, admit, plan, preprocess, forward, commit, postprocess. ``captured``
    pads the step to a batch bucket with the capture template's rows on the
    pool's sink, as a replay does."""

    BUCKETS = (1, 2, 4, 8)

    def __init__(self, submodule: CodecSubmodule, max_slots: int = 8):
        from mstar.engine.resources.base import EngineResourceInfo
        from mstar.engine.resources.recurrent import RecurrentStateConfig, RecurrentStateSpec
        from mstar.engine.resources.recurrent.pool import RecurrentStatePool
        from mstar.engine.resources.runner import StepRunner
        from mstar.model.fish_s2_pro.components.incremental_dac import codec_state_blocks

        self.node = submodule
        blocks = {
            name: type(block)(shape=block.shape, dtype=torch.float64 if block.dtype == torch.float32 else block.dtype)
            for name, block in codec_state_blocks(submodule.codec).items()
        }
        spec = RecurrentStateSpec(CODEC_STATE, {"Codec"}, RecurrentStateConfig(
            num_layers=1, blocks=blocks, max_slots=max_slots))
        self.pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
        self.runner = StepRunner({CODEC_STATE: self.pool})
        self.added: set[str] = set()

    def step(self, items: dict, captured: bool = False) -> dict:
        from mstar.engine.resources.step import BucketKey, SlotLease, StepContext

        rids, inputs = list(items), list(items.values())
        ctx = StepContext(request_ids=tuple(rids), graph_walk="codec_chunk", slot=0, capture=False)
        rows = list(rids)
        if captured:
            bs = next(b for b in self.BUCKETS if b >= len(rids))
            rows += [f"pad{i}" for i in range(bs - len(rids))]
            width = self.node._bucket(max(item.kwargs["frames"] for item in inputs) or 1)
            template = ARNodeInputs(tensor_inputs={"codec_tokens": torch.zeros(width, NUM_CODEBOOKS, dtype=torch.long)},
                                    kwargs={"frames": width})
            ctx.slot_lease = SlotLease(slot=0, bucket=BucketKey("codec_chunk", bs, bs, width))
            ctx.set_padded_rids(tuple(rows))
            inputs = inputs + [template] * (bs - len(rids))
        for rid in rids:
            if rid not in self.added:
                self.pool.ingest_request(rid)
                self.added.add(rid)
        step = self.node.declare_step("codec_chunk", rows, inputs)
        step.set_ctx(ctx)
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(request_ids=rows, per_request_info={},
                                              resources={CODEC_STATE: self.pool}, step=step)
        packed = self.node.preprocess("codec_chunk", engine_inputs, inputs)
        with torch.no_grad():
            audio = self.node.forward_batched("codec_chunk", engine_inputs, **packed).row_outputs["audio_chunk"]
        self.runner.commit(step)
        out = {}
        for i, rid in enumerate(rids):
            out[rid] = {"audio_chunk": [audio[i]]}
            self.node.postprocess(rid, None, out[rid], inputs=items[rid])
        return out


def run_streams(submodule: CodecSubmodule, frames: dict[str, torch.Tensor], captured: bool = False):
    """Stream each request's frames through its own buffer; every step batches the
    requests that have a chunk ready, as the scheduler batches the codec.
    Returns, per request, ``(chunk_info, audio)`` per step and the step count."""
    cfg = submodule.codec
    driver = CodecDriver(submodule)
    buffers = {}
    for rid, request_frames in frames.items():
        buffers[rid] = StreamBuffer(
            request_id=rid,
            edge_name="codec_tokens",
            from_partition="AR",
            policy=ScheduledLeftContextChunkPolicy(cfg.chunk_schedule, cfg.chunk_frames, 0),
        )
        for i, frame in enumerate(request_frames):
            buffers[rid].pre_read_register(f"t{i}")
            buffers[rid].put(f"t{i}", frame)
        buffers[rid].signal_done()
    infos = {rid: SimpleNamespace(request_id=rid, rid_handle=rid) for rid in frames}
    steps = {rid: [] for rid in frames}
    count = 0
    while rids := [rid for rid, buffer in buffers.items() if buffer.has_chunk_ready()]:
        chunks = {rid: buffers[rid].pop_chunk() for rid in rids}
        meta = {rid: InputMetadata(stream_chunks={"codec_tokens": chunk.info}) for rid, chunk in chunks.items()}
        items = {
            rid: submodule.prepare_inputs(
                "codec_chunk", infos[rid], {"codec_tokens": [chunks[rid].data["data"]]}, input_metadata=meta[rid]
            )
            for rid in rids
        }
        assert submodule.can_batch(None, list(items.values()))
        # the capture key, known before prepare_inputs, is the bucket preprocess pads to
        assert submodule.cg_key_info("codec_chunk", meta, meta) == submodule._bucket(
            max(max(item.kwargs["frames"], 1) for item in items.values()))
        outputs = driver.step(items, captured=captured)
        count += 1
        for rid in rids:
            steps[rid].append((chunks[rid].info, outputs[rid]["audio_chunk"][0]))
    return steps, count


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


def test_transformer_receptive_field_matches_autograd(decoder):
    """A post-transformer frame depends on exactly ``layers * (window - 1)`` earlier frames."""
    module = decoder.quantizer.post_module
    frames, target = 20, 15
    x = torch.randn(1, decoder.config.latent_dim, frames, dtype=torch.float64, requires_grad=True)
    module(x)[0, :, target].sum().backward()
    touched = x.grad.abs().sum(1)[0].nonzero().flatten()
    assert int(touched.min()) == target - module.receptive_field
    assert int(touched.max()) == target


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("num_frames", [1, 2, 4, 5, 7, 10, 15, 16, 17, 26, 31, 41, 64, 90])
def test_streamed_chunks_stitch_into_the_whole_decode(config, decoder, num_frames, captured):
    """Ramp chunks, steady chunks and a short final chunk, each padded to its
    bucket (and, as a replay runs it, its batch padded with sink rows)."""
    submodule = CodecSubmodule(decoder, config)
    codes = stream_codes(num_frames)

    steps = run_streams(submodule, {"r": codes}, captured=captured)[0]["r"]

    full = whole_decode(decoder, codes)
    assert full.numel() == num_frames * HOP
    for info, audio in steps:
        assert audio.numel() == (info.num_items - info.context_items) * HOP
    assert_same_audio(streamed(steps), full)


@pytest.mark.parametrize("chunks", [(4, 8, 16, 32, 7), (1, 2, 3, 50), (67,)])
def test_incremental_decode_stitches_into_the_whole_decode(config, decoder, chunks):
    """Chunk by chunk from carried state (each chunk zero-padded to the batch's
    width beside a longer row) is one decode of the whole stream."""
    from mstar.model.fish_s2_pro.components.incremental_dac import IncrementalDACDecoder, codec_state_blocks

    inc = IncrementalDACDecoder(decoder)
    blocks = inc.state_blocks()
    assert {n: (b.shape, b.dtype) for n, b in blocks.items()} == {
        n: (b.shape, b.dtype) for n, b in codec_state_blocks(config.codec).items()
    }
    total = sum(chunks)
    codes = stream_codes(total, seed=7)
    other = stream_codes(total + 64, seed=8)
    state = {n: torch.zeros(2, *b.shape, dtype=torch.float64 if b.dtype == torch.float32 else b.dtype)
             for n, b in blocks.items()}
    out, start = [], 0
    with torch.no_grad():
        for n in chunks:
            width = n + 5
            window = torch.zeros(2, NUM_CODEBOOKS, width, dtype=torch.long)
            window[0, :, :n] = codes[start:start + n].t()
            window[1] = other[start:start + width].t()
            wav, state = inc(window, state, torch.tensor([n, width]))
            out.append(wav[0, : n * HOP])
            start += n
    streamed_audio = (torch.cat(out).clamp(-1, 1) * 32767).to(torch.int16)
    assert_same_audio(streamed_audio, whole_decode(decoder, codes))


def test_incremental_decoder_lays_out_transposed_conv_weights_for_its_gemm(config):
    """``tconv_gemm`` reads a transposed conv's weight as ``[C_in, kernel * C_out]``: a view
    of a channels-last weight, but a copy of the whole weight every step otherwise."""
    from mstar.model.fish_s2_pro.components.incremental_dac import IncrementalDACDecoder

    inc = IncrementalDACDecoder(random_decoder(config))
    tconvs = [m for m in inc.modules() if isinstance(m, torch.nn.ConvTranspose1d)]
    assert tconvs
    for conv in tconvs:
        assert conv.weight.permute(0, 2, 1).is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled codec decode runs on CUDA")
def test_compiled_incremental_decode_matches_eager_on_cuda(config):
    """The codec node compiles the incremental decode (``compile_regions``, dynamic
    shapes), so Inductor traces its gather-index tails, sliced overlap-adds and
    channels-last conv views; chunk after chunk from carried state it must decode
    as the eager module does."""
    from mstar.model.fish_s2_pro.components.incremental_dac import IncrementalDACDecoder

    inc = IncrementalDACDecoder(random_decoder(config).float().cuda())
    compiled = torch.compile(inc, dynamic=True)

    def stream(decode, chunks=(4, 8, 3)):
        state = {n: torch.zeros(2, *b.shape, dtype=b.dtype, device="cuda") for n, b in inc.state_blocks().items()}
        codes = stream_codes(sum(chunks) + 8, seed=3).t().cuda()
        out, start = [], 0
        with torch.no_grad():
            for n in chunks:
                window = torch.stack([codes[:, start:start + n + 2], codes[:, start + 1:start + n + 3]])
                wav, state = decode(window, state, torch.tensor([n, n + 2], device="cuda"))
                out.append(wav)
                start += n
        return out, state

    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):   # matmuls are fp32 by default
        eager, eager_state = stream(inc)
        got, got_state = stream(compiled)
    # Inductor reorders and fuses the decoder's reductions, so fp32 rounding differs by
    # ~1e-4; 1e-3 as the other compiled-vs-eager checks (e.g. test_qwen3_omni_fused_moe)
    for a, b in zip(got, eager, strict=True):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
    for name in eager_state:
        torch.testing.assert_close(got_state[name], eager_state[name], atol=1e-3, rtol=1e-3)


def test_streams_batched_at_different_offsets_each_stitch_into_their_own_decode(config, decoder):
    submodule = CodecSubmodule(decoder, config)
    codes = {"a": stream_codes(23, seed=3), "b": stream_codes(77, seed=4), "c": stream_codes(5, seed=5)}

    steps, _ = run_streams(submodule, codes, captured=True)

    for rid, request_codes in codes.items():
        assert_same_audio(streamed(steps[rid]), whole_decode(decoder, request_codes))


def test_frameless_terminal_flush_emits_nothing(config, decoder):
    """A stream that ends after its consumer drained it gets a final chunk with
    no frames: it decodes at the smallest bucket and emits nothing."""
    submodule = CodecSubmodule(decoder, config)
    item = submodule.prepare_inputs(
        "codec_chunk",
        SimpleNamespace(request_id="r", rid_handle="r"),
        {},
        input_metadata=InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(10, 0, 0, is_final=True)}),
    )
    assert item.kwargs["frames"] == 0
    out = CodecDriver(submodule).step({"r": item})
    assert out["r"]["audio_chunk"][0].numel() == 0


def test_codec_buckets_are_the_chunk_widths(config, decoder):
    submodule = CodecSubmodule(decoder, config)
    assert submodule.windows == [2, 3, 5]
    assert [submodule._bucket(n) for n in (1, 2, 3, 4, 5)] == [2, 2, 3, 5, 5]
    with pytest.raises(ValueError, match="maximum is 5"):
        submodule._bucket(6)


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
