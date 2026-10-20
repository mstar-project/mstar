"""Orpheus SNAC streaming: the stitched chunks are one whole-utterance decode.

Each test drives LLM tokens through the edge's real chunk policy (taken from
``OrpheusModel.get_partition_topology``) and ``StreamBuffer``, then through
``SNACDecoderSubmodule``'s ``cg_key_info`` / ``prepare_inputs`` /
``preprocess`` / ``forward_batched`` / ``postprocess``, and compares the
concatenated audio with one SNAC decode of every whole frame.

SNAC's decoder is not causal: a frame's audio depends on the frames on both
sides of it. A window has to decode enough of them, emit only the frames it
owns, and the stream has to start at frame 0 and end at the last frame.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from mstar.model.orpheus.components.snac import SNAC
from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.orpheus.submodules import SNACDecoderSubmodule
from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine
from mstar.streaming.stream_buffer import StreamBuffer

TPF = 7      # tokens per frame
HOP = 2048   # samples per frame


def _policy(config: OrpheusModelConfig):
    """The LLM -> SNAC edge's chunk policy, as the model builds it."""
    topology = OrpheusModel.get_partition_topology(SimpleNamespace(config=config))
    (connection,) = topology.connections
    return connection.chunk_policy_factory()


def _tokens(config: OrpheusModelConfig, frames: int, seed: int = 0, eos: bool = False) -> torch.Tensor:
    """``frames`` random audio frames as Orpheus token ids (code + 4096 * position)."""
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 4096, (frames, TPF), generator=generator)
    tokens = (codes + 4096 * torch.arange(TPF) + config.custom_token_base_id + 10).flatten()
    if eos:
        tokens = torch.cat([tokens, torch.tensor([config.stop_token_id])])
    return tokens


def _stream_batched(
    submodule: SNACDecoderSubmodule, tokens: dict[str, torch.Tensor], offsets: dict[str, int] | None = None,
) -> tuple[dict[str, list[torch.Tensor]], list]:
    """Stream each request's tokens through its own buffer, one token per LLM
    step. Request ``r`` joins ``offsets[r]`` steps late, so batches mix
    requests at different points of the chunk schedule. Every step batches
    the requests that have a chunk ready. Returns each request's emitted
    audio per chunk and each step's ``cg_key_info``."""
    offsets = offsets or {}
    buffers = {
        rid: StreamBuffer(request_id=rid, edge_name="new_token", from_partition="LLM",
                          policy=_policy(submodule.config))
        for rid in tokens
    }
    fwd_infos = {rid: SimpleNamespace(request_id=rid, rid_handle=rid) for rid in tokens}
    audio = {rid: [] for rid in tokens}
    keys = []
    horizon = max(len(t) + offsets.get(rid, 0) for rid, t in tokens.items()) + 1
    for step in range(horizon + 1):
        for rid, request_tokens in tokens.items():
            i = step - offsets.get(rid, 0)
            if 0 <= i < len(request_tokens):
                buffers[rid].pre_read_register(f"t{i}")
                buffers[rid].put(f"t{i}", request_tokens[i].view(1))
            elif i == len(request_tokens):
                buffers[rid].signal_done()
        while rids := [rid for rid, buffer in buffers.items() if buffer.has_chunk_ready()]:
            chunks = {rid: buffers[rid].pop_chunk() for rid in rids}
            meta = {rid: InputMetadata(stream_chunks={"new_token": chunk.info}) for rid, chunk in chunks.items()}
            keys.append(submodule.cg_key_info(
                "snac_chunk", {rid: fwd_infos[rid] for rid in rids}, per_request_input_metadata=meta,
            ))
            engine_inputs = ModelInputsFromEngine(
                request_ids=rids, per_request_info={}, per_request_input_metadata=meta, captured=False,
            )
            items = [
                submodule.prepare_inputs(
                    "snac_chunk", fwd_infos[rid],
                    # as the worker routes it: an empty chunk carries no tensor
                    {"new_token": [] if chunks[rid].data["data"] is None else [chunks[rid].data["data"]]},
                    input_metadata=meta[rid], is_final_stream_chunk=chunks[rid].is_final,
                )
                for rid in rids
            ]
            assert submodule.can_batch(None, items)
            packed = submodule.preprocess("snac_chunk", engine_inputs, items)
            with torch.no_grad():
                outputs = submodule.forward_batched("snac_chunk", engine_inputs, **packed)
            for rid, item in zip(rids, items, strict=True):
                submodule.postprocess(rid, fwd_infos[rid], outputs[rid], inputs=item)
                audio[rid].append(outputs[rid]["audio_chunk"][0].reshape(-1))
    return audio, keys


def _stream(submodule: SNACDecoderSubmodule, tokens: torch.Tensor) -> list[torch.Tensor]:
    return _stream_batched(submodule, {"r": tokens})[0]["r"]


def _whole(submodule: SNACDecoderSubmodule, tokens: torch.Tensor) -> torch.Tensor:
    """One decode of every whole frame, int16 like the stream."""
    frames = tokens.numel() // TPF
    codes = ((tokens[: frames * TPF] - submodule.config.custom_token_base_id - 10) % 4096).view(1, frames, TPF)
    with torch.no_grad():
        audio = submodule.snac_model.decode(list(submodule._extract_snac_codes(codes)))
    return (audio.clamp(-1, 1) * 32767).to(torch.int16).reshape(-1)


def _snr(ref: torch.Tensor, x: torch.Tensor) -> float:
    ref, x = ref.double(), x.double()
    return float(10 * torch.log10(ref.pow(2).sum() / (ref - x).pow(2).sum().clamp_min(1e-20)))


# ---------------------------------------------------------------------------
# A fake decoder with a known two-sided receptive field: exact arithmetic
# ---------------------------------------------------------------------------

class _NeighbourDecoder(torch.nn.Module):
    """Frame ``f``'s 2048 samples mix frames ``f - 2 .. f + 2`` of the window.

    Frames outside the window count as zero, like the real decoder's zero
    padding, so a window that cuts into a frame's neighbourhood changes it.
    Every sample is keyed to the absolute codes around it, so a dropped,
    repeated or misplaced frame changes the stream.
    """

    REACH = 2

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # gives the module a device

    def decode(self, codes: list[torch.Tensor]) -> torch.Tensor:
        c0, c1, c2 = (c.double() for c in codes)
        frames = c0.shape[1]
        key = c0 + c1.view(-1, frames, 2).sum(-1) * 7 + c2.view(-1, frames, 4).sum(-1) * 13  # (B, frames)
        padded = torch.nn.functional.pad(key, (self.REACH, self.REACH))
        weights = torch.tensor([1.0, 3.0, 5.0, 3.0, 1.0], dtype=torch.float64)
        mixed = sum(weights[k] * padded[:, k : k + frames] for k in range(2 * self.REACH + 1))
        ramp = torch.linspace(0.0, 1.0, HOP, dtype=torch.float64)
        audio = (mixed[..., None] * (1 + ramp)).reshape(-1, 1, frames * HOP)
        return (audio / 2e6).float()


@pytest.fixture
def config() -> OrpheusModelConfig:
    return OrpheusModelConfig()


@pytest.fixture
def fake(config) -> SNACDecoderSubmodule:
    return SNACDecoderSubmodule(_NeighbourDecoder(), config)


@pytest.mark.parametrize("frames", [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 30, 61])
def test_stream_matches_whole_decode_exactly(fake, config, frames):
    tokens = _tokens(config, frames, seed=frames)
    stream = torch.cat(_stream(fake, tokens))
    whole = _whole(fake, tokens)
    assert stream.numel() == whole.numel() == frames * HOP
    torch.testing.assert_close(stream, whole, rtol=0, atol=0)


@pytest.mark.parametrize("frames", [3, 9, 40])
def test_first_and_last_frames_are_emitted(fake, config, frames):
    tokens = _tokens(config, frames, seed=1)
    stream = torch.cat(_stream(fake, tokens))
    whole = _whole(fake, tokens)
    torch.testing.assert_close(stream[:HOP], whole[:HOP], rtol=0, atol=0)
    torch.testing.assert_close(stream[-HOP:], whole[-HOP:], rtol=0, atol=0)


def test_trailing_eos_is_not_audio(fake, config):
    # The LLM streams its EOS token last: 7N + 1 tokens are N frames of audio.
    tokens = _tokens(config, 20, seed=2, eos=True)
    stream = torch.cat(_stream(fake, tokens))
    torch.testing.assert_close(stream, _whole(fake, tokens), rtol=0, atol=0)
    assert stream.numel() == 20 * HOP


def test_stream_shorter_than_a_frame_emits_nothing(fake, config):
    tokens = _tokens(config, 1)[:4]
    chunks = _stream(fake, tokens)
    assert sum(chunk.numel() for chunk in chunks) == 0


def test_first_audio_is_no_later_than_main(fake, config):
    # The old 28-token sliding window released its first audio after 4 frames.
    policy = _policy(config)
    assert not policy.is_ready(4 * TPF - 1) and policy.is_ready(4 * TPF)
    first = _stream(fake, _tokens(config, 4))[0]
    assert first.numel() == (4 - config.snac_right_context_frames) * HOP


def test_mixed_batches_match_each_requests_whole_decode(fake, config):
    # Requests joining one chunk apart put first, ramp, steady and flush
    # windows of different sizes in one step (cg_key_info None: no capture).
    tokens = {
        "a": _tokens(config, 23, seed=3),
        "b": _tokens(config, 9, seed=4, eos=True),
        "c": _tokens(config, 14, seed=5),
    }
    audio, keys = _stream_batched(fake, tokens, offsets={"a": 0, "b": 4 * TPF, "c": 8 * TPF})
    for rid, request_tokens in tokens.items():
        torch.testing.assert_close(torch.cat(audio[rid]), _whole(fake, request_tokens), rtol=0, atol=0)
    assert None in keys and any(isinstance(key, int) for key in keys)


GEOMETRIES = {
    "default": {},
    "no ramp": dict(snac_chunk_schedule=(4,)),
    "right context 3": dict(snac_right_context_frames=3),
    "reference window": dict(snac_chunk_schedule=(4,), snac_chunk_frames=1,
                             snac_left_context_frames=1, snac_right_context_frames=2),
}


@pytest.mark.parametrize("overrides", list(GEOMETRIES.values()), ids=list(GEOMETRIES))
def test_no_window_size_falls_back_to_eager(overrides):
    """Every window of every stream length, final flush included, names a
    captured size, so the scheduler (which batches each size apart) never
    hands SNAC a step without a graph."""
    config = OrpheusModelConfig(**overrides)
    submodule = SNACDecoderSubmodule(_NeighbourDecoder(), config)
    configs = submodule.get_accelerator_graph_configs(torch.device("cpu"))
    captured = {cfg.additional_key_info for cfg in configs}
    assert submodule.split_batches_by_capture_key("snac_chunk")
    assert not any(cfg.compile for cfg in configs)
    for frames in range(0, 64):
        for eos in (False, True):
            tokens = _tokens(config, frames, eos=eos)
            if not tokens.numel():
                continue
            buffer = StreamBuffer(request_id="r", edge_name="new_token", from_partition="LLM", policy=_policy(config))
            for i, token in enumerate(tokens):
                buffer.pre_read_register(f"t{i}")
                buffer.put(f"t{i}", token.view(1))
            buffer.signal_done()
            info = SimpleNamespace(request_id="r", rid_handle="r")
            while buffer.has_chunk_ready():
                chunk = buffer.pop_chunk()
                meta = InputMetadata(stream_chunks={"new_token": chunk.info})
                key = submodule.cg_key_info("snac_chunk", {"r": info}, per_request_input_metadata={"r": meta})
                item = submodule.prepare_inputs("snac_chunk", info, {"new_token": [chunk.data["data"]]},
                                                input_metadata=meta)
                assert key in captured, (frames, eos, chunk.info)
                assert item.input_ids.shape[0] == key
                assert submodule.can_use_accelerator_graphs(SimpleNamespace(graph_walk="snac_chunk"), [item, item])


def test_model_kwargs_override_the_window(monkeypatch):
    from mstar.model.orpheus import orpheus_model

    monkeypatch.setattr(orpheus_model, "_resolve_local_hf_snapshot", lambda repo_id, cache_dir=None: repo_id)
    monkeypatch.setattr(orpheus_model.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None)
    model = OrpheusModel(
        "unused", snac_chunk_schedule=[2], snac_chunk_frames=8,
        snac_left_context_frames=4, snac_right_context_frames=1,
    )
    assert model.config.snac_chunk_schedule == (2,)
    assert {2, 10, 13} <= set(model.config.snac_window_frames()) and max(model.config.snac_window_frames()) == 13
    policy = _policy(model.config)
    assert not policy.is_ready(2 * TPF - 1) and policy.is_ready(2 * TPF)
    with pytest.raises(ValueError):
        OrpheusModelConfig(snac_chunk_schedule=(2,), snac_right_context_frames=2)


# ---------------------------------------------------------------------------
# The real SNAC architecture, tiny and with seeded random weights
# ---------------------------------------------------------------------------

def _tiny_snac() -> SNAC:
    """SNAC 24 kHz's decoder geometry (rates 8, 8, 4, 2; strides 4, 2, 1;
    depthwise), at 64 channels instead of 1024, with the noise block off."""
    torch.manual_seed(0)
    model = SNAC(
        sampling_rate=24000, encoder_dim=4, encoder_rates=[2, 4, 8, 8], decoder_dim=64,
        decoder_rates=[8, 8, 4, 2], attn_window_size=None, codebook_size=4096, codebook_dim=8,
        vq_strides=[4, 2, 1], noise=False, depthwise=True,
    )
    return model.eval().requires_grad_(False)


@pytest.fixture(scope="module")
def tiny() -> SNACDecoderSubmodule:
    return SNACDecoderSubmodule(_tiny_snac(), OrpheusModelConfig())


RECORDED = Path(__file__).parents[1] / "integration" / "data" / "orpheus_tokens.safetensors"


def _tiny_case(config: OrpheusModelConfig, case) -> torch.Tensor:
    if case == "recorded":
        # a real Orpheus generation (38 frames, EOS last)
        return load_file(str(RECORDED))["short"].long()
    return _tokens(config, case, seed=10 + case, eos=True)


def test_no_context_geometry_captures_only_decodable_windows():
    """With no context, a final flush can hold only the EOS token. Its window
    decodes as one frame, so no capture may be zero frames wide: SNAC's
    convolutions reject an empty input during warm-up."""
    config = OrpheusModelConfig(snac_chunk_schedule=(4,), snac_left_context_frames=0,
                                snac_right_context_frames=0)
    assert config.snac_window_frames() == [1, 2, 3, 4]
    submodule = SNACDecoderSubmodule(_tiny_snac(), config)
    for cfg in submodule.get_accelerator_graph_configs(torch.device("cpu")):
        assert cfg.additional_key_info == cfg.single_request_inputs.input_ids.shape[0] >= 1
        codes = submodule.preprocess("snac_chunk", None, [cfg.single_request_inputs])
        audio = submodule._decode(codes["codes_0"], codes["codes_1"], codes["codes_2"])
        assert audio.shape[-1] == cfg.additional_key_info * HOP


def test_no_context_stream_cut_at_a_chunk_boundary_ends_cleanly():
    """No context and no EOS (the token limit hit on a chunk boundary): the
    buffer drains to empty, so the final flush carries no tokens at all."""
    config = OrpheusModelConfig(snac_chunk_schedule=(4,), snac_left_context_frames=0,
                                snac_right_context_frames=0)
    submodule = SNACDecoderSubmodule(_NeighbourDecoder(), config)
    tokens = _tokens(config, 8)
    chunks = _stream(submodule, tokens)
    assert chunks[-1].numel() == 0
    assert sum(chunk.numel() for chunk in chunks) == 8 * HOP


@pytest.mark.parametrize("overrides", [
    dict(snac_chunk_schedule=(1000,)),
    dict(snac_chunk_frames=12),
    dict(snac_left_context_frames=12),
    dict(snac_right_context_frames=8, snac_chunk_schedule=(9,)),
])
def test_a_window_wider_than_supported_is_rejected(overrides):
    """One capture per window size per batch size: a 1000-frame chunk would
    capture 1000 window sizes, so the widest window is bounded."""
    with pytest.raises(ValueError, match="at most 16"):
        OrpheusModelConfig(**overrides)


def test_the_widest_supported_window_bounds_the_captures():
    config = OrpheusModelConfig(snac_chunk_frames=11)  # 3 + 2 + 11 frames
    windows = config.snac_window_frames()
    assert max(windows) <= 16 and len(windows) <= 16
    configs = SNACDecoderSubmodule(_NeighbourDecoder(), config).get_accelerator_graph_configs(torch.device("cpu"))
    assert len(configs) == len(windows)


@pytest.mark.parametrize("case", [3, 6, 25, "recorded"])
def test_tiny_snac_stream_is_close_to_whole_decode(tiny, case):
    tokens = _tiny_case(tiny.config, case)
    chunks = _stream(tiny, tokens)
    stream, whole = torch.cat(chunks), _whole(tiny, tokens)
    assert stream.numel() == whole.numel() == (tokens.numel() // TPF) * HOP
    assert _snr(whole, stream) > 50.0
    # every chunk boundary, +-50 ms
    boundary = 0
    for chunk in chunks[:-1]:
        boundary += chunk.numel()
        if chunk.numel():
            lo, hi = max(boundary - 1200, 0), min(boundary + 1200, whole.numel())
            assert _snr(whole[lo:hi], stream[lo:hi]) > 35.0, boundary
