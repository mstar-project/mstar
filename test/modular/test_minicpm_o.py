"""MiniCPM-o: prompt layout, walk selection and step declarations, without weights.

The resampler's sincos keys and the navit position buckets are checked
against upstream's own formulas on the CPU. Everything else needs the
checkpoint's config and tokenizer (not its weights), so it skips unless
MINICPM_O_CKPT points at a MiniCPM-o 4.5 snapshot.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.model.components.qwen3_lm import Qwen3LMConfig
from mstar.model.minicpm_o.components.vision import Resampler, navit_position_ids, slice_layout
from mstar.model.minicpm_o.config import (
    DECODE_LOOP,
    LLM_SAMPLER,
    PREFILL_WALKS,
    AudioConfig,
    MiniCPMOConfig,
    ResamplerConfig,
    VisionConfig,
)

CKPT = os.environ.get("MINICPM_O_CKPT")
needs_ckpt = pytest.mark.skipif(not CKPT, reason="set MINICPM_O_CKPT to a MiniCPM-o 4.5 snapshot")


# --- upstream formulas, copied for reference -------------------------------

def upstream_sincos_table(embed_dim: int, h: int, w: int) -> np.ndarray:
    grid = np.stack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)), axis=0)

    def axis(dim, pos):
        omega = np.arange(dim // 2, dtype=np.float32)
        omega /= dim / 2.0
        omega = 1.0 / 10000 ** omega
        out = np.einsum("hw,d->hwd", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=-1)

    return np.concatenate([axis(embed_dim // 2, grid[0]), axis(embed_dim // 2, grid[1])], axis=-1)


def upstream_navit_ids(h: int, w: int, side: int) -> torch.Tensor:
    # h and w are rows of the processor's int32 tgt_sizes
    h, w = torch.tensor(h, dtype=torch.int32), torch.tensor(w, dtype=torch.int32)
    boundaries = torch.arange(1 / side, 1.0, 1 / side)
    bh = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / h), boundaries, right=True)
    bw = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / w), boundaries, right=True)
    return (bh[:, None] * side + bw).flatten()


# --- CPU, no checkpoint ----------------------------------------------------

@pytest.mark.parametrize("hw", [(26, 40), (36, 28), (1, 1), (70, 70), (5, 100)])
def test_sincos_keys_match_upstream_table(hw):
    """Per-patch keys, computed from (row, col), equal upstream's precomputed
    table at those cells, including past its 70x70 default size."""
    h, w = hw
    resampler = Resampler(ResamplerConfig(embed_dim=256, kv_dim=16))
    coords = slice_layout([(h, w)], VisionConfig()).grid_coords
    got = resampler.sincos(coords)
    want = torch.from_numpy(upstream_sincos_table(256, h, w)).reshape(h * w, -1)
    assert torch.allclose(got, want, atol=1e-6, rtol=0)


@pytest.mark.parametrize("hw", [(26, 40), (32, 32), (7, 69), (70, 70), (100, 3), (1, 25), (13, 49)])
def test_navit_position_buckets_match_upstream(hw):
    assert torch.equal(navit_position_ids(*hw, 70), upstream_navit_ids(*hw, 70))


def test_slice_layout_concatenates_slices_in_order():
    layout = slice_layout([(2, 3), (1, 2)], VisionConfig())
    assert layout.seq_lengths == (6, 2)
    assert layout.grid_coords.tolist()[:6] == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    assert layout.grid_coords.tolist()[6:] == [[0, 0], [0, 1]]


def a_config(**kwargs) -> MiniCPMOConfig:
    return MiniCPMOConfig(
        llm=Qwen3LMConfig(), vision=VisionConfig(), resampler=ResamplerConfig(), audio=AudioConfig(), **kwargs,
    )


def a_bare_model(**attrs):
    """A MiniCPMOModel without its checkpoint: what reads only ``config``."""
    from mstar.model.minicpm_o.minicpm_o_model import MiniCPMOModel

    model = object.__new__(MiniCPMOModel)
    model.config = a_config()
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


@pytest.mark.parametrize("walk", sorted(PREFILL_WALKS))
def test_prefill_token_counts_toward_the_budget(walk):
    """The prefill samples the reply's first token: a stop id there, or a
    one-token budget, ends the reply, and decode runs one step fewer."""
    from mstar.model.minicpm_o.submodules import LLMSubmodule
    from mstar.model.submodule_base import END_PARTITION

    llm = LLMSubmodule(torch.nn.Identity(), a_config(stop_token_ids=(7,)))

    def stop(token, max_tokens):
        info = SimpleNamespace(
            graph_walk=walk, max_tokens=max_tokens, step_metadata={}, dynamic_loop_iter_counts={},
            resource_configs={LLM_SAMPLER: SimpleNamespace(ignore_eos=False)},
        )
        return llm.check_stop("r", info, {"new_token": [torch.tensor([token])]})

    assert stop(7, 8) == {END_PARTITION}
    assert stop(3, 1) == {END_PARTITION}
    assert stop(3, 8) == set()
    model = a_bare_model()
    assert model.get_loop_caps(max_output_tokens=8)[DECODE_LOOP] == 7
    assert model.get_loop_caps(max_output_tokens=1)[DECODE_LOOP] == 1


def test_tts_code_budget_is_its_loop_cap():
    """The TTS prefill samples the first code, so its decode loop runs one fewer than
    upstream's max_new_tokens: a reply's codes stop there, or at the EOS code."""
    from mstar.model.minicpm_o.components.tts import TTSConfig
    from mstar.model.minicpm_o.config import TTS_SAMPLER, TTSSampling
    from mstar.model.minicpm_o.minicpm_o_model import TTS_DECODE, TTS_DECODE_LOOP
    from mstar.model.minicpm_o.submodules import TTSSubmodule

    sampling = TTSSampling()
    assert a_bare_model().get_loop_caps()[TTS_DECODE_LOOP] == sampling.max_new_tokens - 1
    tts_config = TTSConfig()
    tts = TTSSubmodule(SimpleNamespace(config=tts_config), a_config(), sampling)

    def stop(code, iters):
        info = SimpleNamespace(
            graph_walk=TTS_DECODE, max_tokens=8, step_metadata={},
            dynamic_loop_iter_counts={TTS_DECODE_LOOP: iters},
            resource_configs={TTS_SAMPLER: SimpleNamespace(ignore_eos=False)},
        )
        return tts.check_stop("r", info, {"tts_code": [torch.tensor([code])]})

    assert stop(5, sampling.max_new_tokens - 2) == set()
    assert stop(tts_config.eos_code, 3) == {TTS_DECODE_LOOP}


@pytest.mark.parametrize("knob, value", [
    ("max_slice_nums", "9"), ("max_slice_nums", 0), ("max_slice_nums", 1000), ("max_slice_nums", True),
    ("enable_thinking", "false"),
])
def test_bad_request_knobs_are_refused(knob, value):
    with pytest.raises(ValueError, match=knob):
        a_bare_model().process_prompt("Hi.", ["text"], ["text"], **{knob: value})


# --- with the checkpoint's config and tokenizer ---------------------------

@pytest.fixture(scope="module")
def model():
    from mstar.model.minicpm_o.minicpm_o_model import MiniCPMOModel

    return MiniCPMOModel(CKPT)


def an_image(h=448, w=640):
    return torch.rand(3, h, w, generator=torch.Generator().manual_seed(0))


def a_clip(seconds=3.0):
    return 0.1 * torch.randn(int(16000 * seconds), generator=torch.Generator().manual_seed(1))


@needs_ckpt
@pytest.mark.parametrize("max_slice_nums", [1, None])
def test_image_placeholders_line_up_with_slices(model, max_slice_nums):
    from mstar.model.multimodal import PromptPart

    prompt = model.process_prompt(
        None, ["image", "text"], ["text"],
        tensors={"image_inputs": [an_image()]},
        prompt_parts=[PromptPart("image"), PromptPart("text", "Describe it.")],
        max_slice_nums=max_slice_nums,
    )
    out = prompt.new_input_tensors
    ids = out["text_inputs"][0]
    positions = out["image_positions"][0]
    tgt = torch.tensor(prompt.metadata["image_tgt_sizes"])
    assert positions.numel() == tgt.shape[0] * model.config.resampler.num_queries
    unk = model.tokenizer.convert_tokens_to_ids("<unk>")
    assert (ids[positions] == unk).all()
    assert out["pixel_values"][0].shape[0] == int((tgt[:, 0] * tgt[:, 1]).sum())


@needs_ckpt
def test_chat_turns_keep_their_roles(model):
    """A multi-turn chat renders as upstream's template renders its messages."""
    from mstar.model.multimodal import PromptPart

    turns = [("system", "Be brief."), ("user", "Hi."), ("assistant", "Hello."), ("user", "Name a colour.")]
    out = model.process_prompt(
        None, ["text"] * len(turns), ["text"], prompt_parts=[PromptPart("text", t, role=r) for r, t in turns],
    ).new_input_tensors
    want = model.tokenizer.apply_chat_template(
        [{"role": r, "content": t} for r, t in turns], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    assert out["text_inputs"][0].tolist() == model.tokenizer.encode(want)


@needs_ckpt
def test_audio_placeholders_line_up_with_pooled_tokens(model):
    from mstar.model.multimodal import PromptPart

    prompt = model.process_prompt(
        None, ["text", "audio"], ["text"],
        tensors={"audio_inputs": [a_clip()]},
        prompt_parts=[PromptPart("text", "Transcribe."), PromptPart("audio")],
    )
    out = prompt.new_input_tensors
    lens = prompt.metadata["audio_feature_lens"]
    assert out["audio_positions"][0].numel() == sum(model.config.audio.pooled_tokens(n) for n in lens)
    # audio anywhere turns upstream's speech template on: the prompt ends on <|tts_bos|>
    assert out["text_inputs"][0][-1].item() == model.tokenizer.convert_tokens_to_ids("<|tts_bos|>")


@needs_ckpt
@pytest.mark.parametrize("seconds", [(30.05,), (60.01,), (0.06,), (1.05, 1.05), (0.06, 2.0)])
def test_audio_placeholders_line_up_for_any_clips(model, seconds):
    """Pieces that pool to nothing (a sliver past a 30 s cut, a clip under
    ~0.1 s) are left out, and clips of one message are encoded one by one, as
    their placeholders are sized."""
    from mstar.model.multimodal import PromptPart

    clips = [a_clip(s) for s in seconds]
    prompt = model.process_prompt(
        None, ["text"] + ["audio"] * len(clips), ["text"],
        tensors={"audio_inputs": clips},
        prompt_parts=[PromptPart("text", "Transcribe.")] + [PromptPart("audio")] * len(clips),
    )
    out = prompt.new_input_tensors
    audio = model.config.audio
    if "audio_feature_lens" not in prompt.metadata:  # nothing long enough to encode
        assert seconds == (0.06,)
        return
    lens = prompt.metadata["audio_feature_lens"]
    assert min(audio.pooled_tokens(n) for n in lens) > 0
    assert out["audio_positions"][0].numel() == sum(audio.pooled_tokens(n) for n in lens)
    assert out["audio_features"][0].shape[-1] == sum(lens)


@needs_ckpt
def test_spoken_reply_system_prompt_with_and_without_the_voice_clip(model):
    """By default a spoken reply's system message carries the voice clip (upstream's
    audio_assistant prompt); with ``voice_prompt=False`` it is the request's text system
    prompt. Both use the speech template, which ends the prompt on <|tts_bos|>."""
    from mstar.model.multimodal import PromptPart

    tts_bos = model.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
    parts = [PromptPart("text", "Say hello.")]
    voiced = model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=parts).new_input_tensors
    assert voiced["audio_positions"][0].numel() > 0
    assert voiced["text_inputs"][0][-1].item() == tts_bos
    plain = model.process_prompt(
        None, ["text"], ["text", "audio"], prompt_parts=parts,
        voice_prompt=False, system_prompt="You are a helpful assistant.",
    ).new_input_tensors
    assert "audio_positions" not in plain
    assert plain["text_inputs"][0][-1].item() == tts_bos
    assert "You are a helpful assistant." in model.tokenizer.decode(plain["text_inputs"][0])
    with pytest.raises(ValueError, match="unknown MiniCPM-o voice"):
        model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=parts, voice_prompt=False, voice="nope")


@needs_ckpt
def test_spoken_reply_text_fits_the_tts(model):
    """The TTS speaks the whole reply within its positions: a spoken reply's text
    budget defaults under that, and asking for more is refused (a 400). The default
    sets no length kwarg, so admission still fits the budget to a long prompt."""
    from mstar.model.multimodal import PromptPart

    parts = [PromptPart("text", "Say hello.")]
    limit = model.speech_max_text_tokens
    assert limit + 2 + 2047 == model.tts_config.max_position_embeddings
    out = model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=parts)
    assert not any(k in out.metadata for k in model.LENGTH_KWARGS)
    assert model.get_max_output_tokens(**out.metadata) == min(2048, limit)
    assert model.get_loop_caps(**out.metadata) == {DECODE_LOOP: min(2048, limit) - 1}
    with pytest.raises(ValueError, match="spoken reply"):
        model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=parts, max_output_tokens=limit + 1)
    # text replies are not capped
    text = model.process_prompt(None, ["text"], ["text"], prompt_parts=parts).metadata
    assert model.get_max_output_tokens(**text, max_output_tokens=8192) == 8192


@needs_ckpt
def test_encoders_lay_out_segments_from_the_prompts_step_metadata(model):
    """The slice grids and mel lengths reach the encoders as step metadata,
    and their segments cover the patches and frames the prompt carries."""
    from mstar.model.minicpm_o.submodules import AudioEncoderSubmodule, VisionEncoderSubmodule
    from mstar.model.multimodal import PromptPart

    prompt = model.process_prompt(
        None, ["image", "audio", "text"], ["text"],
        tensors={"image_inputs": [an_image()], "audio_inputs": [a_clip()]},
        prompt_parts=[PromptPart("image"), PromptPart("audio"), PromptPart("text", "What is this?")],
    )
    out = prompt.new_input_tensors
    signals = {name: [None] for name in out}
    args = model.get_initial_forward_pass_args("default", ["text"], ["text"], signals, model_kwargs=prompt.metadata)
    assert args.full_metadata.graph_walk == "prefill_omni"
    info = SimpleNamespace(step_metadata=args.step_metadata)
    vision = VisionEncoderSubmodule(torch.nn.Identity(), model.config).prepare_inputs(
        "prefill_omni", info, {"pixel_values": out["pixel_values"]},
    )
    assert sum(vision.kwargs["seq_lengths"]) == vision.input_seq_len == out["pixel_values"][0].shape[0]
    audio = AudioEncoderSubmodule(torch.nn.Identity(), model.config).prepare_inputs(
        "prefill_omni", info, {"audio_features": out["audio_features"]},
    )
    assert sum(audio.kwargs["mel_lens"]) == out["audio_features"][0].shape[-1]


@needs_ckpt
def test_kv_layout_covers_the_prompt_and_each_decode_step(model):
    """The LLM's layout: one span over every prompt row (placeholders included),
    written by the prefill walk the request takes, then a row per decode step."""
    from mstar.model.base import LabelLayout
    from mstar.model.minicpm_o.config import LLM_KV, TTS_KV
    from mstar.model.multimodal import PromptPart

    out = model.process_prompt(
        None, ["image", "text"], ["text"],
        tensors={"image_inputs": [an_image()]},
        prompt_parts=[PromptPart("image"), PromptPart("text", "Describe it.")],
    )
    (span,) = out.metadata["kv_layout"][LLM_KV]["main"].spans
    rows = out.new_input_tensors["text_inputs"][0].numel()
    assert (span.walk, span.length, span.advance) == ("prefill_image", rows, rows)
    assert out.metadata["kv_layout"][LLM_KV]["main"].growth == ((DECODE_LOOP, 1),)
    assert set(model.kv_labels()) == {LLM_KV, TTS_KV}
    # a text reply opens the TTS's pool with nothing in it; a spoken one prefills its
    # condition (two rows and a row per LLM step) and then writes a row per code
    assert out.metadata["kv_layout"][TTS_KV]["main"] == LabelLayout()
    spoken = model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=[PromptPart("text", "Hi.")])
    tts = spoken.metadata["kv_layout"][TTS_KV]["main"]
    assert [(s.walk, s.length) for s in tts.spans] == [("tts_prefill", 2)]
    assert tts.growth == ((DECODE_LOOP, 1), ("tts_decode_loop", 1))


@needs_ckpt
def test_max_seq_len_sizes_the_kv_and_the_decode_loop():
    from mstar.engine.resources import KVSpec
    from mstar.model.minicpm_o.config import LLM_KV
    from mstar.model.minicpm_o.minicpm_o_model import MiniCPMOModel

    model = MiniCPMOModel(CKPT, max_seq_len=4096)
    (kv,) = [s for s in model.get_node_resources() if isinstance(s, KVSpec) and s.resource_key == LLM_KV]
    assert kv.config.max_seq_len == 4096
    assert model.get_graph_walk_graphs()["decode"].max_iters == 4096
    with pytest.raises(ValueError, match="max_seq_len"):
        MiniCPMOModel(CKPT, max_seq_len=model.config.llm.max_position_embeddings + 1)


@needs_ckpt
@pytest.mark.parametrize(
    "tensors, walk",
    [
        ({"text_inputs"}, "prefill_text"),
        ({"text_inputs", "pixel_values", "image_positions"}, "prefill_image"),
        ({"text_inputs", "audio_features", "audio_positions"}, "prefill_audio"),
        ({"text_inputs", "pixel_values", "image_positions", "audio_features", "audio_positions"}, "prefill_omni"),
    ],
)
def test_initial_walk_feeds_every_input_its_node_reads(model, tensors, walk):
    from mstar.graph.base import TensorPointerInfo

    signals = {name: [TensorPointerInfo.__new__(TensorPointerInfo)] for name in tensors}
    args = model.get_initial_forward_pass_args("default", ["text"], ["text"], signals)
    assert args.full_metadata.graph_walk == walk
    fed = {(e.next_node, e.name) for e in args.inputs}
    produced = {"vision_embeds", "audio_embeds"}
    for name, node in model.get_graph_walk_graphs()[walk].get_nodes().items():
        assert {(name, n) for n in node.input_names if n not in produced} <= fed


@needs_ckpt
def test_text_streamed_a_token_at_a_time_reassembles(model):
    """Each decode step's token reaches the client alone: a character split
    across tokens must join back whole, and the stop tokens are dropped."""
    text = "Paris 🐱猫."
    ids = model.tokenizer.encode(text, add_special_tokens=False) + list(model.config.stop_token_ids)
    chunks = model.postprocess_inline(ids, "text", [None] * len(ids))
    assert b"".join(chunks).decode("utf-8") == text
    assert model.postprocess(torch.tensor(ids), "text") == text.encode("utf-8")


@needs_ckpt
def test_every_node_gets_only_its_resources(model):
    by_node: dict[str, set[str]] = {}
    for spec in model.get_node_resources():
        for node in spec.nodes:
            by_node.setdefault(node, set()).add(spec.resource_key)
    assert by_node == {
        "LLM": {"llm_kv", "llm_attn", "llm_pos", "llm_sampler"},
        "vision_encoder": {"vision_attn", "resampler_attn"},
        "audio_encoder": {"audio_attn"},
        "TTS": {"tts_kv", "tts_attn", "tts_pos", "tts_sampler"},
        "Token2Wav": {"t2w_state", "t2w_enc1_kv", "t2w_enc2_kv", "t2w_dit_kv"},
    }
