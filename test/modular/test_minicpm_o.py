"""MiniCPM-o: prompt layout, walk selection and step declarations, without weights.

The resampler's sincos keys and the navit position buckets are checked
against upstream's own formulas on the CPU. Everything else needs the
checkpoint's tokenizer and processor (not its weights), so it skips unless
MINICPM_O_CKPT points at a MiniCPM-o 4.5 snapshot.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from mstar.model.minicpm_o.components.tts import next_history, windowed_frequency_penalty
from mstar.model.minicpm_o.components.vision import Resampler, navit_position_ids, slice_layout
from mstar.model.minicpm_o.config import ResamplerConfig, VisionConfig

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


@pytest.mark.parametrize("hw", [(26, 40), (32, 32), (7, 69), (70, 70), (100, 3)])
def test_navit_position_buckets_match_upstream(hw):
    assert torch.equal(navit_position_ids(*hw, 70), upstream_navit_ids(*hw, 70))


def test_tts_history_counts_codes_and_keeps_the_window():
    window = 4
    history = torch.tensor([[-1] * window + [0]] * 2)
    codes = [torch.tensor([10, 20]), torch.tensor([11, 21]), torch.tensor([12, 22]),
             torch.tensor([13, 23]), torch.tensor([14, 24])]
    for c in codes:
        history = next_history(history, c)
    assert history.tolist() == [[11, 12, 13, 14, 5], [21, 22, 23, 24, 5]]
    # empty slots count for nothing in the penalty; a repeated code twice
    logits = torch.ones(1, 30)
    out = windowed_frequency_penalty(logits, torch.tensor([[-1, -1, 7, 7]]), 2.0)
    assert out[0, 7] == 0.25 and (out[0, :7] == 1).all() and out[0, 0] == 1


def test_slice_layout_concatenates_slices_in_order():
    layout = slice_layout([(2, 3), (1, 2)], VisionConfig())
    assert layout.seq_lengths == (6, 2)
    assert layout.grid_coords.tolist()[:6] == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    assert layout.grid_coords.tolist()[6:] == [[0, 0], [0, 1]]


# --- with the checkpoint's tokenizer and processor -------------------------

@pytest.fixture(scope="module")
def model():
    from mstar.model.minicpm_o.minicpm_o_model import MiniCPMOModel

    return MiniCPMOModel(CKPT)


def an_image(h=448, w=640):
    torch.manual_seed(0)
    return torch.rand(3, h, w)


def a_clip(seconds=3.0):
    torch.manual_seed(1)
    return 0.1 * torch.randn(int(16000 * seconds))


@needs_ckpt
@pytest.mark.parametrize("max_slice_nums", [1, None])
def test_image_placeholders_line_up_with_slices(model, max_slice_nums):
    from mstar.model.multimodal import PromptPart

    out = model.process_prompt(
        None, ["image", "text"], ["text"],
        tensors={"image_inputs": [an_image()]},
        prompt_parts=[PromptPart("image"), PromptPart("text", "Describe it.")],
        max_slice_nums=max_slice_nums,
    )
    ids = out["text_inputs"][0]
    positions = out["image_positions"][0]
    tgt = out["image_tgt_sizes"][0]
    assert positions.numel() == tgt.shape[0] * model.config.resampler.num_queries
    unk = model.tokenizer.convert_tokens_to_ids("<unk>")
    assert (ids[positions] == unk).all()
    assert out["pixel_values"][0].shape[0] == int((tgt[:, 0] * tgt[:, 1]).sum())


@needs_ckpt
def test_audio_placeholders_line_up_with_pooled_tokens(model):
    from mstar.model.multimodal import PromptPart

    out = model.process_prompt(
        None, ["text", "audio"], ["text"],
        tensors={"audio_inputs": [a_clip()]},
        prompt_parts=[PromptPart("text", "Transcribe."), PromptPart("audio")],
    )
    lens = out["audio_feature_lens"][0].tolist()
    assert out["audio_positions"][0].numel() == sum(model.config.audio.pooled_tokens(n) for n in lens)
    # audio anywhere turns upstream's speech template on: the prompt ends on <|tts_bos|>
    assert out["text_inputs"][0][-1].item() == model.tokenizer.convert_tokens_to_ids("<|tts_bos|>")


@needs_ckpt
def test_spoken_reply_system_prompt_with_and_without_the_voice_clip(model):
    """By default a spoken reply's system message carries the voice clip (upstream's
    audio_assistant prompt); with ``voice_prompt=False`` it is the request's text system
    prompt. Both use the speech template, which ends the prompt on <|tts_bos|>."""
    from mstar.model.multimodal import PromptPart

    tts_bos = model.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
    parts = [PromptPart("text", "Say hello.")]
    voiced = model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=parts)
    assert voiced["audio_positions"][0].numel() > 0
    assert voiced["text_inputs"][0][-1].item() == tts_bos
    plain = model.process_prompt(
        None, ["text"], ["text", "audio"], prompt_parts=parts,
        voice_prompt=False, system_prompt="You are a helpful assistant.",
    )
    assert "audio_positions" not in plain
    assert plain["text_inputs"][0][-1].item() == tts_bos
    assert "You are a helpful assistant." in model.tokenizer.decode(plain["text_inputs"][0])
    with pytest.raises(ValueError, match="unknown MiniCPM-o voice"):
        model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=parts, voice_prompt=False, voice="nope")


@needs_ckpt
@pytest.mark.parametrize(
    "tensors, walk",
    [
        ({"text_inputs"}, "prefill_text"),
        ({"text_inputs", "pixel_values", "image_tgt_sizes", "image_positions"}, "prefill_image"),
        ({"text_inputs", "audio_features", "audio_feature_lens", "audio_positions"}, "prefill_audio"),
        ({"text_inputs", "pixel_values", "image_tgt_sizes", "image_positions",
          "audio_features", "audio_feature_lens", "audio_positions"}, "prefill_omni"),
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
def test_postprocess_drops_stop_tokens(model):
    ids = model.tokenizer.encode("Paris.", add_special_tokens=False) + list(model.config.stop_token_ids)
    assert model.postprocess(torch.tensor(ids), "text") == b"Paris."


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
        "Token2Wav": {"t2w_state"},
    }
