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
    ).new_input_tensors
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
    ).new_input_tensors
    lens = out["audio_feature_lens"][0].tolist()
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
    out = model.process_prompt(
        None, ["text"] + ["audio"] * len(clips), ["text"],
        tensors={"audio_inputs": clips},
        prompt_parts=[PromptPart("text", "Transcribe.")] + [PromptPart("audio")] * len(clips),
    ).new_input_tensors
    audio = model.config.audio
    if "audio_feature_lens" not in out:  # nothing long enough to encode
        assert seconds == (0.06,)
        return
    lens = out["audio_feature_lens"][0].tolist()
    assert min(audio.pooled_tokens(n) for n in lens) > 0
    assert out["audio_positions"][0].numel() == sum(audio.pooled_tokens(n) for n in lens)
    assert out["audio_features"][0].shape[-1] == sum(lens)


@needs_ckpt
def test_kv_layout_covers_the_prompt_and_each_decode_step(model):
    """The LLM's layout: one span over every prompt row (placeholders included),
    written by the prefill walk the request takes, then a row per decode step."""
    from mstar.model.minicpm_o.config import DECODE_LOOP, LLM_KV
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
    assert model.get_loop_caps(max_output_tokens=7) == {DECODE_LOOP: 7}
    assert set(model.kv_labels()) == {LLM_KV}


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
    }
