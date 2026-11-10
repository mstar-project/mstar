"""MiniCPM-o on the VLM scaffold: prompt layout, walks and step declarations, without weights.

The resampler's sincos keys, the navit position buckets and the image
slicing are checked against upstream's own formulas on the CPU. What needs
the checkpoint's real tokenizer (not its weights) skips unless
MINICPM_O_CKPT points at a MiniCPM-o 4.5 snapshot.
"""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.model.components.qwen3_lm import Qwen3LMConfig
from mstar.model.keys import KV_CACHE, SAMPLER
from mstar.model.minicpm_o.components.vision import Resampler, navit_position_ids, slice_layout
from mstar.model.minicpm_o.config import (
    AUDIO_ENCODER,
    DECODE_SPOKEN,
    IMAGE_ENCODER,
    AudioConfig,
    MiniCPMOConfig,
    ResamplerConfig,
    VisionConfig,
)
from mstar.model.multimodal import MEDIA_LAYOUT

DECODE_LOOP = "decode_loop"
# the scaffold's prefill walks, named by the two encoders
PREFILL_WALKS = ("prefill_text", "prefill_image", "prefill_audio", "prefill_image_audio")

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


def a_model(tmp_path, **kwargs):
    """A MiniCPMOModel off a shrunk config.json, a stand-in tokenizer and a 2 s bundled
    voice clip: no weights, and no chat template."""
    import soundfile as sf
    from test_trust_remote_code import MINICPM_O_CONFIG, _tokenizer_dir

    from mstar.model.minicpm_o.minicpm_o_model import VOICES, MiniCPMOModel

    _tokenizer_dir(tmp_path, {}, {"config.json": json.dumps(MINICPM_O_CONFIG)})
    (tmp_path / "assets").mkdir(exist_ok=True)
    sf.write(tmp_path / "assets" / VOICES["default"], a_clip(2.0).numpy(), 16000)
    return MiniCPMOModel(str(tmp_path), **kwargs)


class _ChatTokenizer:
    """MiniCPM-o's special tokens, each its own id (parsed in any text, as the
    real tokenizer's), every other byte its own id, and a plain chat template:
    enough to render and place a prompt without the checkpoint."""

    SPECIAL = (
        "<unk>", "<image>", "</image>", "<slice>", "</slice>", "<image_id>", "</image_id>",
        "<|audio_start|>", "<|audio_end|>", "<|listen|>", "<|im_start|>", "<|im_end|>",
        "<|endoftext|>", "<|tts_bos|>", "<|tts_eos|>",
    )

    def convert_tokens_to_ids(self, token):
        return 1000 + self.SPECIAL.index(token)

    def encode(self, text):
        pieces = re.split("(" + "|".join(map(re.escape, self.SPECIAL)) + ")", text)
        return [i for p in pieces for i in ([self.convert_tokens_to_ids(p)] if p in self.SPECIAL else p.encode())]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, use_tts_template, enable_thinking):
        turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return turns + "<|im_start|>assistant\n" + ("<|tts_bos|>" if use_tts_template else "")


@pytest.fixture(params=["stub", pytest.param("ckpt", marks=needs_ckpt)])
def chat_model(request, tmp_path, monkeypatch):
    """A MiniCPMOModel that renders and places prompts: ``a_model`` with a stand-in
    tokenizer of MiniCPM-o's special tokens, and the checkpoint's where it is cached."""
    if request.param == "ckpt":
        return request.getfixturevalue("model")
    from mstar.model.minicpm_o import minicpm_o_model

    stub = SimpleNamespace(from_pretrained=lambda path: _ChatTokenizer())
    monkeypatch.setattr(minicpm_o_model, "Qwen2Tokenizer", stub)
    return a_model(tmp_path)


def a_bare_model(**attrs):
    """A MiniCPMOModel without its checkpoint: what reads only ``config``."""
    from mstar.model.minicpm_o.minicpm_o_model import MiniCPMOModel

    model = object.__new__(MiniCPMOModel)
    model.config = a_config()
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def an_llm(model=None, decoder=None, **kwargs):
    """The LLM node, without weights; given a ``model``, built through its
    `get_submodule`, as a worker builds it."""
    from mstar.model.minicpm_o.submodules import MiniCPMOLLM

    llm = MiniCPMOLLM(decoder or torch.nn.Identity(), a_config(**kwargs), (IMAGE_ENCODER, AUDIO_ENCODER))
    if model is None:
        return llm
    model._create_submodule = lambda *args: llm
    return model.get_submodule("LLM")


@pytest.mark.parametrize("walk", PREFILL_WALKS)
def test_prefill_token_counts_toward_the_budget(walk):
    """Each prefill walk the two encoders name samples the reply's first token:
    a stop id there, or a one-token budget, ends the reply, and decode runs one
    step fewer."""
    from mstar.conductor.request_info import FINAL_PREFILL
    from mstar.model.submodule_base import END_PARTITION

    llm = an_llm(stop_token_ids=(7,))

    def stop(token, max_tokens):
        info = SimpleNamespace(
            graph_walk=walk, max_tokens=max_tokens, step_metadata={FINAL_PREFILL: True},
            dynamic_loop_iter_counts={}, resource_configs={SAMPLER: SimpleNamespace(ignore_eos=False)},
        )
        return llm.check_stop("r", info, {"new_token": [torch.tensor([token])]})

    assert stop(7, 8) == {END_PARTITION}
    assert stop(3, 1) == {END_PARTITION}
    assert stop(3, 8) == set()
    model = a_bare_model()
    assert model.get_loop_caps(max_output_tokens=8)[DECODE_LOOP] == 7
    assert model.get_loop_caps(max_output_tokens=1)[DECODE_LOOP] == 1


@pytest.mark.parametrize("speech", [False, True])
def test_a_text_replys_decode_steps_loop_back(tmp_path, speech):
    """A text reply's decode outputs only its token, which the LLM keeps on the device,
    so its steps run rows-only (device loop-back). A spoken reply's decode also hands
    the TTS its condition, which goes through the host. Either way the client's token
    rides inline."""
    from mstar.model.minicpm_o.minicpm_o_model import MAIN

    model = a_model(tmp_path)
    outputs = ["text", "audio"] if speech else ["text"]
    walk = model.walk_plan(MAIN, ["text"], outputs, {"text_inputs": [None]}, {})[1].walk
    signals = {edge.name for edge in model.get_graph_walk_graphs()[walk].section.outputs}
    llm = an_llm(model)
    llm.node_resources = {SAMPLER: SimpleNamespace(has_slot_masters=True)}
    # the worker's test for a rows-only step (``Worker._all_on_device``)
    assert (signals <= llm.device_loopback_signals(walk)) == (not speech)
    assert llm.inline_client_signals(walk) == {"text_inputs": "new_token"}


def test_a_spoken_reply_plans_the_tts_after_its_text_and_streams_to_the_vocoder(tmp_path):
    """A spoken reply decodes on ``decode_spoken``, then the TTS prefills over it and
    decodes on the prefill's code; Token2Wav runs only for it."""
    from mstar.conductor.request_info import FINAL_PREFILL
    from mstar.model.minicpm_o.config import TTS_DECODE, TTS_PREFILL
    from mstar.model.minicpm_o.minicpm_o_model import DEFAULT_VOICE, MAIN, T2W_CHUNK, TOKEN2WAV

    model = a_model(tmp_path)
    plan = model.walk_plan(MAIN, ["text"], ["text", "audio"], {"text_inputs": [None]}, {})
    assert [(s.walk, dict(s.bind)) for s in plan] == [
        ("prefill_text", {}),
        (DECODE_SPOKEN, {"text_inputs": "new_token"}),
        (TTS_PREFILL, {}),
        (TTS_DECODE, {}),
    ]
    assert plan[0].step_metadata[FINAL_PREFILL]
    assert model.walk_plan(TOKEN2WAV, ["text"], ["text"], {}, {}) == []
    [chunk] = model.walk_plan(TOKEN2WAV, ["text"], ["text", "audio"], {}, {})
    assert (chunk.walk, dict(chunk.step_metadata)) == (T2W_CHUNK, {"voice": DEFAULT_VOICE})
    graphs = model.get_graph_walk_graphs()
    for step in plan:
        inputs = set().union(*(node.input_names for node in graphs[step.walk].get_nodes().values()))
        assert set(step.bind) <= inputs


class _Decoder(torch.nn.Module):
    """A stand-in decoder module: a row's hidden is its id and the id plus one."""

    embed_dtype = torch.float32

    def embed(self, ids):
        return torch.stack([ids.float(), ids.float() + 1], dim=-1)

    def forward(self, embeds, label):
        return embeds

    def logits(self, hidden):
        return hidden


def test_the_spoken_decode_hands_the_tts_each_rows_id_and_hidden(tmp_path):
    """Only ``decode_spoken``'s graph routes each row's input id and hidden, which
    the TTS prefill reads. The LLM, its graph's decode and hidden walks bound by
    ``get_submodule``, returns them from the one decode capture both decode walks
    replay; a text reply's row drops them, and each of the step's spoken rows
    keeps its own copy for its loop to accumulate, which holds no other row alive."""
    from mstar.model.components.sampled_token_decoder import ROW_HIDDEN, ROW_IDS
    from mstar.model.submodule_base import BatchedModelOutput

    model = a_model(tmp_path)
    graphs = model.get_graph_walk_graphs()
    walks = model.decoder_walks()
    assert walks == ({"decode", DECODE_SPOKEN}, {DECODE_SPOKEN})
    assert set(graphs["tts_prefill"].input_names) == {ROW_IDS, ROW_HIDDEN}
    llm = an_llm(model, _Decoder())
    (decode,) = [c for c in llm.get_accelerator_graph_configs(torch.device("cpu")) if c.capture_graph_walk == "decode"]
    assert DECODE_SPOKEN in decode.replay_graph_walks

    sampler = SimpleNamespace(sample=lambda rids, logits: logits[:, 0].long())
    # a mixed step of one-token rows: each row is its own last token
    attn = SimpleNamespace(select_last_hidden=lambda rows: rows)
    walks = {"text": "decode", "spoken": DECODE_SPOKEN, "spoken2": DECODE_SPOKEN}
    engine_inputs = SimpleNamespace(request_ids=list(walks), resources={SAMPLER: sampler, "attn": attn})
    rows = llm.forward_batched("mixed", engine_inputs, input_ids=torch.tensor([5, 6, 7])).row_outputs
    assert rows[ROW_IDS].tolist() == [5, 6, 7]
    assert rows[ROW_HIDDEN].tolist() == [[5.0, 6.0], [6.0, 7.0], [7.0, 8.0]]
    # the engine's per-request outputs: views of the step's one batch clone
    outputs = BatchedModelOutput(per_rid_outputs={
        rid: {name: [rows[name][i:i + 1]] for name in rows} for i, rid in enumerate(walks)
    })
    infos = {rid: SimpleNamespace(graph_walk=walk) for rid, walk in walks.items()}
    llm.batch_postprocess(outputs, infos, [None] * len(walks))
    assert ROW_IDS not in outputs.get("text") and ROW_HIDDEN not in outputs.get("text")
    spoken = [outputs.get(rid)[ROW_HIDDEN][0] for rid in ("spoken", "spoken2")]
    assert [h.tolist() for h in spoken] == [[[6.0, 7.0]], [[7.0, 8.0]]]
    assert [outputs.get(rid)[ROW_IDS][0].tolist() for rid in ("spoken", "spoken2")] == [[6], [7]]
    # each row its own copy, apart from the batch clone and from its step-mates
    ptrs = {h.untyped_storage().data_ptr() for h in (*spoken, rows[ROW_HIDDEN])}
    assert len(ptrs) == 3 and spoken[0].untyped_storage().nbytes() == spoken[0].nbytes


def test_tts_code_budget_is_its_loop_cap():
    """The TTS prefill samples the first code, so its decode loop runs one fewer than
    upstream's max_new_tokens: a reply's codes stop there, or at the EOS code."""
    from mstar.model.minicpm_o.components.tts import TTSConfig
    from mstar.model.minicpm_o.config import TTS_DECODE, TTS_DECODE_LOOP, TTS_SAMPLER, TTSSampling
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


def test_min_p_is_served(tmp_path):
    """Upstream's ``chat`` hands ``min_p`` to HF ``generate``, which honours it."""
    from mstar.engine.resources import SamplerSpec

    model = a_model(tmp_path)
    [spec] = [s for s in model.get_node_resources() if isinstance(s, SamplerSpec) and s.resource_key == SAMPLER]
    model.get_request_resource_configs(["text"], ["text"], {"min_p": 0.05})[SAMPLER].validate(spec)


def test_an_attachment_tag_in_the_users_text_is_refused():
    from mstar.model.minicpm_o import processor

    with pytest.raises(ValueError, match="attachment tags"):
        processor.expand_tags(f"{processor.IMAGE_TAG} {processor.IMAGE_TAG}", ["<image>"], [])


def test_max_seq_len_sizes_the_kv_and_the_decode_loop(tmp_path):
    from mstar.engine.resources import KVSpec
    from mstar.model.minicpm_o.minicpm_o_model import MiniCPMOModel

    model = a_model(tmp_path, max_seq_len=4096)
    (kv,) = [s for s in model.get_node_resources() if isinstance(s, KVSpec) and s.resource_key == KV_CACHE]
    assert kv.config.max_seq_len == 4096
    assert {model.get_graph_walk_graphs()[w].max_iters for w in ("decode", DECODE_SPOKEN)} == {4096}
    with pytest.raises(ValueError, match="max_seq_len"):
        MiniCPMOModel(str(tmp_path), max_seq_len=model.config.llm.max_position_embeddings + 1)


def test_the_voice_clip_takes_the_request_clips_mel(tmp_path):
    """A spoken reply's prompt holds the bundled voice clip as its first audio, so the
    clip goes through the intake that the request's own clips (torch tensors) do."""
    model = a_model(tmp_path)
    prepared = model._audio_intake(model.voice_audio(None), {})
    features = prepared.inputs["audio_features"]
    assert features.shape == (model.config.audio.num_mel_bins, sum(prepared.grids[0]))
    assert torch.isfinite(features.float()).all()


@pytest.mark.parametrize("as_turn", [False, True])
def test_a_spoken_requests_own_system_prompt_is_refused(tmp_path, as_turn):
    """A spoken reply's system message is its voice prompt, so a request's own (the kwarg
    or a system turn) is refused, not dropped."""
    from mstar.model.multimodal import PromptPart

    parts, kwargs = [PromptPart("text", "Say hello.")], {"system_prompt": "Be brief."}
    if as_turn:
        parts, kwargs = [PromptPart("text", "Be brief.", role="system"), *parts], {}
    with pytest.raises(ValueError, match="voice_prompt"):
        a_model(tmp_path).process_prompt(None, ["text"], ["text", "audio"], prompt_parts=parts, **kwargs)


@pytest.mark.parametrize("size, grid, slices, placeholder", [
    # (width, height) -> the (cols, rows) grid, each slice's (h, w) in patches
    # (the source, then the crops) and the placeholder text, worked by hand
    # from upstream's get_sliced_grid, find_best_resize and get_refine_size
    # tall: the source is 316x633 px, rounded to 322x630
    ((500, 1000), (1, 2), [(45, 23), (32, 32), (32, 32)], "{h}<slice>{u}</slice>\n<slice>{u}</slice>"),
    # 1x2 and 2x1 tie; the first found wins. Each crop is 600x300 px, sized to 632x316, then 630x322
    ((600, 600), (1, 2), [(32, 32), (23, 45), (23, 45)], "{h}<slice>{u}</slice>\n<slice>{u}</slice>"),
    ((896, 896), (2, 2), [(32, 32)] * 5, (
        "{h}<slice>{u}</slice><slice>{u}</slice>\n<slice>{u}</slice><slice>{u}</slice>"
    )),
    # small: whole, upscaled to 547x365 px, rounded to 546x364
    ((300, 200), None, [(26, 39)], "{h}"),
])
def test_image_slicing_matches_upstream(size, grid, slices, placeholder):
    from PIL import Image

    from mstar.model.minicpm_o import processor

    # red rises left to right and green top to bottom, so a crop's colour says where it came from
    w, h = size
    ramp = np.zeros((h, w, 3), dtype=np.uint8)
    ramp[..., 0], ramp[..., 1] = np.linspace(0, 255, w)[None, :], np.linspace(0, 255, h)[:, None]
    images, got = processor.slice_image(Image.fromarray(ramp), processor.MAX_SLICE_NUMS, 14)
    assert got == grid
    assert [processor.slice_patches(image, 14)[1] for image in images] == slices
    # the crops go row by row, as upstream's split_to_patches
    where = [np.asarray(image, dtype=float).mean(axis=(0, 1))[[1, 0]].tolist() for image in images[1:]]
    assert where == sorted(where)
    # one <unk> a run, which intake expands to the slice's 64 queries
    want = placeholder.format(h="<image_id>3</image_id><image><unk></image>", u="<unk>")
    assert processor.image_placeholder(3, grid) == want
    assert processor.image_runs(grid) == len(slices)


# --- with the checkpoint's config and tokenizer ---------------------------

@pytest.fixture(scope="module")
def model():
    from mstar.model.minicpm_o.minicpm_o_model import MiniCPMOModel

    return MiniCPMOModel(CKPT)


def an_image(h=448, w=640):
    return torch.rand(3, h, w, generator=torch.Generator().manual_seed(0))


def a_clip(seconds=3.0):
    return 0.1 * torch.randn(int(16000 * seconds), generator=torch.Generator().manual_seed(1))


def media_runs(prompt, output: str) -> list[tuple[int, int, tuple[int, ...]]]:
    """The prompt's runs of encoder ``output``: (offset, tokens, grid) each."""
    layout = prompt.metadata[MEDIA_LAYOUT]
    return [
        (offset, tokens, tuple(grid))
        for offset, tokens, out, grid in zip(
            layout["offset"], layout["tokens"], layout["output"], layout["grid"], strict=True,
        )
        if out == output
    ]


@pytest.mark.parametrize("max_slice_nums", [1, None])
def test_image_placeholders_line_up_with_slices(chat_model, max_slice_nums):
    """Each slice is a run of the resampler's queries of ``<unk>`` whose grid is
    its (h, w) in patches, as many as the request's ``max_slice_nums`` allows,
    and the prompt is upstream's, its placeholders written out whole."""
    from mstar.model.minicpm_o import processor
    from mstar.model.multimodal import PromptPart

    model = chat_model
    prompt = model.process_prompt(
        None, ["image", "text"], ["text"],
        tensors={"image_inputs": [an_image()]},
        prompt_parts=[PromptPart("image"), PromptPart("text", "Describe it.")],
        max_slice_nums=max_slice_nums,
    )
    out = prompt.new_input_tensors
    runs = media_runs(prompt, IMAGE_ENCODER.output)
    queries = model.config.resampler.num_queries
    grid = processor.slice_grid((640, 448), max_slice_nums or processor.MAX_SLICE_NUMS)
    assert [tokens for _, tokens, _ in runs] == [queries] * processor.image_runs(grid)
    assert out["pixel_values"][0].shape[0] == sum(h * w for _, _, (h, w) in runs)
    if isinstance(model.tokenizer, _ChatTokenizer):
        return  # upstream's prompt needs the checkpoint's chat template
    upstream = processor.image_placeholder(0, grid).replace("<unk>", "<unk>" * queries)
    text = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{upstream}\nDescribe it."}], tokenize=False, add_generation_prompt=True,
        use_tts_template=False, enable_thinking=False,
    )
    assert out["text_inputs"][0].tolist() == model.tokenizer.encode(text)


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
    [(_, tokens, lens)] = media_runs(prompt, AUDIO_ENCODER.output)
    assert tokens == sum(model.config.audio.pooled_tokens(n) for n in lens)
    assert out["audio_features"][0].shape[-1] == sum(lens)
    # audio anywhere turns upstream's speech template on: the prompt ends on <|tts_bos|>
    assert out["text_inputs"][0][-1].item() == model.tokenizer.convert_tokens_to_ids("<|tts_bos|>")


@needs_ckpt
@pytest.mark.parametrize("seconds", [(30.05,), (60.01,), (0.06,), (1.05, 1.05), (0.06, 2.0)])
def test_audio_placeholders_line_up_for_any_clips(model, seconds):
    """Pieces that pool to nothing (a sliver past a 30 s cut) are left out, a
    clip under ~0.1 s has no run (upstream's empty placeholder), and each clip
    is its own run, as its placeholder is sized."""
    from mstar.model.minicpm_o import processor
    from mstar.model.multimodal import PromptPart

    clips = [a_clip(s) for s in seconds]
    prompt = model.process_prompt(
        None, ["text"] + ["audio"] * len(clips), ["text"],
        tensors={"audio_inputs": clips},
        prompt_parts=[PromptPart("text", "Transcribe.")] + [PromptPart("audio")] * len(clips),
    )
    out = prompt.new_input_tensors
    audio = model.config.audio
    runs = media_runs(prompt, AUDIO_ENCODER.output)
    assert [tokens for _, tokens, _ in runs] == [
        processor.audio_tokens(clip.numel(), audio) for clip in clips if processor.audio_tokens(clip.numel(), audio)
    ]
    assert all(tokens == sum(audio.pooled_tokens(n) for n in lens) for _, tokens, lens in runs)
    assert min((audio.pooled_tokens(n) for _, _, lens in runs for n in lens), default=1) > 0
    # one features tensor a clip with a run, which the encoder joins
    assert [t.shape[-1] for t in out.get("audio_features", [])] == [sum(lens) for _, _, lens in runs]


@needs_ckpt
def test_spoken_reply_system_prompt_with_and_without_the_voice_clip(model):
    """By default a spoken reply's system message carries the voice clip (upstream's
    audio_assistant prompt); with ``voice_prompt=False`` it is the request's text system
    prompt. Both use the speech template, which ends the prompt on <|tts_bos|>."""
    from mstar.model.multimodal import PromptPart

    tts_bos = model.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
    parts = [PromptPart("text", "Say hello.")]
    voiced = model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=parts)
    assert len(media_runs(voiced, AUDIO_ENCODER.output)) == 1
    assert voiced.new_input_tensors["text_inputs"][0][-1].item() == tts_bos
    plain = model.process_prompt(
        None, ["text"], ["text", "audio"], prompt_parts=parts,
        voice_prompt=False, system_prompt="You are a helpful assistant.",
    )
    assert not plain.metadata[MEDIA_LAYOUT]["output"]
    ids = plain.new_input_tensors["text_inputs"][0]
    assert ids[-1].item() == tts_bos
    assert "You are a helpful assistant." in model.tokenizer.decode(ids)
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
    assert model.get_loop_caps(**out.metadata)[DECODE_LOOP] == min(2048, limit) - 1
    with pytest.raises(ValueError, match="spoken reply"):
        model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=parts, max_output_tokens=limit + 1)
    # text replies are not capped
    text = model.process_prompt(None, ["text"], ["text"], prompt_parts=parts).metadata
    assert model.get_max_output_tokens(**text, max_output_tokens=8192) == 8192


def test_an_image_and_a_clip_sharing_unk_take_their_own_encoders_rows(chat_model):
    """Images and audio share ``<unk>``: every one of the prompt's ``<unk>`` is
    a run's, each run names its own encoder, each encoder lays its segments out
    by its runs' grids, and the LLM splices each encoder's rows at its own runs."""
    from mstar.model.keys import ROPE
    from mstar.model.minicpm_o.submodules import AudioEncoderSubmodule, VisionEncoderSubmodule
    from mstar.model.multimodal import PromptPart, splice_media

    model = chat_model
    prompt = model.process_prompt(
        None, ["image", "audio", "text"], ["text"],
        tensors={"image_inputs": [an_image()], "audio_inputs": [a_clip()]},
        prompt_parts=[PromptPart("image"), PromptPart("audio"), PromptPart("text", "What is this?")],
    )
    out = prompt.new_input_tensors
    ids = out["text_inputs"][0]
    layout = prompt.metadata[MEDIA_LAYOUT]
    images, clips = media_runs(prompt, IMAGE_ENCODER.output), media_runs(prompt, AUDIO_ENCODER.output)
    # the image's slices, then the clip, as the prompt holds them
    assert layout["output"] == [IMAGE_ENCODER.output] * (len(layout["output"]) - 1) + [AUDIO_ENCODER.output]
    fill = torch.zeros(len(ids))
    for runs, value in ((images, 1.0), (clips, 2.0)):
        for offset, tokens, _ in runs:
            fill[offset:offset + tokens] = value
    assert torch.equal(fill > 0, ids == model.tokenizer.convert_tokens_to_ids("<unk>"))

    signals = {name: [None] for name, tensors in out.items() if tensors}
    prefill, _ = model.walk_plan("main", ["text"], ["text"], signals, prompt.metadata)
    info = SimpleNamespace(request_id="r", rid_handle=0, step_metadata=prefill.step_metadata)
    vision = VisionEncoderSubmodule(torch.nn.Identity(), model.config).prepare_inputs(
        prefill.walk, info, {"pixel_values": out["pixel_values"]},
    )
    assert sum(vision.kwargs["seq_lengths"]) == vision.input_seq_len == out["pixel_values"][0].shape[0]
    audio = AudioEncoderSubmodule(torch.nn.Identity(), model.config).prepare_inputs(
        prefill.walk, info, {"audio_features": out["audio_features"]},
    )
    assert sum(audio.kwargs["mel_lens"]) == out["audio_features"][0].shape[-1]

    llm = an_llm()
    llm.node_resources = {ROPE: SimpleNamespace(position=lambda rid, label: 0)}
    row = llm.prepare_inputs(prefill.walk, info, {
        "text_inputs": [ids],
        IMAGE_ENCODER.output: [torch.full((sum(t for _, t, _ in images), 1), 1.0)],
        AUDIO_ENCODER.output: [torch.full((sum(t for _, t, _ in clips), 1), 2.0)],
    })
    spliced = splice_media(torch.zeros(len(ids), 1), row.tensor_inputs["media_mask"], row.tensor_inputs["media_embeds"])
    assert torch.equal(spliced[:, 0], fill)


@pytest.mark.parametrize("media", [(), ("image", "audio")])
def test_an_unk_the_user_types_stays_a_token(chat_model, media):
    """``<unk>`` places the media, and a user may also type it: one that follows
    none of the media's sentinels is the prompt's own token, as upstream embeds
    it, and every run stays at its attachment."""
    from mstar.model.multimodal import PromptPart

    attachments = {"image": an_image, "audio": a_clip}
    prompt = chat_model.process_prompt(
        None, [*media, "text"], ["text"],
        tensors={f"{m}_inputs": [attachments[m]()] for m in media},
        prompt_parts=[*(PromptPart(m) for m in media), PromptPart("text", "Is <unk> a word? <unk>")],
    )
    ids = prompt.new_input_tensors["text_inputs"][0]
    layout = prompt.metadata[MEDIA_LAYOUT]
    typed = torch.ones(len(ids), dtype=torch.bool)
    for offset, tokens in zip(layout["offset"], layout["tokens"], strict=True):
        typed[offset:offset + tokens] = False
    unk = chat_model.tokenizer.convert_tokens_to_ids("<unk>")
    assert int(((ids == unk) & typed).sum()) == 2
    assert bool((ids[~typed] == unk).all()) and len(set(layout["output"])) == len(media)


@needs_ckpt
def test_kv_layout_covers_the_prompt_and_each_decode_step(model):
    """The LLM's layout: every prompt row (placeholders included), written by
    the prefill walk the request takes, the text as ``ids`` spans and each
    run of pads as a ``rows`` span, then a row per decode step."""
    from mstar.model.base import LabelLayout
    from mstar.model.minicpm_o.config import TTS_KV
    from mstar.model.multimodal import PromptPart

    out = model.process_prompt(
        None, ["image", "text"], ["text"],
        tensors={"image_inputs": [an_image()]},
        prompt_parts=[PromptPart("image"), PromptPart("text", "Describe it.")],
    )
    spans = out.metadata["kv_layout"][KV_CACHE]["main"].spans
    rows = out.new_input_tensors["text_inputs"][0].numel()
    assert {s.walk for s in spans} == {"prefill_image"}
    assert sum(s.length for s in spans) == sum(s.advance for s in spans) == rows
    layout = out.metadata["media_layout"]
    assert [s.length for s in spans if s.kind == "rows"] == layout["tokens"]
    assert out.metadata["kv_layout"][KV_CACHE]["main"].growth == ((DECODE_LOOP, 1),)
    assert set(model.kv_labels()) == {KV_CACHE, TTS_KV}
    # a text reply opens the TTS's pool with nothing in it; a spoken one prefills its
    # condition (two rows and a row per LLM step) and then writes a row per code
    assert out.metadata["kv_layout"][TTS_KV]["main"] == LabelLayout()
    spoken = model.process_prompt(None, ["text"], ["text", "audio"], prompt_parts=[PromptPart("text", "Hi.")])
    tts = spoken.metadata["kv_layout"][TTS_KV]["main"]
    assert [(s.walk, s.length) for s in tts.spans] == [("tts_prefill", 2)]
    assert tts.growth == ((DECODE_LOOP, 1), ("tts_decode_loop", 1))


@needs_ckpt
@pytest.mark.parametrize("media, walk", [
    ((), "prefill_text"),
    (("image",), "prefill_image"),
    (("audio",), "prefill_audio"),
    (("image", "audio"), "prefill_image_audio"),
])
def test_the_plan_prefills_by_the_media_then_decodes_its_token(model, media, walk):
    """The prefill walk the prompt's media pick, whose inputs the request's signals
    all name (the conductor sends each input the signal of its name), then decode
    fed the prefill's token."""
    from mstar.conductor.request_info import FINAL_PREFILL
    from mstar.model.multimodal import PromptPart

    attachments = {"image": an_image, "audio": a_clip}
    prompt = model.process_prompt(
        None, [*media, "text"], ["text"],
        tensors={f"{m}_inputs": [attachments[m]()] for m in media},
        prompt_parts=[*(PromptPart(m) for m in media), PromptPart("text", "Hi.")],
    )
    signals = {name: [None] for name, tensors in prompt.new_input_tensors.items() if tensors}
    prefill, decode = model.walk_plan("main", ["text"], ["text"], signals, prompt.metadata)
    assert (prefill.walk, prefill.step_metadata[FINAL_PREFILL]) == (walk, True)
    assert (decode.walk, decode.bind) == ("decode", {"text_inputs": "new_token"})
    produced = {IMAGE_ENCODER.output, AUDIO_ENCODER.output}
    for node in model.get_graph_walk_graphs()[walk].get_nodes().values():
        assert set(node.input_names) - produced <= set(signals)


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
        "LLM": {"kv_cache", "attn", "rope", "sampler"},
        "vision_encoder": {"vision_attn", "resampler_attn"},
        "audio_encoder": {"audio_attn"},
        "TTS": {"tts_kv", "tts_attn", "tts_pos", "tts_sampler"},
        "Token2Wav": {"t2w_state", "t2w_enc1_kv", "t2w_enc2_kv", "t2w_dit_kv"},
    }


def test_audio_block_loop_leases_within_the_resources_segment_limit():
    """The captured block loop's plan holds the attention resource's
    ``max_segments_per_request`` pieces, as configured (a YAML override
    included): a request with more gets no lease and runs eagerly."""
    from mstar.model.minicpm_o.config import AUDIO_ATTN
    from mstar.model.minicpm_o.submodules import AUDIO_BLOCK_LOOP, AudioEncoderSubmodule
    from mstar.model.submodule_base import NodeInputs

    audio = AudioEncoderSubmodule(torch.nn.Identity(), a_config())
    audio.bind_node_resources({AUDIO_ATTN: SimpleNamespace(max_segments_per_request=2)})
    region = audio.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)[AUDIO_BLOCK_LOOP]

    def request(pieces):
        return [NodeInputs(input_seq_len=100 * pieces, kwargs={"frames": [100] * pieces})]

    assert region.can_lease(request(2))
    assert not region.can_lease(request(3))


def a_deployment(cap: int | None = 32, tts_pages: int = 32 * 32 + 1, **resources) -> dict:
    """A YAML whose tts_kv holds ``cap`` full spoken replies (32 pages each) and its sink page."""
    from mstar.model.minicpm_o.config import TTS_KV

    config = {"resources": {TTS_KV: {"max_num_pages": tts_pages}, **resources}}
    return config if cap is None else {"max_concurrent_requests": cap, **config}


def test_vocoder_slot_counts_must_agree():
    """A spoken reply holds a slot of each t2w resource, so one count sizes all four:
    t2w_state has one more (its sink). A YAML that sets them apart is refused."""
    from mstar.model.minicpm_o.components.tts import TTSConfig
    from mstar.model.minicpm_o.config import T2W_KV, T2W_STATE

    model = a_bare_model(tts_config=TTSConfig())

    def yaml(state, kv):
        return a_deployment(**{T2W_STATE: {"max_slots": state}, **{key: {"max_slots": kv} for key in T2W_KV.values()}})

    model.validate_config_yaml(yaml(33, 32), "minicpm_o.yaml")
    model.validate_config_yaml(a_deployment(), "minicpm_o.yaml")
    with pytest.raises(ValueError, match="t2w_state"):
        model.validate_config_yaml(yaml(33, 33), "minicpm_o.yaml")
    with pytest.raises(ValueError, match="t2w_state"):
        model.validate_config_yaml(a_deployment(**{T2W_STATE: {"max_slots": 33}}), "minicpm_o.yaml")


def test_tts_kv_must_hold_every_admitted_spoken_reply():
    """Nothing gates the TTS on the vocoder's slots: it holds every spoken reply the
    LLM has finished, each up to its 4096 positions, until its last code. A tts_kv
    that can't hold `max_concurrent_requests` full replies, or no cap, is refused:
    a burst of long replies would fill it and stall (G2.1)."""
    from mstar.model.minicpm_o.components.tts import TTSConfig

    model = a_bare_model(tts_config=TTSConfig())

    model.validate_config_yaml(a_deployment(), "minicpm_o.yaml")
    model.validate_config_yaml(a_deployment(cap=8, tts_pages=8 * 32 + 1), "minicpm_o.yaml")
    for config in (a_deployment(tts_pages=32 * 32), a_deployment(cap=None)):
        with pytest.raises(ValueError, match="tts_kv"):
            model.validate_config_yaml(config, "minicpm_o.yaml")


def test_the_model_pickles_for_its_workers(chat_model):
    """The server pickles the Model into every worker process it spawns; a
    mappingproxy on the instance breaks that."""
    import pickle

    from mstar.model.minicpm_o.minicpm_o_model import SAMPLING_DEFAULTS

    loaded = pickle.loads(pickle.dumps(chat_model))
    assert [m.modality for m in loaded.media] == [m.modality for m in chat_model.media]
    assert loaded.sampling_defaults == dict(SAMPLING_DEFAULTS)
