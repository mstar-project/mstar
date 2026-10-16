"""Qwen3-Omni says how many slots each request takes in both of its caches.

The pool admits a request only when its pages fit, and it can count only
what the request declares. Each attachment's walk writes its encoder's tokens
between two sentinels no text span carries, and the Talker writes a copy of
the Thinker's prefill (less the system turn), an assistant prefix and up to
``talker_max_output_tokens`` codes. Declaring less lets the pool admit a
request whose own prefill or speech then finds no page; declaring more than
the walks write refuses a request that fits.

The Thinker's own `prepare_inputs`, `declare_step` and `postprocess` run for
each scheduled walk, with each encoder's output at the length the encoder
makes. Needs the Qwen/Qwen3-Omni-30B-A3B-Instruct tokenizer and processor
configs in the HF cache, not the weights.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from api_uploads import decode_uploads, mp4, png, preprocess, wav
from huggingface_hub import try_to_load_from_cache
from torch import nn
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)

from mstar.api_server.entrypoint import _detect_modality
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.qwen3_omni.config import TALKER_KV, THINKER_KV, THINKER_POS

REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


class _StubLM(nn.Module):
    """Only the embedding the prefill walks read."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, 1)


class _StubPos:
    """Every request starts at position 0."""

    def position(self, rid, label):
        return 0


@pytest.fixture(scope="module")
def model():
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    for name in (
        "config.json", "vocab.json", "merges.txt", "tokenizer_config.json",
        "preprocessor_config.json", "chat_template.json",
    ):
        path = try_to_load_from_cache(REPO, name)
        if not isinstance(path, str):
            pytest.skip(f"{REPO} {name} is not in the HF cache")
    return Qwen3OmniModel(os.path.dirname(path))


@pytest.fixture(scope="module")
def thinker(model):
    from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

    thinker = ThinkerSubmodule(_StubLM(model.config.thinker_text.vocab_size), model.config)
    thinker.node_resources = {THINKER_POS: _StubPos()}
    return thinker


def _info(walk: str, **step_metadata) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="r", graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=1, step_metadata=step_metadata,
    )


def _thinker_inputs(model, walk: str, entry: dict) -> dict:
    """What the Thinker node receives for one scheduled walk."""
    if walk == "prefill_text":
        return {"text_inputs": [entry["text_inputs"]]}
    if walk == "prefill_audio":
        tokens = int(_get_feat_extract_output_lengths(entry["audio_seqlens"]))
        return {"audio_embeds": [torch.zeros(tokens, 1)]}
    # the vision encoder's merger leaves one token per merge x merge patches
    grid = entry["image_grid_thw"]
    tokens = int(grid.prod()) // model.config.vision.spatial_merge_size ** 2
    inputs = {"vision_embeds": [torch.zeros(tokens, 1)], "image_grid_thw": [grid], "deepstack": []}
    if "video_second_per_grid" in entry:
        inputs["video_second_per_grid"] = [entry["video_second_per_grid"]]
    return inputs


def _written(model, thinker, modalities: list[str], tensors) -> tuple[int, int]:
    """The slots the Thinker's prefill walks declare, and the rows they hand the Talker."""
    written = handed = 0
    for walk, entry in model._build_thinker_prefill_schedule(modalities, tensors):
        inp = thinker.prepare_inputs(walk, _info(walk), _thinker_inputs(model, walk, entry))
        written += sum(s.span for s in thinker.declare_step(walk, ["r"], [inp]).segments)
        outputs = {
            "thinker_states": [torch.zeros(inp.input_seq_len, 2 * model.config.thinker_hidden_size)],
            "thinker_mask": [inp.tensor_inputs["masks_for_talker"]],
        }
        thinker.postprocess("r", _info(walk, audio_output=True), outputs)
        handed += outputs["thinker_states"][0].shape[0]
    return written, handed


def _talker_decode_writes(model, talker_max: int) -> int:
    """The talker_decode steps that write a slot for a request that never says eos.

    `check_stop` ends the loop at ``talker_max``, and `Worker._try_speculate_next`
    has already submitted one more step behind it. The loop's ``max_iters`` ends it
    with no step past it.
    """
    from mstar.model.qwen3_omni.submodules import TalkerSubmodule

    max_iters = model.get_graph_walk_graphs()["talker_decode"].max_iters
    info = _info("talker_decode", talker_max_tokens=talker_max)
    talker = SimpleNamespace(config=model.config)
    steps = 0
    while True:
        steps += 1
        if steps == max_iters:
            return steps
        if TalkerSubmodule.check_stop(talker, "r", info, {"layer0_codes": [torch.tensor(0)]}):
            return steps + 1
        info.dynamic_loop_iter_counts["talker_decode_loop"] = steps


def _talker_written(model, thinker, modalities, tensors, talker_max: int) -> int:
    """The slots the Talker writes: the rows the Thinker hands it, its assistant prefix and its decode."""
    from mstar.model.qwen3_omni.submodules import TalkerSubmodule

    _, handed = _written(model, thinker, modalities, tensors)
    return handed + TalkerSubmodule.TALKER_LAST_PREFILL_TOKENS_PER_REQ + _talker_decode_writes(model, talker_max)


@pytest.mark.parametrize(("prompt", "files", "output_modalities", "kwargs"), [
    ("Describe a cat.", lambda tmp: [], ["text"], {}),
    ("Compare these.", lambda tmp: [png(224, 224), wav(3, 16000), png(480, 640), wav(7.5, 16000)], ["text"], {}),
    ("What happens?", lambda tmp: [mp4(8, 224, 224, tmp)], ["text"], {}),
    # a talker bound under 1 that check_stop ends at its first step, and one past the loop's max_iters
    ("Describe a cat.", lambda tmp: [], ["text", "audio"], {"talker_max_output_tokens": 0}),
    ("Describe this.", lambda tmp: [png(224, 224)], ["text", "audio"], {"talker_max_output_tokens": 5000}),
], ids=["text", "images_and_clips", "video", "speech_bound_under_one", "speech_bound_past_the_loop"])
def test_each_cache_reserves_what_its_walks_write(
    model, thinker, prompt, files, output_modalities, kwargs, monkeypatch, tmp_path,
):
    decode_uploads(monkeypatch)
    # `load_video` imports qwen_omni_utils, whose audio half needs audioread
    # even to read a video; the CPU venv has no audioread
    if "audioread" not in sys.modules:
        monkeypatch.setitem(sys.modules, "audioread", ModuleType("audioread"))
    uploads = files(tmp_path)
    tensors, declared = preprocess(model, None, prompt, uploads, output_modalities, kwargs, tmp_path)

    modalities = [_detect_modality(name) for name, _ in uploads] + ["text"]
    thinker_written, _ = _written(model, thinker, modalities, tensors)
    talker = 0
    if "audio" in output_modalities:
        talker = _talker_written(model, thinker, modalities, tensors, model._get_max_talker_output_tokens(**kwargs))
    assert (declared["prompt_slots"], declared["decode_labels"]) == (
        {THINKER_KV: {"main": thinker_written}, TALKER_KV: {"main": talker}},
        {THINKER_KV: ["main"], TALKER_KV: []},
    ), (
        "a cache was declared at other than its walks write: the Thinker's prompt and attachments, "
        "the Talker's copy of it, its prefix and its own bounded decode"
    )
