"""Cosmos3 says how much each cache label holds, because its prompt does not.

The pool admits a request only when what it may take fits, and a Cosmos3
prompt is a fraction of that: every denoise step allocates the whole
generation span (video frames, a sound band, an action chunk) on each guidance
branch without committing it, and a kv-mode window keeps the committed context
of the windows before it. Counting the prompt alone admits a request whose
first denoise step finds no page, which is the hold admission exists to end.

The counts below come from the DiT's own prefill, run on CPU over a request
the data worker built from uploaded files, as the API sends it: the statics
its packing builds, the spans its window plans commit and the pool's own
page-floored release.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from api_uploads import mp4, png, preprocess
from huggingface_hub import try_to_load_from_cache

from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import (
    COND_LABEL,
    KV_CACHE,
    UNCOND_LABEL,
    Cosmos3DiTSubmodule,
)

PAGE_SIZE = PagedKVConfig.page_size
TEXT = "a robot arm stacks red cubes"


def _deployment(name: str) -> dict:
    with open(f"configs/{name}.yaml") as f:
        return yaml.safe_load(f)


class _StubPool:
    """Releases a stream's oldest pages the way the pool does, into nowhere."""

    _apply_retention = KVManager._apply_retention
    _release_oldest_locked = KVManager._release_oldest_locked

    def __init__(self):
        self.config = SimpleNamespace(page_size=PAGE_SIZE)
        self._arena = SimpleNamespace(release=lambda pages: None)


def _stream(stored_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        stored_len=stored_len, protected_prefix=stored_len, released=0, generation=0,
        page_indices=list(range(-(-stored_len // PAGE_SIZE))),
    )


def _cached(repo: str) -> str:
    if not isinstance(try_to_load_from_cache(repo, "config.json"), str):
        pytest.skip(f"{repo} is not in the Hugging Face cache")
    return repo


@pytest.fixture(scope="module")
def nano():
    return Cosmos3Model(_cached("nvidia/Cosmos3-Nano"), enable_windowed_video=True)


def _dit(model) -> Cosmos3DiTSubmodule:
    dit = Cosmos3DiTSubmodule(transformer=None, config=model.config)
    dit.transformer = SimpleNamespace(
        proj_in=SimpleNamespace(weight=torch.zeros(1)), action_dim=model.config.max_action_dim,
    )
    dit._new_scheduler = lambda *args, **kwargs: None
    return dit


def _walked(model, files, output_modalities, kwargs, tmp_path) -> tuple[dict, dict]:
    """(what the worker declares for an upload, the peak each label reaches as the DiT walks it)."""
    tensors, declared = preprocess(
        model, _deployment("cosmos3_nano_ar"), TEXT, files, output_modalities, kwargs, tmp_path,
    )
    input_modalities = [name for name in ("image", "video") if f"{name}_inputs" in tensors] + ["text"]
    params = model._resolve_gen_params(kwargs, input_modalities, output_modalities)
    dit = _dit(model)
    fwd = SimpleNamespace(request_id="r", rid_handle="r", step_metadata=params, random_seed=0)
    prefill = dit._prepare_prefill(fwd, tensors, "cpu")
    st = dit.request_states["r"]
    peaks = {}
    for label, prompt in prefill.kwargs["seq_lens"].items():
        static = st[{COND_LABEL: "cond", UNCOND_LABEL: "uncond"}[label]]
        if "ar_schedule" not in st:
            # every denoise step allocates this span and commits none of it
            span = static["num_vision_tokens"] + (st.get("num_sound") or 0) + static.get("num_action_tokens", 0)
            peaks[label] = prompt + span
            continue
        pool, stream, peak = _StubPool(), _stream(prompt), 0
        for plan in st["ar_schedule"].windows():
            cond, uncond = dit._window_statics_for(st, plan, "cpu")
            span = (cond if label == COND_LABEL else uncond)["num_vision_tokens"]
            peak = max(peak, stream.stored_len + span)
            if not st["ar_kv_mode"]:
                continue
            commit = dit._prepare_commit(st, plan.index, None, None)
            stream.stored_len += commit.input_seq_len
            stream.page_indices = list(range(-(-stream.stored_len // PAGE_SIZE)))
            policy = (commit.resource_step_info.retention or {}).get(label)
            if policy is not None:
                pool._apply_retention(stream, policy)
        peaks[label] = peak
    return declared, peaks


@pytest.mark.parametrize(
    "files, output_modalities, kwargs",
    [
        (lambda tmp: [], ["image"], {"guidance_scale": 1.0, "size": "832x480"}),
        (lambda tmp: [mp4(17, 480, 832, tmp)], ["video"], {"num_frames": 61}),
        (lambda tmp: [], ["video"], {"num_frames": 61, "generate_sound": True}),
        (
            lambda tmp: [png(480, 832)], ["action"],
            {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10},
        ),
        (lambda tmp: [png(480, 832)], ["video"], {"num_frames": 57, "window_mode": "chained"}),
        # a short context the pool trims past in whole pages, and an unbounded one it never trims
        (lambda tmp: [], ["video"], {"num_frames": 189, "size": "832x480", "window_mode": "kv", "context_frames": 13}),
        (
            lambda tmp: [png(480, 832)], ["video"],
            {"num_frames": 189, "size": "832x480", "window_mode": "kv", "context_frames": 0},
        ),
    ],
    ids=["t2i_no_guidance", "v2v", "video_sound", "action", "i2v_chained", "kv_short_context", "kv_unbounded_context"],
)
def test_each_branch_reserves_exactly_the_peak_its_denoise_steps_reach(
    nano, files, output_modalities, kwargs, tmp_path,
):
    declared, peaks = _walked(nano, files(tmp_path), output_modalities, kwargs, tmp_path)

    assert (declared["prompt_slots"], declared["decode_labels"]) == (
        {KV_CACHE: {COND_LABEL: peaks[COND_LABEL], UNCOND_LABEL: peaks.get(UNCOND_LABEL, 0)}}, {KV_CACHE: []},
    ), "a branch was counted other than the most its prompt, kept context and denoise span hold at once"
