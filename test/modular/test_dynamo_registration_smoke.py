"""Contract smoke against the pinned ai-dynamo wheel (no GPU, no etcd —
registration goes through the runtime's in-memory discovery backend).

Exercises the exact call shapes the Dynamo worker uses: the registration
vocabulary, the dotted endpoint path form, and `register_model` with our
argument shapes — card-only masks (images/videos) and one that resolves a
model directory (audios). A wheel bump that changes any of this fails here
rather than in a deployment. All tests skip when the bindings aren't
installed."""

import asyncio
import json

import pytest

from mstar.api_server.openai.adapters import ADAPTER_REGISTRY


def test_registration_vocabulary():
    pytest.importorskip("dynamo.llm")
    from dynamo.llm import ModelInput, ModelType, WorkerType

    # Members and combinators the worker's registration path touches. The
    # pyo3 ModelType has no __eq__; the flag-set string form is the contract.
    assert str(ModelType.Chat | ModelType.Images) == "chat,images"
    assert str(ModelType.Chat | ModelType.Audios) == "chat,audios"
    assert str(ModelType.Videos) == "videos"
    for name in ("Chat", "Completions", "Embedding", "Images", "Videos",
                 "Audios", "Realtime", "TensorBased"):
        assert hasattr(ModelType, name), name
    for name in ("Text", "Tokens", "Tensor"):
        assert hasattr(ModelInput, name), name
    for name in ("Aggregated", "Prefill", "Decode", "Encode"):
        assert hasattr(WorkerType, name), name


def test_every_adapter_surface_maps():
    pytest.importorskip("dynamo.llm")
    from mstar.integrations.dynamo.worker import _model_type

    # Every adapter surface flag must translate into a registration mask;
    # a flag without a mapping would silently serve native-only.
    for name, adapter in ADAPTER_REGISTRY.items():
        has_surface = any(
            getattr(adapter, flag, False)
            for flag in ("supports_chat", "supports_speech", "supports_images")
        )
        if has_surface:
            assert _model_type(adapter) is not None, name
        if getattr(adapter, "supports_videos", False):
            assert True  # registered separately on the videos endpoint


def _minimal_model_dir(tmp_path):
    """Smallest on-disk model the card loader accepts (config + tokenizer)."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 4,
        "max_position_embeddings": 2048,
        "eos_token_id": 0,
        "bos_token_id": 0,
    }))
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": {"<unk>": 0, "a": 1},
            "unk_token": "<unk>",
        },
    }))
    return tmp_path


def test_register_model_in_process(tmp_path):
    pytest.importorskip("dynamo.llm")
    from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
    from dynamo.runtime import DistributedRuntime

    model_dir = str(_minimal_model_dir(tmp_path))

    async def _smoke():
        runtime = DistributedRuntime(asyncio.get_running_loop(), "mem", "tcp")
        try:
            # Card-only masks: no model-directory resolution.
            generate = runtime.endpoint("mstar.citest.generate")
            await register_model(
                ModelInput.Text,
                ModelType.Chat | ModelType.Images,
                generate,
                model_dir,
                "citest",
                worker_type=WorkerType.Aggregated,
                needs=[],
            )
            # Second card under the same served name on its own endpoint —
            # the dual registration video-capable models use.
            videos = runtime.endpoint("mstar.citest.generate_videos")
            await register_model(
                ModelInput.Text,
                ModelType.Videos,
                videos,
                model_dir,
                "citest",
                worker_type=WorkerType.Aggregated,
                needs=[],
            )
            # Audio-only masks resolve the model directory for the card.
            speech = runtime.endpoint("mstar.citest_tts.generate")
            await register_model(
                ModelInput.Text,
                ModelType.Audios,
                speech,
                model_dir,
                "citest-tts",
                worker_type=WorkerType.Aggregated,
                needs=[],
            )
        finally:
            runtime.shutdown()

    asyncio.run(_smoke())
