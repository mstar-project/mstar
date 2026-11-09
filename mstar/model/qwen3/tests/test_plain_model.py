"""Plain Qwen3 with no engine, on the CPU: what the model refuses, what reaches its
chat template and which sampling knobs it serves. `test_qwen3_real_weights.py`
checks its tokens against transformers on a GPU."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

# the model's resources import Triton kernels (a torch dependency on Linux)
pytest.importorskip("triton")

STOP = 96
CONFIG = dict(
    architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=32, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48, vocab_size=97,
    rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=512, tie_word_embeddings=False,
    eos_token_id=STOP, torch_dtype="float32",
)


def _model(tmp_path, config: dict | None = None, **kwargs):
    from mstar.model.qwen3.qwen3_model import Qwen3Model

    (tmp_path / "config.json").write_text(json.dumps({**CONFIG, **(config or {})}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": False}))
    return Qwen3Model(model_path_hf=str(tmp_path), max_seq_len=256, accelerator_graph=False, **kwargs)


def _sampler_spec(model):
    from mstar.engine.resources.sampler.config import SamplerSpec

    (spec,) = [s for s in model.get_node_resources() if isinstance(s, SamplerSpec)]
    return spec


def test_a_node_declares_kv_attention_positions_and_a_sampler(tmp_path):
    keys = {spec.resource_key for spec in _model(tmp_path).get_node_resources()}
    assert keys == {"kv_cache", "attn", "rope", "sampler"}


def test_the_download_the_tokenizer_loads_fetches_a_template_kept_in_its_own_file(tmp_path, monkeypatch):
    """On an empty cache, with the files filtered as the hub filters them."""
    import shutil

    from huggingface_hub.utils import filter_repo_objects

    from mstar.model.qwen3.qwen3_model import Qwen3Model

    repo, snapshot = tmp_path / "repo", tmp_path / "snapshot"
    repo.mkdir()
    snapshot.mkdir()
    (repo / "config.json").write_text(json.dumps(CONFIG))
    (repo / "chat_template.jinja").write_text("{{ messages }}")
    (repo / "model.safetensors").write_text("")

    def download(repo_id, cache_dir=None, allow_patterns=None):
        names = [p.name for p in repo.iterdir()]
        for name in filter_repo_objects(names, allow_patterns=allow_patterns):
            shutil.copy(repo / name, snapshot / name)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    model = Qwen3Model(model_path_hf="org/fine-tune", max_seq_len=256, accelerator_graph=False)
    assert (model.local_dir / "chat_template.jinja").is_file()
    assert not (model.local_dir / "model.safetensors").exists()


def test_a_scaled_rope_in_rope_parameters_is_refused(tmp_path):
    with pytest.raises(ValueError, match="scaled RoPE"):
        _model(tmp_path, {"rope_parameters": {"rope_type": "yarn", "factor": 4.0}})
    assert _model(tmp_path, {"rope_parameters": {"rope_theta": 5e5}}).config.rope_theta == 5e5


@pytest.mark.parametrize("change, message", [
    ({"model_type": "llama"}, "not a dense Qwen3 checkpoint"),
    ({"rope_scaling": {"rope_type": "yarn"}}, "rope_scaling"),
    ({"use_sliding_window": True}, "sliding-window"),
    ({"attention_bias": True}, "attention_bias"),
    ({"hidden_act": "gelu"}, "hidden_act"),
])
def test_what_the_shared_decoder_doesnt_implement_is_refused_at_startup(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        _model(tmp_path, change)


def _prompt(model, ids: list[int]):
    tensors = {"text_inputs": [torch.tensor(ids)]}
    return model.process_prompt(None, ["text"], ["text"], tensors=tensors, max_output_tokens=4)


def test_a_prompt_past_the_limit_or_the_vocabulary_is_refused(tmp_path):
    model = _model(tmp_path)
    assert _prompt(model, [1] * 252).new_input_tensors["text_inputs"][0].numel() == 252
    with pytest.raises(ValueError, match="must fit max_seq_len"):
        _prompt(model, [1] * 253)
    with pytest.raises(ValueError, match="within the vocabulary"):
        _prompt(model, [1, CONFIG["vocab_size"]])


def test_every_output_of_a_decode_step_stays_on_the_device(tmp_path):
    """The worker routes a decode step without a tensor only if every signal it outputs is the
    one the node keeps on the device, the client's copy included."""
    from mstar.model.qwen3.config import DECODE_WALK, LLM_NODE, SAMPLER
    from mstar.model.qwen3.submodules import Qwen3LMSubmodule
    from mstar.worker.step_executor import GpuStepBody

    sub = Qwen3LMSubmodule(None, [STOP])
    sub.node_resources = {SAMPLER: SimpleNamespace(has_slot_masters=True)}
    body = GpuStepBody.__new__(GpuStepBody)
    body._graph_runtime = SimpleNamespace(supports_inline_emit=True)
    body._device_loopback = {}
    engine = SimpleNamespace(device_loopback_signals=lambda node, walk: sub.device_loopback_signals(walk))
    signals = [edge.name for edge in _model(tmp_path).get_graph_walk_graphs()[DECODE_WALK].section.outputs]
    assert body.all_on_device(engine, LLM_NODE, DECODE_WALK, signals)


def test_enable_thinking_reaches_the_template_flat_or_as_chat_template_kwargs(tmp_path):
    model = _model(tmp_path)
    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.append(kwargs.get("enable_thinking"))
            return [1, 2, 3]

    model._tokenizer = Tokenizer()
    cases = (
        ({"enable_thinking": False}, False), ({"chat_template_kwargs": {"enable_thinking": True}}, True), ({}, None),
    )
    for kwargs, want in cases:
        model.process_prompt("hi", ["text"], ["text"], max_output_tokens=4, **kwargs)
        assert seen[-1] is want


@pytest.mark.parametrize("knob, served", [
    ({"min_tokens": 4}, False), ({"repetition_penalty": 1.1}, True), ({"penalize_prompt": False}, True),
    ({"repetition_window": 8}, False), ({"top_p_first": True}, False), ({"top_p_min_keep": 3}, False),
])
def test_it_serves_its_knobs_and_refuses_the_rest_at_ingest(tmp_path, knob, served):
    """A knob the node can't honour is a 400 at ingest, never later at the worker."""
    model = _model(tmp_path)
    config = model.get_request_resource_configs({}, knob)["sampler"]
    if served:
        config.validate(_sampler_spec(model))
    else:
        with pytest.raises(ValueError):
            config.validate(_sampler_spec(model))


@pytest.mark.parametrize("opt_in", [False, True])
def test_min_tokens_is_an_opt_in_that_keeps_the_default_sampler_fast(tmp_path, opt_in):
    """Without ``serve_min_tokens`` the sampler declares no stop ids (the fused
    FlashInfer path) and min_tokens is a 400."""
    model = _model(tmp_path, serve_min_tokens=opt_in)
    spec = _sampler_spec(model)
    assert bool(spec.min_tokens_stop_ids) is opt_in
    config = model.get_request_resource_configs({}, {"min_tokens": 4})["sampler"]
    if opt_in:
        config.validate(spec)
    else:
        with pytest.raises(ValueError, match="min_tokens"):
            config.validate(spec)


def test_an_eager_decode_step_reads_its_routed_tokens():
    """``accelerator_graph: false`` builds no slot masters, so an uncaptured step whose rows
    all brought their token must not read them."""
    from mstar.model.qwen3.config import DECODE_WALK
    from mstar.model.qwen3.submodules import Qwen3LMSubmodule
    from mstar.model.submodule_base import ARNodeInputs

    sub = Qwen3LMSubmodule(None, [STOP], accelerator_graph=False)
    inputs = [ARNodeInputs(input_ids=torch.tensor([t]), input_seq_len=1) for t in (3, 5)]
    out = sub.preprocess(DECODE_WALK, SimpleNamespace(captured=False), inputs)
    assert out["input_ids"].tolist() == [3, 5]
