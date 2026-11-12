"""Plain Qwen3 with no engine, on the CPU: what the model refuses, what reaches its
chat template, its stop ids and its sampling defaults. `test_qwen3_real_weights.py`
checks its tokens against transformers on a GPU, and `test/modular/test_qwen3_intake.py`
what the conductor makes of a prompt's length."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# the model's resources import Triton kernels (a torch dependency on Linux)
pytest.importorskip("triton")

STOP = 96
CONFIG = dict(
    architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=32, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48, vocab_size=97,
    rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=512, tie_word_embeddings=False,
    eos_token_id=STOP, torch_dtype="float32",
)


def _model(tmp_path, config: dict | None = None, generation: dict | None = None):
    from mstar.model.qwen3.qwen3_model import Qwen3Model

    (tmp_path / "config.json").write_text(json.dumps({**CONFIG, **(config or {})}))
    (tmp_path / "generation_config.json").write_text(json.dumps(generation or {"do_sample": False}))
    return Qwen3Model(model_path_hf=str(tmp_path))


def _sampler_spec(model):
    from mstar.engine.resources.sampler.config import SamplerSpec

    (spec,) = [s for s in model.get_node_resources() if isinstance(s, SamplerSpec)]
    return spec


def test_a_node_declares_kv_attention_positions_and_a_sampler(tmp_path):
    keys = {spec.resource_key for spec in _model(tmp_path).get_node_resources()}
    assert keys == {"kv_cache", "attn", "rope", "sampler"}


def test_the_kv_pool_holds_the_checkpoints_context(tmp_path):
    from mstar.engine.resources.kv.config import KVSpec

    (kv,) = [s for s in _model(tmp_path).get_node_resources() if isinstance(s, KVSpec)]
    assert kv.config.max_seq_len == CONFIG["max_position_embeddings"]


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
    model = Qwen3Model(model_path_hf="org/fine-tune")
    assert (Path(model.local_dir) / "chat_template.jinja").is_file()
    assert not (Path(model.local_dir) / "model.safetensors").exists()


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


def test_the_stop_ids_are_every_eos_the_configs_name(tmp_path):
    model = _model(tmp_path, {"eos_token_id": [STOP, 95]}, {"eos_token_id": 94, "do_sample": False})
    assert model.config.stop_token_ids == {STOP, 95, 94}
    assert model.text_output().stop_token_ids == {STOP, 95, 94}


def test_every_output_of_a_decode_step_stays_on_the_device(tmp_path):
    """The worker routes a decode step without a tensor only if every signal it outputs is the
    one the node keeps on the device, the client's copy included."""
    from mstar.model.qwen3.config import LLM_NODE
    from mstar.model.qwen3.submodules import Qwen3LMSubmodule
    from mstar.testing_utils.worker import make_worker

    model = _model(tmp_path)
    sub = Qwen3LMSubmodule(None, model.config)
    sub.node_resources = {sub.SAMPLER: SimpleNamespace(has_slot_masters=True)}
    # the worker's own step body, on a runtime that sends the client's copy inline
    body = make_worker(graph_runtime=SimpleNamespace(supports_inline_emit=True)).executor._body
    engine = SimpleNamespace(device_loopback_signals=lambda node, walk: sub.device_loopback_signals(walk))
    walk = sub.DECODE_WALK
    signals = [edge.name for edge in model.get_graph_walk_graphs()[walk].section.outputs]
    assert body.all_on_device(engine, LLM_NODE, walk, signals)


def test_decode_reads_its_rows_off_the_resident_tables(tmp_path):
    from mstar.model.qwen3.submodules import Qwen3LMSubmodule

    sub = Qwen3LMSubmodule(None, _model(tmp_path).config)
    assert sub.resident_walks() == {sub.DECODE_WALK}


class _Tokenizer:
    """Renders to text, then encodes it, as the model asks (no Rust backend, so
    `encode_ids` calls it)."""

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get("tokenize") is False
        self.seen.append((messages, kwargs.get("enable_thinking")))
        return "rendered"

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=[1, 2, 3])


def test_enable_thinking_reaches_the_template_flat_or_as_chat_template_kwargs(tmp_path):
    model = _model(tmp_path)
    model.tokenizer = tokenizer = _Tokenizer()
    cases = (
        ({"enable_thinking": False}, False), ({"chat_template_kwargs": {"enable_thinking": True}}, True), ({}, None),
    )
    for kwargs, want in cases:
        model.process_prompt("hi", ["text"], ["text"], **kwargs)
        assert tokenizer.seen[-1][1] is want


@pytest.mark.parametrize("kwargs", [{"enable_thinking": "no"}, {"chat_template_kwargs": {"enable_thinking": 0}}])
def test_an_enable_thinking_that_isnt_a_boolean_is_a_400(tmp_path, kwargs):
    model = _model(tmp_path)
    model.tokenizer = _Tokenizer()
    with pytest.raises(ValueError, match="enable_thinking must be a boolean"):
        model.process_prompt("hi", ["text"], ["text"], **kwargs)


@pytest.mark.parametrize("template_kwargs", ["enable_thinking", ["enable_thinking"], 1])
def test_chat_template_kwargs_that_isnt_an_object_is_a_400(tmp_path, template_kwargs):
    """A ValueError, which the data worker answers with a 400, not an AttributeError (a 500)."""
    model = _model(tmp_path)
    model.tokenizer = _Tokenizer()
    with pytest.raises(ValueError, match="chat_template_kwargs must be an object"):
        model.process_prompt("hi", ["text"], ["text"], chat_template_kwargs=template_kwargs)


def test_a_messages_text_parts_are_joined_by_newlines(tmp_path):
    from mstar.model.multimodal import TEXT, PromptPart

    model = _model(tmp_path)
    model.tokenizer = tokenizer = _Tokenizer()
    parts = [PromptPart(modality=TEXT, text="one", role="user"), PromptPart(modality=TEXT, text="two", role="user")]
    # the joined text as the prompt, as TextChatAdapter sends it
    model.process_prompt("one\ntwo", ["text", "text"], ["text"], prompt_parts=parts)
    assert tokenizer.seen[-1][0] == [{"role": "user", "content": "one\ntwo"}]


def test_sampling_defaults_come_from_generation_config_and_min_p_turns_its_filter_on(tmp_path):
    generation = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.05}
    model = _model(tmp_path, generation=generation)
    config = model.get_request_resource_configs(["text"], ["text"], {"top_k": 5})["sampler"]
    assert (config.temperature, config.top_p, config.top_k, config.min_p) == (0.6, 0.95, 5, 0.05)
    assert _sampler_spec(model).enable_min_p
    greedy = _model(tmp_path)
    assert greedy.get_request_resource_configs(["text"], ["text"], {})["sampler"].temperature == 0
    assert not _sampler_spec(greedy).enable_min_p
