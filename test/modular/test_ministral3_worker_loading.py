"""Real local safetensors I/O through both node factories, with tiny random weights.

Only the text factory's GPU eligibility check is bypassed to inspect storage on
CPU. FP8 execution still refuses CPU; no pretrained or GPU parity is claimed.
"""

import json
import shutil

import pytest
import torch
from safetensors.torch import save_file
from test_ministral3_model import checkpoint as _checkpoint
from test_ministral3_vision import bind

import mstar.model.ministral3.ministral3_model as model_module
from mstar.engine.node_dtype import resolve_node_dtype
from mstar.model.components.scalar_fp8_linear import FP8_DTYPE
from mstar.model.ministral3.checkpoint import HF_FILENAME, checkpoint_node_keys, load_vision_weights, text_tensor_schema
from mstar.model.ministral3.components.vision import Ministral3Vision, PixtralConfig
from mstar.model.ministral3.config import ATTN, LLM_NODE, Ministral3TextConfig
from mstar.model.ministral3.fp8 import FP8_TEXT_RULES
from mstar.model.ministral3.ministral3_model import Ministral3Model
from mstar.model.ministral3.submodules import Ministral3Decoder
from mstar.model.ministral3.vision import VISION_NODE, Ministral3VisionEncoder

checkpoint = _checkpoint


@pytest.fixture
def payload(checkpoint, tmp_path):
    directory = tmp_path / "model"
    shutil.copytree(checkpoint, directory)
    raw = json.loads((directory / "config.json").read_text())
    config = Ministral3TextConfig.from_hf(directory, tokenizer_eos_token_id=2)
    vision_config = PixtralConfig.from_hf(raw)
    generator = torch.Generator().manual_seed(21)
    values = {}
    for name, (shape, dtype) in text_tensor_schema(config).items():
        if name.endswith(("activation_scale", "weight_scale_inv")):
            values[name] = torch.tensor(0.017 + len(values) * .0001, dtype=torch.bfloat16)
        else:
            values[name] = torch.randn(shape, generator=generator).to(
                FP8_DTYPE if dtype == "F8_E4M3" else torch.bfloat16)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        values.update(Ministral3Vision(vision_config).to(dtype=torch.bfloat16).state_dict())
    save_file(values, directory / HF_FILENAME)
    return directory, values


@pytest.fixture
def cpu_factory(monkeypatch):
    # Do not patch the component's execution guard or matmul.
    monkeypatch.setattr(model_module, "require_scalar_fp8_device", lambda device: None)


def test_both_factories_load_only_their_own_tensors_and_cache_nodes(payload, cpu_factory, monkeypatch):
    directory, values = payload
    model = Ministral3Model(str(directory))
    reader = model_module.iter_safetensors_file
    reads, closed = [], []

    def observed(*args, **kwargs):
        names = []
        reads.append(names)
        try:
            yield from recording(reader(*args, **kwargs), names)
        finally:
            closed.append(True)

    def recording(source, names):
        for name, tensor in source:
            names.append(name)
            yield name, tensor

    monkeypatch.setattr(model_module, "iter_safetensors_file", observed)
    decoder = model.get_submodule(LLM_NODE, device="cpu")
    vision = model.get_submodule(VISION_NODE, device="cpu")
    assert isinstance(decoder, Ministral3Decoder) and isinstance(vision, Ministral3VisionEncoder)
    assert decoder._encoders == model.encoders
    assert decoder.ATTN_KEY == ATTN
    assert decoder.model.lm_head.weight is decoder.model.embed_tokens.weight
    assert model.get_submodule(LLM_NODE, device="cpu") is decoder
    assert model.get_submodule(VISION_NODE, device="cpu") is vision
    assert len(reads) == len(closed) == 2
    assert set(reads[0]).isdisjoint(reads[1])
    assert set(reads[0]) | set(reads[1]) == set(values)
    for node, names in ((decoder, reads[0]), (vision, reads[1])):
        assert not node.training and not any(p.requires_grad for p in node.parameters())
        assert resolve_node_dtype(node, None) == torch.bfloat16
        node.to(device="cpu", dtype=resolve_node_dtype(node, torch.bfloat16))
        params = dict(node.model.named_parameters())
        for name in names:
            target = FP8_TEXT_RULES.native(name) if node is decoder else name
            param, source = params[target], values[name]
            expected = source.view(torch.uint8) if source.dtype == FP8_DTYPE else source.to(param.dtype)
            torch.testing.assert_close(param, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="requires NVIDIA"):
        decoder.model.layers[0].self_attn._project_qkv(decoder.model.embed(torch.tensor([2])))


@torch.no_grad()
def test_loaded_vision_forward_matches_direct_weight_assignment(payload):
    directory, values = payload
    model = Ministral3Model(str(directory))
    node = model.get_submodule(VISION_NODE, device="cpu")
    reference = Ministral3Vision(model.vision_config).to(dtype=torch.bfloat16).eval()
    reference.load_state_dict({name: values[name] for name in reference.state_dict()}, strict=True)
    coords = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    patches = torch.linspace(-1, 1, 48).reshape(4, 12)
    outputs = []
    for tower in (node.model, reference):
        bind(tower, [4])
        hidden, cos, sin = tower.embed(patches, coords)
        outputs.append(tower.project(tower.encode(hidden, cos, sin)))
    assert outputs[0].dtype == torch.bfloat16
    assert torch.isfinite(outputs[0]).all()
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("node", [LLM_NODE, VISION_NODE])
@pytest.mark.parametrize("damage", ["missing_text_scale", "missing_vision", "unknown", "dtype", "shape"])
def test_bad_header_fails_before_allocation_and_does_not_cache(payload, cpu_factory, node, damage, monkeypatch):
    directory, values = payload
    bad = dict(values)
    if damage == "missing_text_scale":
        del bad["language_model.model.layers.0.self_attn.q_proj.activation_scale"]
    elif damage == "missing_vision":
        del bad["multi_modal_projector.linear_2.weight"]
    elif damage == "unknown":
        bad["vision_tower.unexpected.weight"] = torch.ones(1, dtype=torch.bfloat16)
    elif damage == "dtype":
        bad["vision_tower.ln_pre.weight"] = bad["vision_tower.ln_pre.weight"].float()
    else:
        bad["vision_tower.ln_pre.weight"] = bad["vision_tower.ln_pre.weight"][:-1]
    save_file(bad, directory / HF_FILENAME)
    model = Ministral3Model(str(directory))
    with monkeypatch.context() as ctx:
        ctx.setattr(model_module, "iter_safetensors_file",
                    lambda *a, **k: pytest.fail("payload read before validation"))
        with pytest.raises(ValueError):
            model.get_submodule(node, device="cpu")
    assert not model._submodule_cache
    save_file(values, directory / HF_FILENAME)
    assert model.get_submodule(node, device="cpu") is not None


def test_bad_scale_closes_iterator_and_leaves_factory_retryable(payload, cpu_factory, monkeypatch):
    directory, values = payload
    bad = dict(values)
    name = "language_model.model.layers.0.self_attn.q_proj.activation_scale"
    bad[name] = torch.tensor(0., dtype=torch.bfloat16)
    save_file(bad, directory / HF_FILENAME)
    reader = model_module.iter_safetensors_file
    closed = []

    def observed(*args, **kwargs):
        try:
            yield from reader(*args, **kwargs)
        finally:
            closed.append(True)

    monkeypatch.setattr(model_module, "iter_safetensors_file", observed)
    model = Ministral3Model(str(directory))
    with pytest.raises(ValueError, match="positive"):
        model.get_submodule(LLM_NODE, device="cpu")
    assert not model._submodule_cache and closed == [True]
    save_file(values, directory / HF_FILENAME)
    assert isinstance(model.get_submodule(LLM_NODE, device="cpu"), Ministral3Decoder)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "dtype", "shape", "unknown"])
def test_strict_vision_stream_loading(payload, damage):
    directory, values = payload
    config = PixtralConfig.from_hf(json.loads((directory / "config.json").read_text()))
    native = Ministral3Vision(config).to(dtype=torch.bfloat16)
    weights = [(name, values[name]) for name in dict(native.named_parameters())]
    if damage == "missing":
        weights.pop()
    elif damage == "duplicate":
        weights.append(weights[0])
    elif damage == "dtype":
        weights[0] = weights[0][0], weights[0][1].float()
    elif damage == "shape":
        weights[0] = weights[0][0], weights[0][1][:1]
    else:
        weights.append(("unexpected", torch.ones(1, dtype=torch.bfloat16)))
    with pytest.raises(ValueError):
        load_vision_weights(native, weights)


@pytest.mark.parametrize("node", [LLM_NODE, VISION_NODE])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_factory_rejects_dtype_changes_before_io(checkpoint, node, dtype, monkeypatch):
    model = Ministral3Model(str(checkpoint))
    monkeypatch.setattr(model, "_resolve_weights", lambda: pytest.fail("must reject before IO"))
    with pytest.raises(ValueError, match="BF16 activations"):
        model.get_submodule(node, autocast_dtype=dtype)


def test_local_weights_do_not_download_or_accept_consolidated(checkpoint, tmp_path, monkeypatch):
    directory = tmp_path / "metadata"
    shutil.copytree(checkpoint, directory)
    (directory / "consolidated.safetensors").write_bytes(b"not the HF format")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: pytest.fail("local paths must stay local"))
    model = Ministral3Model(str(directory))
    with pytest.raises(FileNotFoundError, match=HF_FILENAME):
        model.checkpoint_path()


def test_remote_resolution_fetches_only_hf_weights_at_metadata_revision(payload, monkeypatch):
    directory, _ = payload
    model = Ministral3Model(str(directory))
    model.model_path_hf = "mistralai/Ministral-3-3B-Instruct-2512"
    revision = "b35d4dfe56c142746f54dbd64f579faab2744308"
    model.local_dir = str(directory.parent / "snapshots" / revision)
    calls = []
    import huggingface_hub

    def download(**kwargs):
        calls.append(kwargs)
        return str(directory)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert model._resolve_weights() == str(directory)
    assert calls == [{"repo_id": model.model_path_hf, "cache_dir": None,
                      "revision": revision, "allow_patterns": [HF_FILENAME]}]


def test_all_checkpoint_keys_have_exactly_one_node_owner(payload):
    directory, values = payload
    model = Ministral3Model(str(directory))
    text, vision = checkpoint_node_keys(directory / HF_FILENAME, model.config, model.vision_config)
    assert not text & vision and text | vision == values.keys()
    sharding = model.get_default_sharding_config()
    assert not sharding.tp_enabled_nodes and not sharding.sp_enabled_nodes
    assert set(model.get_graph_walk_graphs()["prefill_vision"].get_nodes()) == {LLM_NODE, VISION_NODE}
