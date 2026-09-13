from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, ".")

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.registry import HF_MODELS, get_model_class
from mstar.model.waypoint.checkpoint import (
    TAEHV_CHECKPOINT_FILE,
    WAYPOINT_WEIGHT_ALLOW_PATTERNS,
    require_taehv_runtime,
    resolve_taehv_checkpoint,
    resolve_waypoint_checkpoint,
)
from mstar.model.waypoint.config import (
    WAYPOINT_VARIANT_360P,
    WAYPOINT_VARIANT_720P,
    WAYPOINT_VARIANT_HF_REPOS,
    WaypointConfig,
    waypoint_1_5_1b_360p,
    waypoint_1_5_1b_720p,
)
from mstar.model.waypoint.waypoint_model import (
    DIT_NODE,
    VAE_DECODER_NODE,
    VAE_ENCODER_NODE,
    WaypointModel,
)
from mstar.worker.engine_manager import EngineManager


def _manifest(geometry: tuple[int, int, int] = (512, 16, 32)) -> dict:
    tokens_per_frame, height, width = geometry
    return {
        "model_type": "waypoint-1.5",
        "inference_fps": 60,
        "temporal_compression": 4,
        "taehv_ae": True,
        "ae_uri": "Overworld-Models/taehv1_5",
        "prompt_conditioning": None,
        "channels": 32,
        "n_layers": 24,
        "n_heads": 32,
        "n_kv_heads": 16,
        "d_model": 2048,
        "mlp_ratio": 4,
        "causal": True,
        "moe": False,
        "n_buttons": 256,
        "tokens_per_frame": tokens_per_frame,
        "height": height,
        "width": width,
        "conv_kw": {"kernel_size": [2, 2], "stride": [2, 2]},
        "patch": [2, 2],
        "base_fps": 15,
        "local_window": 16,
        "global_window": 128,
        "global_pinned_dilation": 8,
        "global_attn_period": 4,
        "global_attn_offset": -1,
        "n_frames": 512,
        "rope_impl": "ortho",
        "value_residual": True,
        "gated_attn": False,
        "noise_conditioning": "wan",
        "ctrl_conditioning": True,
        "ctrl_cond_dropout": 0.0,
        "ctrl_conditioning_period": 3,
        "scheduler_sigmas": [1.0, 0.9, 0.75, 0.3, 0.0],
    }


def _checkpoint(tmp_path: Path, manifest: dict | None = None) -> Path:
    directory = tmp_path / "waypoint"
    directory.mkdir()
    (directory / "config.yaml").write_text(yaml.safe_dump(manifest or _manifest()))
    (directory / "model.safetensors").touch()
    return directory


def test_serving_defaults_to_reference_compatible_optimized_execution():
    for config in (waypoint_1_5_1b_720p(), waypoint_1_5_1b_360p()):
        assert config.reference_compat is True
        assert config.compile_dit is True
        assert config.cuda_graph is True


@pytest.mark.parametrize("compile_dit", [False, True])
@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("reference_compat", [False, True])
def test_execution_and_numerical_modes_are_independent(
    compile_dit, cuda_graph, reference_compat,
):
    config = replace(
        waypoint_1_5_1b_720p(),
        compile_dit=compile_dit,
        cuda_graph=cuda_graph,
        reference_compat=reference_compat,
    )
    config.validate_supported_deployment()
    assert config.compile_dit is compile_dit
    assert config.cuda_graph is cuda_graph
    assert config.reference_compat is reference_compat


def test_model_constructor_threads_all_execution_modes_independently():
    model = WaypointModel(
        skip_weight_loading=True,
        compile_dit=False,
        cuda_graph=False,
        reference_compat=False,
    )
    assert model.config.compile_dit is False
    assert model.config.cuda_graph is False
    assert model.config.reference_compat is False


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"scheduler_sigmas": (1.0, 0.3, 0.4, 0.0)}, "strictly descending"),
        ({"scheduler_sigmas": (1.0, 0.3)}, "end at 0.0"),
        ({"inference_fps": 59}, "temporal_compression"),
        ({"base_fps": 16}, "latent fps"),
        ({"patch": (2, 0)}, "two positive integers"),
        ({"d_model": 2016}, "OrthoRoPE"),
    ],
)
def test_config_rejects_invalid_scheduler_geometry_and_fps(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        WaypointConfig(**kwargs)


@pytest.mark.parametrize(
    "changes,fragment",
    [
        ({"tokens_per_frame": 128, "height": 8, "width": 16}, "requires.*tokens_per_frame"),
        ({"d_model": 4096}, "d_model"),
        ({"global_window": 256}, "global_window"),
        ({"scheduler_sigmas": (1.0, 0.5, 0.0)}, "scheduler_sigmas"),
        ({"base_fps": 30}, "base_fps"),
    ],
)
def test_deployment_validation_rejects_checkpoint_fact_drift(changes, fragment):
    config = replace(waypoint_1_5_1b_720p(), **changes)
    with pytest.raises(ValueError, match=fragment):
        config.validate_supported_deployment()


def test_local_checkpoint_is_validated_without_importing_huggingface(tmp_path, monkeypatch):
    directory = _checkpoint(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert resolve_waypoint_checkpoint(directory, waypoint_1_5_1b_720p()) == directory.resolve()


@pytest.mark.parametrize(
    ("config", "geometry"),
    [
        (waypoint_1_5_1b_720p(), (512, 16, 32)),
        (waypoint_1_5_1b_360p(), (128, 8, 16)),
    ],
)
def test_selected_variant_accepts_only_its_manifest_geometry(tmp_path, config, geometry):
    directory = _checkpoint(tmp_path, _manifest(geometry))
    assert resolve_waypoint_checkpoint(directory, config) == directory.resolve()


@pytest.mark.parametrize(
    ("config", "wrong_geometry"),
    [
        (waypoint_1_5_1b_720p(), (128, 8, 16)),
        (waypoint_1_5_1b_360p(), (512, 16, 32)),
    ],
)
def test_selected_variant_rejects_the_other_manifest_geometry(
    tmp_path, config, wrong_geometry
):
    directory = _checkpoint(tmp_path, _manifest(wrong_geometry))
    with pytest.raises(ValueError, match=rf"checkpoint geometry.*{config.variant}"):
        resolve_waypoint_checkpoint(directory, config)


@pytest.mark.parametrize(
    "key,value,fragment",
    [
        ("n_layers", 23, "n_layers"),
        ("tokens_per_frame", 129, "checkpoint geometry"),
        ("scheduler_sigmas", [1.0, 0.5, 0.0], "scheduler_sigmas"),
        ("inference_fps", 30, "inference_fps"),
    ],
)
def test_manifest_mismatch_fails_before_loading_weights(tmp_path, key, value, fragment):
    manifest = _manifest()
    manifest[key] = value
    directory = _checkpoint(tmp_path, manifest)
    with pytest.raises(ValueError, match=fragment):
        resolve_waypoint_checkpoint(directory, waypoint_1_5_1b_720p())


def test_missing_or_partial_local_checkpoint_fails_clearly(tmp_path):
    directory = tmp_path / "partial"
    directory.mkdir()
    (directory / "config.yaml").write_text(yaml.safe_dump(_manifest()))
    with pytest.raises(FileNotFoundError, match="no model.safetensors"):
        resolve_waypoint_checkpoint(directory, waypoint_1_5_1b_720p())
    with pytest.raises(FileNotFoundError, match="local path does not exist"):
        resolve_waypoint_checkpoint(tmp_path / "typo", waypoint_1_5_1b_720p())


def test_hf_resolution_downloads_only_native_checkpoint_files(tmp_path, monkeypatch):
    directory = _checkpoint(tmp_path)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(directory)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download)
    )
    resolved = resolve_waypoint_checkpoint(
        "Overworld/Waypoint-1.5-1B",
        waypoint_1_5_1b_720p(),
        cache_dir=tmp_path / "cache",
        revision="weights-revision",
    )
    assert resolved == directory.resolve()
    assert calls == [
        {
            "repo_id": "Overworld/Waypoint-1.5-1B",
            "cache_dir": str(tmp_path / "cache"),
            "revision": "weights-revision",
            "allow_patterns": list(WAYPOINT_WEIGHT_ALLOW_PATTERNS),
        }
    ]


def test_taehv_local_and_hf_resolution_fetch_exactly_one_file(tmp_path, monkeypatch):
    ae_dir = tmp_path / "ae"
    ae_dir.mkdir()
    checkpoint = ae_dir / TAEHV_CHECKPOINT_FILE
    checkpoint.touch()
    assert resolve_taehv_checkpoint(ae_dir) == checkpoint.resolve()

    calls = []

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(checkpoint)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download)
    )
    assert resolve_taehv_checkpoint(
        "Overworld-Models/taehv1_5", revision="ae-revision"
    ) == checkpoint.resolve()
    assert calls == [
        {
            "repo_id": "Overworld-Models/taehv1_5",
            "filename": TAEHV_CHECKPOINT_FILE,
            "cache_dir": None,
            "revision": "ae-revision",
        }
    ]


@pytest.mark.parametrize("installed", [None, types.SimpleNamespace()])
def test_taehv_runtime_preflight_fails_clearly(installed, monkeypatch):
    monkeypatch.setitem(sys.modules, "taehv", installed)
    with pytest.raises(RuntimeError, match=r"TAEHV.*\.\[waypoint\].*(uv|pip)"):
        require_taehv_runtime()


@pytest.mark.parametrize(
    ("variant", "expected_source"),
    list(WAYPOINT_VARIANT_HF_REPOS.items()),
)
def test_model_resolves_variant_artifacts_before_first_module_allocation(
    monkeypatch, variant, expected_source
):
    calls = []

    def resolve_waypoint(source, config, **kwargs):
        calls.append(("waypoint", source, kwargs))
        return Path("/resolved/waypoint")

    def resolve_taehv(source, **kwargs):
        calls.append(("taehv", source, kwargs))
        return Path("/resolved/taehv1_5.pth")

    def build_dit(config, checkpoint_dir, device):
        calls.append(("allocate", checkpoint_dir, device))
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(
        "mstar.model.waypoint.checkpoint.require_taehv_runtime",
        lambda: calls.append(("runtime",)),
    )
    monkeypatch.setattr(
        "mstar.model.waypoint.checkpoint.resolve_waypoint_checkpoint",
        resolve_waypoint,
    )
    monkeypatch.setattr(
        "mstar.model.waypoint.checkpoint.resolve_taehv_checkpoint", resolve_taehv,
    )
    monkeypatch.setattr(
        "mstar.model.waypoint.weight_loader.build_waypoint_dit", build_dit,
    )
    model = WaypointModel(
        **HF_MODELS["waypoint"],
        variant=variant,
        ae_path="Overworld-Models/taehv1_5",
        cache_dir="/cache",
        checkpoint_revision="dit-rev",
        ae_revision="ae-rev",
    )

    model.get_submodule(DIT_NODE)

    assert [entry[0] for entry in calls] == ["runtime", "waypoint", "taehv", "allocate"]
    assert calls[1][1:] == (
        expected_source,
        {"cache_dir": "/cache", "revision": "dit-rev"},
    )
    assert calls[2][1:] == (
        "Overworld-Models/taehv1_5",
        {"cache_dir": "/cache", "revision": "ae-rev"},
    )
    assert calls[3][1] == "/resolved/waypoint"


def test_registry_hub_id_uses_the_waypoint_startup_contract():
    waypoint = get_model_class("waypoint")
    assert waypoint is WaypointModel
    assert HF_MODELS["waypoint"] == {"model_path_hf": None}
    assert WAYPOINT_VARIANT_HF_REPOS == {
        WAYPOINT_VARIANT_720P: "Overworld/Waypoint-1.5-1B",
        WAYPOINT_VARIANT_360P: "Overworld/Waypoint-1.5-1B-360P",
    }


@pytest.mark.parametrize("variant", [WAYPOINT_VARIANT_720P, WAYPOINT_VARIANT_360P])
def test_registry_default_selects_the_repository_for_the_variant(variant):
    model = get_model_class("waypoint")(
        **HF_MODELS["waypoint"], variant=variant, skip_weight_loading=True
    )
    assert model.model_path_hf == WAYPOINT_VARIANT_HF_REPOS[variant]
    assert model.checkpoint_dir == WAYPOINT_VARIANT_HF_REPOS[variant]


@pytest.mark.parametrize(
    "explicit_source",
    ["Example/custom-waypoint", "/models/local-waypoint"],
)
def test_explicit_model_source_is_not_rewritten_for_the_selected_variant(explicit_source):
    model = WaypointModel(
        model_path_hf=explicit_source,
        variant=WAYPOINT_VARIANT_360P,
        skip_weight_loading=True,
    )
    assert model.model_path_hf == explicit_source
    assert model.checkpoint_dir == explicit_source


def test_explicit_checkpoint_directory_overrides_the_variant_repository():
    model = WaypointModel(
        model_path_hf=None,
        checkpoint_dir="/models/local-waypoint",
        variant=WAYPOINT_VARIANT_360P,
        skip_weight_loading=True,
    )
    assert model.model_path_hf == WAYPOINT_VARIANT_HF_REPOS[WAYPOINT_VARIANT_360P]
    assert model.checkpoint_dir == "/models/local-waypoint"


def test_shipped_config_builds_through_registry_and_engine_manager_without_network():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "waypoint.yaml"
    model_config = yaml.safe_load(config_path.read_text())
    model_class = get_model_class(model_config["model"])
    model = model_class(
        **HF_MODELS[model_config["model"]],
        **model_config["model_kwargs"],
        skip_weight_loading=True,
    )

    model.get_worker_graphs(str(config_path))
    manager = EngineManager.build(
        node_names={DIT_NODE, VAE_ENCODER_NODE, VAE_DECODER_NODE},
        device=torch.device("cpu"),
        model_config=model_config,
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="test",
            my_session_id="test",
            transfer_engine=LocalTransferEngine("test"),
        ),
        model=model,
    )
    try:
        assert manager.node_names == {DIT_NODE, VAE_ENCODER_NODE, VAE_DECODER_NODE}
        assert manager.engine._autocast_dtype is torch.bfloat16
        assert model.config.reference_compat is True
        assert model.config.compile_dit is True
        assert model.config.cuda_graph is True
        assert model.checkpoint_dir == "Overworld/Waypoint-1.5-1B"
        assert model._checkpoints_resolved is False
    finally:
        manager.shutdown()
