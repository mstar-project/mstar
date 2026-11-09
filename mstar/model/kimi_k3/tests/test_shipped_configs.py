"""Every shipped Kimi config (``configs/kimi_k3*.yaml``) boots through the startup a worker runs, on
the CPU: the conductor's ``validate_deployment`` at the release's geometry, then
``EngineManager.build`` (the submodule's load under its quantization scope, the post-load checks,
the engine's resources) on the tiny MXFP4 release with the config's model kwargs and resources, one
rank. B300 04c: tp8 and tp8_c128 died at startup on every rank because the LLM node chunks its
prefill without reporting ``get_input_sequence_len``, a check no engine-level test reaches."""
import types
from pathlib import Path

import pytest
import torch
import yaml

from mstar.model.kimi_k3.tests import engine_helpers as H
from mstar.model.kimi_k3.tests.test_truncate import make_tiny_release

ROOT = Path(__file__).resolve().parents[4]
CONFIGS = sorted((ROOT / "configs").glob("kimi_k3*.yaml"))
# the engine pool a test allocates on the CPU (the tiny geometry); the shipped size is validated above it
_MAX_TEST_PAGES = 256


def _model_class(cfg: dict):
    from mstar.model.kimi_k3.kimi_k3_model import KimiK3Model, KimiK31Model

    return {"kimi_k31": KimiK31Model, "kimi_k3": KimiK3Model}[cfg["model"]]


_RELEASES = {"kimi_k31": "moonshotai/Kimi-K3.1-Pro", "kimi_k3": "moonshotai/Kimi-K3"}


def _release_kwargs(cfg: dict) -> dict:
    """The config's model kwargs, minus what points at a checkpoint (the 04c item adds the path)."""
    return {k: v for k, v in (cfg.get("model_kwargs") or {}).items() if k != "checkpoint_path"}


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    H.install_cpu_stubs(monkeypatch)
    torch.set_num_threads(2)


def test_every_kimi_config_is_covered():
    assert {p.name for p in CONFIGS} >= {"kimi_k31_tp8.yaml", "kimi_k31_tp8_c128.yaml", "kimi_k31_ep8.yaml",
                                         "kimi_k31_tp8_longctx.yaml", "kimi_k3_tp8.yaml"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_the_shipped_config_passes_the_deployment_check(path):
    from mstar.validation import validate_deployment

    cfg = yaml.safe_load(path.read_text())
    model = _model_class(cfg)(_RELEASES[cfg["model"]], **_release_kwargs(cfg))
    validate_deployment(model, cfg, str(path))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_the_shipped_config_boots_a_worker(path, tmp_path):
    """``EngineManager.build`` as a worker calls it, one rank: the node groups at TP1 and the KV pool
    capped to the tiny geometry's test size; every other key as shipped."""
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.resources.kv.transfer import TransferEngineInfo
    from mstar.worker.engine_manager import EngineManager

    make_tiny_release(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["node_groups"] = [dict(g, ranks=[0], tp_size=1) for g in cfg["node_groups"]]
    for pool in cfg.get("resources", {}).values():
        if isinstance(pool, dict) and "max_num_pages" in pool:
            pool["max_num_pages"] = min(pool["max_num_pages"], _MAX_TEST_PAGES)
    kwargs = dict(_release_kwargs(cfg), strict_config=False, tokenizer_mode="byte", kda_conv_dtype=torch.float32)
    model = _model_class(cfg)(str(tmp_path), **kwargs)
    manager = EngineManager.build(
        node_names={"LLM"},
        device=torch.device("cpu"),
        model_config={**cfg, "autocast_dtype": torch.float32},
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e0", my_session_id="s", transfer_engine=LocalTransferEngine("localhost")),
        model=model,
        graph_runtime=types.SimpleNamespace(),
    )
    sub = manager.engine._submodules["LLM"].submodule
    assert sub.get_input_sequence_len("prefill", None, {"text_inputs": [torch.zeros(37, dtype=torch.long)]}) \
        == (37, 0)


@pytest.mark.parametrize("name", ["kimi_k31", "kimi_k3"])
def test_the_openai_chat_endpoint_serves_kimi_with_the_benches_fields(name, tmp_path):
    """/v1/chat/completions has an adapter for Kimi (B300 04c: without one it answered 404 after a ~10 min
    load), and the fields the benches send reach the model: ``max_tokens`` caps the reply, ``ignore_eos``
    reaches the sampler, and ``enable_thinking: false`` (flat, M*'s; or in ``chat_template_kwargs``,
    vLLM's) renders the prompt without thinking, as vLLM's item 02b replies were."""
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import ChatCompletionRequest
    from mstar.model.kimi_k3.config import SAMPLER

    adapter = get_adapter(name)
    assert adapter is not None and adapter.supports_chat
    model = _model_class({"model": name})(_RELEASES[name])

    class _Template:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            return [1, 2, 3]

    model._tokenizer = template = _Template()
    for extra, thinks in (({}, True), ({"enable_thinking": False}, False),
                          ({"chat_template_kwargs": {"enable_thinking": False}}, False),
                          ({"reasoning_effort": "none"}, False)):
        req = ChatCompletionRequest(model=name, messages=[{"role": "user", "content": "Say hi."}], max_tokens=7,
                                    temperature=0, stream=True, ignore_eos=True, **extra)
        args = adapter.chat_to_request(req, tmp_path)
        mk = args.model_kwargs
        assert model.get_max_output_tokens(**mk) == 7
        assert model.get_request_resource_configs({}, mk)[SAMPLER].ignore_eos is True
        model.process_prompt(args.text, ["text"], ["text"], prompt_parts=args.prompt_parts, **mk)
        assert template.calls[-1]["thinking"] is thinks, extra
