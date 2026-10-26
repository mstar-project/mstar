"""Prefix-cache keys name the LoRA adapter a request runs on.

The same tokens fill different KV pages under two adapters, so a request on an adapter
chains its page keys from a root that names the adapter's weights (a digest read once,
when the adapter is registered, and taken when the API server selects the adapter); a
base-model request chains from the empty root exactly as before LoRA.
"""

from __future__ import annotations

import hashlib
import queue
import threading
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

from mstar.api_server.data_worker import PreprocessWorkerThread  # noqa: E402
from mstar.api_server.request_types import PreprocessInput  # noqa: E402
from mstar.api_server.served_models import ServedModels  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, PagedKVConfig, PrefixSpan  # noqa: E402
from mstar.engine.resources.kv.keys import chain, page_key  # noqa: E402
from mstar.engine.resources.kv.manager import PrefixChain  # noqa: E402
from mstar.engine.resources.lora import (  # noqa: E402
    LORA_ADAPTER_KWARG,
    LORA_RESOURCE_KEY,
    LoraModuleRef,
    LoraSpec,
    LoraTarget,
)
from mstar.engine.resources.lora.identity import adapter_prefix_root  # noqa: E402
from mstar.engine.resources.lora.serving import lora_spec, resolve_lora_adapters  # noqa: E402
from mstar.engine.resources.lora.store import LoraAdapterStore  # noqa: E402
from mstar.model.base import (  # noqa: E402
    KVLabel,
    LabelLayout,
    PrefixKeys,
    ProcessPromptOutput,
    Span,
    serving_resources,
)

PAGE_SIZE = 16
NODE, WALK = "LLM", "prefill"
TENSOR = "text_inputs"
PROMPT = list(range(100, 200))  # six whole pages and a tail of four


def _lora_spec() -> LoraSpec:
    return LoraSpec(
        resource_key=LORA_RESOURCE_KEY, nodes={NODE}, num_layers=1,
        targets={"q": LoraTarget(in_features=8, out_slices=(8,))},
        module_map=lambda m: LoraModuleRef("q", 0) if m == "layers.0.q" else None,
        backend="torch",
    )


class _Model:
    """Declares one id-keyed label and a LoRA spec."""

    def kv_labels(self):
        return {"kv": {"main": KVLabel(keys=PrefixKeys(TENSOR, (WALK,)))}}

    def max_input_files(self):
        return {}

    def get_node_resources(self):
        return [KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, max_num_pages=64, page_size=128,
        )), _lora_spec()]

    def process_prompt(self, *args, **kwargs):
        layout = LabelLayout(spans=(Span("ids", WALK, len(PROMPT), len(PROMPT), TENSOR),))
        return ProcessPromptOutput(
            {TENSOR: [torch.tensor(PROMPT, dtype=torch.long)]}, {"kv_layout": {"kv": {"main": layout}}},
        )


def _write_adapter(path, seed: int) -> str:
    g = torch.Generator().manual_seed(seed)
    save_file({
        "base_model.model.layers.0.q.lora_A.weight": torch.randn(2, 8, generator=g),
        "base_model.model.layers.0.q.lora_B.weight": torch.randn(8, 2, generator=g),
    }, str(path))
    return str(path)


@pytest.fixture
def adapters(tmp_path):
    # "alias" holds a's weights under another name
    return {name: _write_adapter(tmp_path / f"{name}.safetensors", seed)
            for name, seed in (("a", 1), ("b", 2), ("alias", 1))}


def _deployment(adapters: dict[str, str] | None) -> dict:
    resources = {"kv": {"page_size": PAGE_SIZE}}
    if adapters is not None:
        resources[LORA_RESOURCE_KEY] = {"adapters": dict(adapters)}
    return {"model": "stub", "resources": resources}


def _worker(model, config: dict) -> PreprocessWorkerThread:
    sent = []
    return PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(), profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(), abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(), discard_tensor_queue=queue.Queue(),
        stop_event=threading.Event(), model=model, model_config=config,
        communicator=SimpleNamespace(sent=sent, get_all_new_messages=lambda: [], send=lambda _, m: sent.append(m)),
        tensor_manager=SimpleNamespace(store_and_return_tensor_info=lambda *a, **k: {},
                                       register_for_send=lambda *a, **k: None, set_persist=lambda *a, **k: None),
    )


def _served(config: dict) -> ServedModels:
    """The API server's selection, from the adapters its startup check digested."""
    digests = resolve_lora_adapters(lora_spec(serving_resources(_Model(), config)), config)
    return ServedModels(base=("stub",), roots={n: adapter_prefix_root(d) for n, d in digests.items()})


_rids = iter(range(10**6))


def _run(
    worker: PreprocessWorkerThread, adapter: str | None = None, served: ServedModels | None = None, **extra,
) -> dict:
    """Preprocess one request, its adapter selected as the API server selects it, and
    return the kwargs the conductor was sent."""
    kwargs = dict(extra)
    if adapter is not None:
        kwargs[LORA_ADAPTER_KWARG] = adapter
    root = b""
    if served is not None:
        kwargs, root = served.select(None, kwargs)
    worker._process_input(PreprocessInput(
        request_id=f"req{next(_rids)}", text="hello", file_paths=None,
        input_modalities=["text"], output_modalities=["text"], model_kwargs=kwargs,
        prefix_root=root,
    ))
    return worker.communicator.sent[-1].body.model_kwargs


def _handed_over(model_kwargs: dict) -> KVReqConfig:
    """The KV config a request is opened with, as the conductor hands each config its chain."""
    cfg = KVReqConfig()
    cfg.apply_conductor_config(
        seed=1,
        prefix_keys=(model_kwargs.get("prefix_keys") or {}).get("kv"),
        prefix_root=model_kwargs.get("prefix_root"),
        prefix_cache=model_kwargs.get("prefix_cache"),
    )
    return cfg


def _pages(tokens: list[int]) -> list[list[int]]:
    return [tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)]


def _keys(model_kwargs: dict) -> list[bytes]:
    return model_kwargs["prefix_keys"]["kv"]["main"]


def _main_chain(pages: list[list[int]]) -> list[bytes]:
    """main's page keys, written out independently of ``keys.py``: SHA-256 over the
    length-prefixed parent key, little-endian int32 ids and an empty item field."""

    def field(payload: bytes) -> bytes:
        return len(payload).to_bytes(4, "little") + payload

    keys, prev = [], b""
    for tokens in pages:
        ids = b"".join(t.to_bytes(4, "little") for t in tokens)
        prev = hashlib.sha256(field(prev) + field(ids) + field(b"")).digest()
        keys.append(prev)
    return keys


def test_base_keys_are_byte_identical_to_mains(adapters):
    """A base request on a LoRA deployment sends what one did before LoRA, and what one
    on a deployment with no LoRA sends: same keys, same layout, no root."""
    with_lora = _run(_worker(_Model(), _deployment(adapters)))
    without = _run(_worker(_Model(), _deployment(None)))
    assert _keys(with_lora) == _main_chain(_pages(PROMPT))
    assert "prefix_root" not in with_lora
    prefix = {k: v for k, v in with_lora.items() if k.startswith("prefix_")}
    assert prefix == {k: v for k, v in without.items() if k.startswith("prefix_")}
    assert _handed_over(with_lora).prefix_root == b""


def test_each_adapters_requests_are_keyed_under_its_weights(adapters):
    config = _deployment(adapters)
    worker, served = _worker(_Model(), config), _served(config)
    planted = b"\x00" * 32  # a client cannot choose its root
    on_a = _run(worker, "a", served, prefix_root=planted)
    base = _run(worker, None, served, prefix_root=planted)
    on_b, on_alias = _run(worker, "b", served), _run(worker, "alias", served)
    # the root names the weights a worker's pool serves under the pinned digest
    spec = lora_spec(serving_resources(_Model(), config))
    pinned = LoraAdapterStore(spec, pin=False).load("a", adapters["a"], spec.adapter_digests["a"])
    root = adapter_prefix_root(pinned.digest)
    assert on_a["prefix_root"] == root and len(root) == 32
    assert "prefix_root" not in base and _handed_over(base).prefix_root == b""
    assert _keys(on_a) == chain(_pages(PROMPT), root=root)
    assert _handed_over(on_a).prefix_root == root and _handed_over(on_a).prefix_cache is True
    # no page shared between two adapters or an adapter and the base model; the root
    # names the weights, not the name
    assert len({_keys(on_a)[0], _keys(on_b)[0], _keys(base)[0]}) == 3
    assert _keys(on_alias) == _keys(on_a)


def test_control_without_a_root_adapters_share_the_base_models_keys(adapters):
    """Mechanism off: every adapter on the empty root keys exactly as the base model."""
    worker = _worker(_Model(), _deployment(adapters))
    assert _keys(_run(worker, "a")) == _keys(_run(worker, "b")) == _keys(_run(worker))


@pytest.mark.parametrize("root", [b"", adapter_prefix_root(b"\x07" * 32)])
def test_a_generated_first_page_hangs_from_the_requests_root(root):
    """A prompt shorter than a page leaves generation to key the first page."""
    tail = PROMPT[:10]
    generated = [900 + i for i in range(PAGE_SIZE - len(tail))]
    chain_ = PrefixChain.seed(
        chain([tail], root=root), list(tail), [PrefixSpan(len(tail), len(tail), WALK)], PAGE_SIZE, root=root,
    )
    chain_.extend(generated, PAGE_SIZE)
    assert chain_.keys[0] == page_key(root, tail + generated)
    if root == b"":
        assert chain_.keys[0] == _main_chain([tail + generated])[0]
    else:
        assert chain_.keys[0] != page_key(b"", tail + generated)
