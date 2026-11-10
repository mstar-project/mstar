"""Prefix-cache keys name the LoRA adapter a request runs on.

The same tokens fill different KV pages under two adapters, so a request on an adapter
chains its page keys from a root that names the adapter's weights (a digest read once,
when the adapter is registered, and taken when the API server selects the adapter, or
when the model's ``process_prompt`` sets it); a base-model request chains from the empty
root exactly as before LoRA. A resource's per-request ``prefix_variant`` folds into the root.
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
from mstar.engine.resources.kv.keys import chain, page_key, variant_root  # noqa: E402
from mstar.engine.resources.kv.manager import PrefixChain  # noqa: E402
from mstar.engine.resources.lora import (  # noqa: E402
    LORA_ADAPTER_KWARG,
    LORA_DIGEST_KWARG,
    LORA_RESOURCE_KEY,
    LoraModuleRef,
    LoraSpec,
    LoraTarget,
)
from mstar.engine.resources.lora.identity import adapter_prefix_root  # noqa: E402
from mstar.engine.resources.lora.serving import lora_spec, resolve_lora_adapters  # noqa: E402
from mstar.engine.resources.lora.store import LoraAdapterStore  # noqa: E402
from mstar.engine.resources.spec import NodeResourceSpec  # noqa: E402
from mstar.model.base import (  # noqa: E402
    KEEP_ADAPTER,
    KVLabel,
    LabelLayout,
    PrefixKeys,
    ProcessPromptOutput,
    Span,
    declared_resources,
    serving_resources,
)
from mstar.model.components.lora.modules import ModuleLoraMap  # noqa: E402
from mstar.testing_utils.model import StubModel

PAGE_SIZE = 16
NODE, WALK = "LLM", "prefill"
TENSOR = "text_inputs"
PROMPT = list(range(100, 200))  # six whole pages and a tail of four


def _lora_spec() -> LoraSpec:
    return LoraSpec(
        resource_key=LORA_RESOURCE_KEY, nodes={NODE}, num_layers=1,
        targets={"q": LoraTarget(in_features=8, out_slices=(8,))},
        module_map=ModuleLoraMap({"layers.0.q": LoraModuleRef("q", 0)}),
        backend="torch",
    )


class _Model(StubModel):
    """Declares one id-keyed label and a LoRA spec."""

    # names a checkpoint, so its prefix cache opens without a salt
    checkpoint_path = "stub"

    def kv_labels(self):
        return {"kv": {"main": KVLabel(keys=PrefixKeys(TENSOR, (WALK,)))}}

    def get_node_resources(self):
        return [KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, max_num_pages=64, page_size=128,
        )), _lora_spec(), *self.extra_specs]

    def process_prompt(self, *args, **kwargs):
        layout = LabelLayout(spans=(Span("ids", WALK, len(PROMPT), len(PROMPT), TENSOR),))
        return ProcessPromptOutput(
            {TENSOR: [torch.tensor(PROMPT, dtype=torch.long)]},
            {"kv_layout": {"kv": {"main": layout}}, **self.metadata}, self.lora_adapter,
        )

    def prompt_lora_adapters(self):
        return self.prompt_adapters

    # what process_prompt returns beside the prompt, and the adapters it may choose
    metadata: dict = {}
    lora_adapter = KEEP_ADAPTER
    prompt_adapters: frozenset[str] = frozenset()
    extra_specs: tuple = ()


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
    return {"model": "stub", "resources": resources, "node_groups": [{"ranks": [0], "node_names": [NODE]}]}


def _worker(model, config: dict, served: ServedModels | None = None) -> PreprocessWorkerThread:
    sent = []
    return PreprocessWorkerThread(served_models=served,
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(), profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(), abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(), discard_tensor_queue=queue.Queue(),
        stop_event=threading.Event(), model=model, model_config=config,
        communicator=SimpleNamespace(sent=sent, get_all_new_messages=lambda: [], send=lambda _, m: sent.append(m)),
        tensor_manager=SimpleNamespace(store_and_return_tensor_info=lambda *a, **k: {},
                                       register_for_send=lambda *a, **k: None, set_persist=lambda *a, **k: None,
                                       force_cleanup_request=lambda *a, **k: None),
    )


def _served(config: dict) -> ServedModels:
    """The API server's selection, from the adapters its startup check digested."""
    digests = resolve_lora_adapters(lora_spec(declared_resources(_Model(), config)), config)
    return ServedModels(base=("stub",), digests=digests)


_rids = iter(range(10**6))


def _input(adapter: str | None, served: ServedModels | None, **extra) -> PreprocessInput:
    """One request, its adapter selected as the API server selects it."""
    kwargs = dict(extra)
    if adapter is not None:
        kwargs[LORA_ADAPTER_KWARG] = adapter
    root = b""
    if served is not None:
        kwargs, root = served.select(None, kwargs)
    return PreprocessInput(
        request_id=f"req{next(_rids)}", text="hello", file_paths=None,
        input_modalities=["text"], output_modalities=["text"], model_kwargs=kwargs,
        prefix_root=root,
    )


def _run(
    worker: PreprocessWorkerThread, adapter: str | None = None, served: ServedModels | None = None, **extra,
) -> dict:
    """Preprocess one request and return the kwargs the conductor was sent."""
    worker._process_input(_input(adapter, served, **extra))
    return worker.communicator.sent[-1].body.model_kwargs


def _refusal(worker: PreprocessWorkerThread, adapter: str | None, served: ServedModels) -> tuple[int, str]:
    """Preprocess one request the data worker refuses: the status and text its client gets."""
    sent = len(worker.communicator.sent)
    worker._process_waiting_input(_input(adapter, served))
    assert len(worker.communicator.sent) == sent  # never reached the conductor
    chunk = worker.out_queue.get_nowait()
    return chunk.metadata["status"], chunk.data.decode()


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


def test_an_adapter_request_without_a_root_is_not_keyed(adapters):
    """Under the base model's empty root an adapter's pages would serve base requests, so
    a request on an adapter that reaches the data worker with no root is not cached."""
    worker = _worker(_Model(), _deployment(adapters))
    on_a = _run(worker, "a")
    assert on_a["prefix_cache"] is False and "prefix_keys" not in on_a
    assert _keys(_run(worker)) == _main_chain(_pages(PROMPT))


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



class _Choosing(_Model):
    """Runs every prompt on ``lora_adapter``, as Phi-4-multimodal picks its adapter from
    the prompt's media."""

    def __init__(self, lora_adapter=KEEP_ADAPTER, prompt_adapters=("a", "b"), **attrs):
        super().__init__()
        self.lora_adapter, self.prompt_adapters = lora_adapter, frozenset(prompt_adapters)
        self.__dict__.update(attrs)


class _VariantSpec(NodeResourceSpec):
    """A resource whose KV differs per request (a LongRoPE table): it names the table
    its config would be built from."""

    @property
    def resource_class(self):
        raise NotImplementedError

    def prefix_variant(self, model_kwargs):
        return model_kwargs.get("rope_table")


def _variants(**attrs) -> _Choosing:
    return _Choosing(extra_specs=(_VariantSpec(resource_key="rope", nodes={NODE}),), **attrs)


def test_a_model_chosen_adapter_is_keyed_as_a_named_one(adapters):
    """An adapter ``process_prompt`` sets gets the digest and root ``select`` gives a
    client that names it, so its pages are cached apart from the base model's."""
    config = _deployment(adapters)
    served = _served(config)
    named = _run(_worker(_Model(), config), "a", served)
    worker = _worker(_Choosing("a"), config, served)
    chosen, again = _run(worker, None, served), _run(worker, None, served)
    root = adapter_prefix_root(served.check("a"))
    assert chosen[LORA_ADAPTER_KWARG] == "a" and chosen[LORA_DIGEST_KWARG] == served.check("a")
    assert chosen["prefix_root"] == root and _handed_over(chosen).prefix_cache is True
    assert _keys(chosen) == _keys(named) == _keys(again) == chain(_pages(PROMPT), root=root)
    # a client naming the adapter the model picks is the same request
    assert _keys(_run(worker, "a", served)) == _keys(chosen)
    on_b = _run(_worker(_Choosing("b"), config, served), None, served)
    base = _run(_worker(_Choosing(None), config, served), None, served)
    assert "prefix_root" not in base and LORA_DIGEST_KWARG not in base
    assert _keys(base) == _main_chain(_pages(PROMPT))
    assert len({_keys(chosen)[0], _keys(on_b)[0], _keys(base)[0]}) == 3


def test_a_model_chosen_adapter_follows_runtime_unloads(adapters):
    """The data worker reads the API server's ``ServedModels``, so an adapter is checked
    against what is served now: one unloaded since startup is refused, as a client's is."""
    config = _deployment(adapters)
    served = _served(config)
    worker = _worker(_Choosing("b"), config, served)
    digest = served.remove("b")
    status, text = _refusal(worker, None, served)
    assert status == 400 and "unknown LoRA adapter 'b'" in text
    served.add("b", digest)
    assert _run(worker, None, served)["prefix_root"] == adapter_prefix_root(digest)


def test_a_model_choosing_another_adapter_than_the_client_is_refused(adapters):
    config = _deployment(adapters)
    served = _served(config)
    status, text = _refusal(_worker(_Choosing("a"), config, served), "b", served)
    assert status == 400 and "names LoRA adapter 'b'" in text and "'a'" in text
    # the base model, where the client named an adapter, is another adapter too
    assert _refusal(_worker(_Choosing(None), config, served), "b", served)[0] == 400


@pytest.mark.parametrize("output", [
    {"lora_adapter": "c"},  # not among its prompt_lora_adapters
    {"lora_adapter": ["a"]},
    {"metadata": {LORA_ADAPTER_KWARG: "a"}},  # in metadata, not the field
    {"extra_specs": (_VariantSpec(resource_key="rope", nodes={NODE}),), "metadata": {"rope_table": "long"}},
])
def test_a_model_hook_breaking_its_contract_is_a_server_error(adapters, output):
    """A model's or resource's bad value is a 500, never the client's refusal."""
    config = _deployment(adapters)
    served = _served(config)
    status, text = _refusal(_worker(_Choosing(**output), config, served), None, served)
    assert status == 500, text


def test_a_deployment_not_serving_the_models_adapters_is_refused_at_startup(adapters):
    """A model that may choose adapters needs a deployment serving them all."""
    model = _Choosing(prompt_adapters=("a", "vision"))
    with pytest.raises(ValueError, match=r"\['vision'\].*`resources: lora: adapters:`"):
        _served(_deployment(adapters)).refuse_unserved(model)
    with pytest.raises(ValueError, match=r"\['a', 'vision'\]"):
        ServedModels.of("stub", _deployment(None)).refuse_unserved(model)
    _served(_deployment({**adapters, "vision": adapters["a"]})).refuse_unserved(model)
    ServedModels.of("stub", _deployment(None)).refuse_unserved(_Model())


def test_a_request_the_model_sets_nothing_on_is_unchanged(adapters):
    """No adapter from ``process_prompt`` and no resource variant sends what main sends."""
    config = _deployment(adapters)
    served = _served(config)
    plain = _run(_worker(_Model(), config, served), None, served)
    assert _keys(plain) == _main_chain(_pages(PROMPT)) and "prefix_root" not in plain
    assert LORA_ADAPTER_KWARG not in plain and LORA_DIGEST_KWARG not in plain
    on_a = _run(_worker(_Model(), config, served), "a", served)
    assert on_a["prefix_root"] == adapter_prefix_root(served.check("a"))
    assert _keys(on_a) == chain(_pages(PROMPT), root=on_a["prefix_root"])
    # metadata that echoes the adapter the request was selected under changes nothing
    echoed = _run(_worker(_Choosing(metadata={LORA_ADAPTER_KWARG: "a"}), config, served), "a", served)
    assert echoed["prefix_root"] == on_a["prefix_root"] and _keys(echoed) == _keys(on_a)
    # a resource that names no variant for the request leaves its root as it was
    assert _keys(_run(_worker(_variants(), config, served), None, served)) == _keys(plain)


def test_prefix_variants_key_apart(adapters):
    """The same prompt under two variants of a resource shares no page; under one variant
    it does; and the variant folds in after the adapter's root, a model-chosen one's too."""
    config = _deployment(adapters)
    served = _served(config)

    def run(table, model_adapter=KEEP_ADAPTER, client_adapter=None):
        model = _variants(lora_adapter=model_adapter, metadata={"rope_table": table})
        return _run(_worker(model, config, served), client_adapter, served)

    short, short2, long = run(b"short"), run(b"short"), run(b"long")
    root = variant_root(b"", "rope", b"short")
    assert short["prefix_root"] == root and _handed_over(short).prefix_root == root
    assert _keys(short) == _keys(short2) == chain(_pages(PROMPT), root=root)
    assert len({_keys(short)[0], _keys(long)[0], _keys(run(None))[0]}) == 3
    adapter_root = variant_root(adapter_prefix_root(served.check("a")), "rope", b"short")
    chosen, named = run(b"short", model_adapter="a"), run(b"short", client_adapter="a")
    assert chosen["prefix_root"] == named["prefix_root"] == adapter_root
    assert _keys(chosen) == _keys(named) != _keys(short)
