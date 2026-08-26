"""Load a deployment the way the real conductor loads it — without GPUs.

Placement, worker-graph partitioning, walk topology, engine types, and
streaming chunk policies all come from calling mstar's own code, not from a
re-reading of the YAML. That is the whole point: if ``node_groups`` semantics
change, or a model rewires its graph, the simulator follows automatically
because it asks the same functions the server asks.

The one thing deliberately not done here is loading weights. Models build
their graph, config, and partition topology from the checkpoint's
``config.json`` alone; ``get_submodule`` — the only method that needs weights
— is never called. That is what lets a deployment be analyzed on a laptop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SimWorkerGraph:
    """One worker graph, resolved to the ranks that will run it."""

    wg_id: str
    section: Any            # GraphSection — deep-copied per request at runtime
    graph_walks: set[str]
    ranks: list[int]
    tp_size: int            # instance size (tp*sp), the lockstep routing unit
    sp_size: int
    tp_comm_size: int
    instance_ranks: list[list[int]]
    group_id: int
    consumes_stream: bool
    node_names: list[str] = field(default_factory=list)


@dataclass
class Deployment:
    """Everything the simulator needs about a model + placement."""

    model_key: str
    model: Any
    config_path: str
    config: dict
    #: walk name -> the worker graphs that walk decomposes into
    walk_to_wgs: dict[str, list[SimWorkerGraph]]
    #: every worker rank that appears in the placement
    ranks: list[int]
    #: node name -> EngineType value ("kv_cache" | "stateless")
    node_engine_types: dict[str, str]
    #: node name -> the ranks that run it (union across walks)
    node_to_ranks: dict[str, list[int]]
    partitions: list[Any]
    partition_topology: Any
    max_concurrent_requests: int | None
    max_output_tokens: int
    #: Bytes of KV cache one token occupies, summed over the model's cache
    #: configs. Used to price a prefill→decode handoff, where the whole
    #: context has to move between GPUs.
    kv_bytes_per_token: int = 0

    def workers(self) -> list[str]:
        return [f"worker_{r}" for r in self.ranks]

    def tp_size_for(self, node: str) -> int:
        for wgs in self.walk_to_wgs.values():
            for wg in wgs:
                if node in wg.node_names:
                    return wg.tp_comm_size
        return 1

    def sp_size_for(self, node: str) -> int:
        for wgs in self.walk_to_wgs.values():
            for wg in wgs:
                if node in wg.node_names:
                    return wg.sp_size
        return 1

    def describe(self) -> str:
        lines = [
            f"model:  {self.model_key} ({type(self.model).__name__})",
            f"config: {self.config_path}",
            f"ranks:  {self.ranks}",
            f"walks:  {len(self.walk_to_wgs)}",
        ]
        for walk, wgs in sorted(self.walk_to_wgs.items()):
            for wg in wgs:
                par = ""
                if wg.tp_comm_size > 1 or wg.sp_size > 1:
                    par = f" tp={wg.tp_comm_size} sp={wg.sp_size}"
                lines.append(
                    f"  {walk:<18} {','.join(wg.node_names):<28} "
                    f"ranks={wg.ranks}{par}"
                )
        return "\n".join(lines)


#: Tokenizer attributes a stub may safely answer. Every one is a token
#: *identity* — which id means end-of-sequence, which ids are special. The
#: simulator never tokenizes and never inspects token values, so these cannot
#: reach a graph shape, a placement, or a step cost. Anything outside this set
#: is refused, because a model that truly needs the tokenizer to build its
#: graph must not be simulated against a fabricated one.
_INERT_TOKENIZER_ATTRS: dict[str, Any] = {
    "all_special_ids": [],
    "all_special_tokens": [],
    "additional_special_tokens": [],
    "eos_token_id": 0,
    "bos_token_id": 0,
    "pad_token_id": 0,
    "unk_token_id": 0,
    "eos_token": "",
    "bos_token": "",
    "pad_token": "",
    "vocab_size": 0,
    "model_max_length": 0,
    "padding_side": "right",
}


class _StubTokenizer:
    """Stands in for a tokenizer the simulator will never call.

    Answers the inert metadata reads above and raises on everything else, so
    a model that genuinely depends on tokenization while building its graph
    surfaces as a clear error rather than as a silently wrong simulation.
    """

    def __getattr__(self, name: str):
        if name in _INERT_TOKENIZER_ATTRS:
            return _INERT_TOKENIZER_ATTRS[name]
        raise RuntimeError(
            f"the simulator stubbed out this model's tokenizer, but the model "
            f"asked for tokenizer.{name}. Construct the deployment with real "
            f"model weights/tokenizer access if this path is really needed."
        )

    def __len__(self) -> int:
        return 0


def _construct_without_tokenizer(
    model_cls: Any, model_path: str, cache_dir: str | None, model_kwargs: dict
) -> Any:
    """Build a model with ``AutoTokenizer.from_pretrained`` stubbed."""
    import transformers

    real = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _StubTokenizer()
    )
    try:
        return model_cls(
            model_path_hf=model_path, cache_dir=cache_dir, **model_kwargs
        )
    finally:
        transformers.AutoTokenizer.from_pretrained = real


def _kv_bytes_per_token(model: Any) -> int:
    """KV bytes one token occupies, from the model's own cache configs.

    Two tensors (K and V) per layer, each ``num_kv_heads × head_dim``
    elements. Dtype is taken as 2 bytes: mstar allocates the cache in the
    model's autocast dtype, which is 16-bit for every shipped model.
    Returns 0 when the model declares no KV cache — a pure diffusion or
    encoder deployment has no context to hand off.
    """
    total = 0
    try:
        configs = model.get_kv_cache_config() or []
    except Exception:
        return 0
    for cfg in configs:
        layers = getattr(cfg, "num_layers", 0) or 0
        heads = getattr(cfg, "num_kv_heads", 0) or 0
        dim = getattr(cfg, "head_dim", 0) or 0
        total += 2 * layers * heads * dim * 2
    return total


def _section_node_names(section: Any) -> list[str]:
    """Node names inside a graph section, in declaration order."""
    try:
        return list(section.get_nodes().keys())
    except Exception:
        return []


def load_deployment(
    config_path: str,
    model_key: str | None = None,
    cache_dir: str | None = None,
) -> Deployment:
    """Build a :class:`Deployment` from a config YAML.

    Instantiates the real model class (cheap — no weights) so that walk
    graphs, engine types, partitions, and the KV config come from the model
    itself rather than being restated here.
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}

    key = model_key or cfg.get("model")
    if not key:
        raise ValueError(f"{config_path} has no 'model:' key and none was given")

    from mstar.model.registry import HF_MODELS, get_model_class

    model_cls = get_model_class(key)
    model_kwargs = dict(cfg.get("model_kwargs") or {})
    model_path = HF_MODELS.get(key, {}).get("model_path_hf", "")

    try:
        model = model_cls(
            model_path_hf=model_path, cache_dir=cache_dir, **model_kwargs
        )
    except Exception as exc:
        # Several models load a tokenizer in __init__ — Orpheus, for one,
        # pulls it from a gated repo. The simulator never tokenizes anything:
        # it needs the walk graphs, engine types, and placement, all of which
        # come from the model's config. Refusing to analyze a deployment
        # because a tokenizer download is gated would be a bad trade, so
        # retry once with tokenizer construction stubbed out.
        logger.warning(
            "constructing %s normally failed (%s); retrying without a "
            "tokenizer — the simulator does not need one",
            key, exc,
        )
        model = _construct_without_tokenizer(
            model_cls, model_path, cache_dir, model_kwargs
        )

    # The real partitioning: same call the conductor makes at startup.
    worker_graphs = model.get_worker_graphs(config_path)

    walk_to_wgs: dict[str, list[SimWorkerGraph]] = {}
    ranks: set[int] = set()
    node_to_ranks: dict[str, set[int]] = {}

    for wg in worker_graphs:
        names = _section_node_names(wg.section)
        swg = SimWorkerGraph(
            wg_id=wg.worker_graph_id,
            section=wg.section,
            graph_walks=set(wg.graph_walks),
            ranks=list(wg.ranks),
            tp_size=wg.tp_size,
            sp_size=wg.sp_size,
            tp_comm_size=wg._tp_comm_size,
            instance_ranks=[list(b) for b in wg._instance_ranks] or [list(wg.ranks)],
            group_id=wg._group_id,
            consumes_stream=wg.consumes_stream,
            node_names=names,
        )
        for walk in swg.graph_walks:
            walk_to_wgs.setdefault(walk, []).append(swg)
        ranks.update(swg.ranks)
        for n in names:
            node_to_ranks.setdefault(n, set()).update(swg.ranks)

    engine_types = {
        node: et.value for node, et in model.get_node_engine_types().items()
    }

    return Deployment(
        kv_bytes_per_token=_kv_bytes_per_token(model),
        model_key=key,
        model=model,
        config_path=config_path,
        config=cfg,
        walk_to_wgs=walk_to_wgs,
        ranks=sorted(ranks),
        node_engine_types=engine_types,
        node_to_ranks={n: sorted(r) for n, r in node_to_ranks.items()},
        partitions=list(model.get_partitions()),
        partition_topology=model.get_partition_topology(),
        max_concurrent_requests=cfg.get("max_concurrent_requests"),
        max_output_tokens=model.get_max_output_tokens(),
    )
