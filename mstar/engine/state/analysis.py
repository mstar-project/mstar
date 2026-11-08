"""What the engine works out from declarations: the checks that need the model, and their digest."""

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import fields, is_dataclass
from itertools import product
from typing import TYPE_CHECKING, get_origin

import torch

from mstar.engine.state.contract import EMPTY, Array, Bidirectional, Block, GatedDeltaNet, Sampler, is_value

if TYPE_CHECKING:
    from mstar.model.base import Model


def state_declaration_problems(model: "Model") -> list[str]:
    """Every problem with ``model``'s declarations that its nodes and walks show.

    A model that declares nothing returns before its walk graphs are built, since building them can read checkpoints.
    """
    declarations = model.declare_state()
    if not declarations:
        return []
    arrays = [d for d in declarations if isinstance(d, Array)]
    samplers = [d for d in declarations if isinstance(d, Sampler)]
    problems = _name_problems(arrays, samplers)
    walks = {walk: set(graph.get_nodes()) for walk, graph in model.get_graph_walk_graphs().items()}
    nodes = sorted(set().union(*walks.values()))
    combined = {walk for by_name in model.get_combined_graph_walks().values() for walk in by_name}
    for array in arrays:
        problems += [
            f"Array {array.name!r}.nodes: {node!r} is not a node of the model, which has {nodes}"
            for node in sorted(array.nodes - set(nodes))
        ]
        problems += _read_problems(array, walks, combined)
        problems += _array_problems(array)
    problems += _kv_label_problems(model.kv_labels(), arrays)
    problems += _cross_read_problems(arrays)
    problems += _regrouping_problems(model, arrays)
    problems += _value_problems(arrays, model.kv_labels())
    for sampler in samplers:
        problems += _sampler_problems(sampler, nodes)
    return problems


def sdpa_nodes(node_groups: Sequence[Mapping[str, object]]) -> frozenset[str]:
    """The nodes whose node groups set ``attention_backend: sdpa``. A node listed by several groups, one per graph walk,
    must get the same answer from each."""
    asked: dict[str, str] = {}
    for group in node_groups:
        backend = group.get("attention_backend", "flashinfer")
        if backend not in ("flashinfer", "sdpa"):
            raise ValueError(f"attention_backend takes flashinfer or sdpa, not {backend!r}")
        for node in group["node_names"]:
            if asked.setdefault(node, backend) != backend:
                raise ValueError(f"node {node!r} gets different attention_backend settings from two node_groups")
    return frozenset(node for node, backend in asked.items() if backend == "sdpa")


def position_key(array: Array) -> str:
    """The resource key of ``array``'s positions: ``thinker_pos`` for ``thinker_kv``, ``rope`` for any other name."""
    return f"{array.name.removesuffix('_kv')}_pos" if array.name.endswith("_kv") else "rope"


def _name_problems(arrays: Sequence[Array], samplers: Sequence[Sampler]) -> list[str]:
    """Names share one namespace while they are resource keys, so a read can't reuse an array's name."""
    problems: list[str] = []
    declared: dict[str, str] = {}
    for field, name in (
        *(
            pair for array in arrays for pair in (
                (f"Array {array.name!r}.name", array.name),
                *((f"Array {array.name!r}.reads", key) for key in array.reads),
                *((f"Array {array.name!r}.updates", key) for key in array.updates),
                *(() if array.positions is None else ((f"Array {array.name!r}.positions", position_key(array)),)),
            )
        ),
        *((f"Sampler {sampler.name!r}.name", sampler.name) for sampler in samplers),
    ):
        if name in declared:
            problems.append(f"{field}: the name {name!r} is declared twice, also by {declared[name]}")
        declared.setdefault(name, field)
    return problems


def _array_problems(array: Array) -> list[str]:
    where = f"Array {array.name!r}"
    problems: list[str] = []
    for key, read in array.reads.items():
        if isinstance(read.mask, Block) and array.max_len:
            problems.append(f"{where}.reads[{key!r}].mask: Block needs max_len 0, and max_len is {array.max_len}")
        elif not isinstance(read.mask, Block) and not array.max_len:
            problems.append(f"{where}.reads[{key!r}].mask: every read of an array with max_len 0 is a Block read, "
                            f"and this one is {type(read.mask).__name__}")
    if array.positions is not None:
        if array.rewritable or not array.max_len:
            problems.append(f"{where}.positions: positions sit only on an append-only array that keeps entries")
        rope = array.positions.rope
        llama31 = () if rope is None else (rope.low_freq_factor, rope.high_freq_factor, rope.old_context_len)
        if None in llama31 and any(factor is not None for factor in llama31):
            problems.append(f"{where}.positions.rope: low_freq_factor, high_freq_factor and old_context_len are "
                            f"all set or all None, and they are {llama31}")
    kinds = [type(update) for update in array.updates.values()]
    for kind in sorted({kind for kind in kinds if kinds.count(kind) > 1}, key=lambda kind: kind.__name__):
        problems.append(f"{where}.updates: it carries {kinds.count(kind)} updates of type {kind.__name__}")
    return problems


def _kv_label_problems(kv_labels: Mapping[str, Mapping[str, object]], arrays: Sequence[Array]) -> list[str]:
    by_name = {array.name: array for array in arrays}
    problems: list[str] = []
    for key, labels in kv_labels.items():
        if key not in by_name:
            problems.append(f"kv_labels()[{key!r}]: it names no declared array, of {sorted(by_name)}")
            continue
        problems += [
            f"kv_labels()[{key!r}][{label!r}]: it is not in Array {key!r}.labels, {sorted(by_name[key].labels)}"
            for label in sorted(labels.keys() - by_name[key].labels)
        ]
    return problems


def _cross_read_problems(arrays: Sequence[Array]) -> list[str]:
    by_name = {array.name: array for array in arrays}
    problems: list[str] = []
    for array in arrays:
        for key, read in array.reads.items():
            if read.queries is None:
                continue
            where = f"Array {array.name!r}.reads[{key!r}].queries"
            queried = by_name.get(read.queries)
            if queried is None:
                problems.append(f"{where}: {read.queries!r} is not a declared array, of {sorted(by_name)}")
            elif not array.nodes <= queried.nodes:
                problems.append(f"{where}: {read.queries!r} isn't on nodes {sorted(array.nodes - queried.nodes)}, so a "
                                "step there has no segments to lay the queries out by")
    return problems


def _regrouping_problems(model: "Model", arrays: Sequence[Array]) -> list[str]:
    regrouped = [
        *((node, walk, "a chunking policy splits across steps", "leave it out of the policy")
          for node, policies in model.get_chunking_policies().items() for policy in policies if policy.chunkable
          for walk in policy.walks),
        *((node, walk, f"kv_labels()[{array.name!r}][{label!r}] keys for prefix reuse, and a hit can end inside a "
                       "step", "leave it out of the label's keys")
          for array in arrays for label, kv_label in model.kv_labels().get(array.name, {}).items()
          if kv_label.keys is not None for node in array.nodes
          for walk in (*kv_label.keys.walks, *(grown for grown, _ in kv_label.growth if kv_label.keys.growth))),
        *((node, policy.walk, f"{node}'s rebuild policy runs again in one step with the ids sampled after it",
           "drop the rebuild policy")
          for node, policy in model.rebuild_policies().items()),
    ]
    return [
        f"Array {array.name!r}.reads[{key!r}]: {type(read.mask).__name__} on walk {walk!r}, which {cause}. A Block or "
        f"Bidirectional read sees each step's adds as a whole, so {fix}"
        for node, walk, cause, fix in regrouped
        for array in arrays if node in array.nodes
        for key, read in array.reads.items()
        if isinstance(read.mask, Block | Bidirectional) and (read.walks is None or walk in read.walks)
    ]


def _value_problems(arrays: Sequence[Array], kv_labels: Mapping[str, Mapping[str, object]]) -> list[str]:
    """What a Value's entry can't hold yet. A prefix hit restores a Value's snapshot, and a snapshot carries neither its
    host fields nor a plain field, which has no kernel to write mid-prefill snapshots."""
    plain = [array for array in arrays if is_value(array) and array.entry.tensors and not array.updates]
    if not plain and not any(array.entry.host for array in arrays):
        return []
    keyed = {
        node for array in arrays if any(label.keys is not None for label in kv_labels.get(array.name, {}).values())
        for node in array.nodes
    }
    problems = [
        f"Array {array.name!r}.layers is empty, and a Value with tensor fields and no update is read and written per "
        "layer"
        for array in plain if not array.layers
    ]
    problems += [
        f"Array {array.name!r}.entry.host: nodes {sorted(array.nodes & keyed)} have prefix keys, and a prefix hit "
        "wouldn't restore host fields"
        for array in arrays if array.entry.host and array.nodes & keyed
    ]
    problems += [
        f"Array {array.name!r}: nodes {sorted(array.nodes & keyed)} have prefix keys, and a prefix hit wouldn't "
        "restore its entry"
        for array in plain if array.nodes & keyed
    ]
    return problems


def _sampler_problems(sampler: Sampler, nodes: Sequence[str]) -> list[str]:
    where = f"Sampler {sampler.name!r}"
    problems = [
        f"{where}.nodes: {node!r} is not a node of the model, which has {nodes}"
        for node in sorted(sampler.nodes - set(nodes))
    ]
    if sampler.vocab_size < 1:
        problems.append(f"{where}.vocab_size must be at least 1, got {sampler.vocab_size}")
    if sampler.draws < 1:
        problems.append(f"{where}.draws must be at least 1, got {sampler.draws}")
    return problems


def _read_problems(array: Array, walks: Mapping[str, set[str]], combined: set[str]) -> list[str]:
    problems: list[str] = []
    array_walks = {walk for walk, nodes in walks.items() if nodes & array.nodes}
    reader: dict[tuple[int, str], str] = {}
    clashes: set[tuple[str, str]] = set()
    for key, read in array.reads.items():
        where = f"Array {array.name!r}.reads[{key!r}]"
        for walk in sorted(read.walks or ()):
            if walk in combined:
                problems.append(f"{where}.walks: {walk!r} is a combined walk, whose rows each read by their own walk")
            elif walk not in walks:
                problems.append(f"{where}.walks: {walk!r} is not a walk of the model, which has {sorted(walks)}")
            elif walk not in array_walks:
                problems.append(f"{where}.walks: {walk!r} runs none of the array's nodes {sorted(array.nodes)}")
        for layer, walk in product(sorted(read.layers or array.layers), sorted(read.walks or array_walks)):
            other = reader.setdefault((layer, walk), key)
            if other != key and (other, key) not in clashes:
                clashes.add((other, key))
                problems.append(f"{where}.layers: layer {layer} on walk {walk!r} is also read through reads[{other!r}]")
    return problems


def deployment_problems(
    model: "Model", resources: Mapping[str, Mapping[str, object]], *, sdpa: Collection[str] = frozenset(),
    head_shards: Mapping[str, int] = EMPTY,
) -> list[str]:
    """What a deployment asks of the declarations that they can't do: its ``resources:`` block of a sampler, ``sdpa``
    of the nodes it names, and ``head_shards``, TP times SP per node above SP1, of a read sharded on both sides."""
    declarations = model.declare_state()
    arrays = {declared.name: declared for declared in declarations if isinstance(declared, Array)}
    problems = [
        f"Sampler {declared.name!r}.draws: enable_logprobs, and draws is {declared.draws}; log-probs take one token "
        "per step"
        for declared in declarations
        if isinstance(declared, Sampler) and declared.draws > 1
        and resources.get(declared.name, {}).get("enable_logprobs")
    ]
    reading = {node for array in arrays.values() if array.reads for node in array.nodes}
    problems += [f"node {node!r}: attention_backend sdpa, and it holds no array with a read for SDPA to run"
                 for node in sorted(set(sdpa) - reading)]
    for array in arrays.values():
        gated = any(isinstance(update, GatedDeltaNet) for update in array.updates.values())
        if is_value(array) and not gated and "state_dtype" in resources.get(array.name, {}):
            problems.append(f"Array {array.name!r}: resources.{array.name}.state_dtype, and only a Value with a "
                            "GatedDeltaNet picks a stored dtype")
        if array.max_len and array.nodes & sdpa:
            problems.append(f"Array {array.name!r}: attention_backend sdpa on nodes {sorted(array.nodes & sdpa)}, and "
                            "its entries are kept between steps, so SDPA has no read of them")
        for key, read in array.reads.items():
            shards = max((head_shards.get(node, 1) for node in array.nodes), default=1)
            if array.seq_sharded and arrays.get(read.queries, array).seq_sharded and read.op.q_heads % shards:
                problems.append(f"Array {array.name!r}.reads[{key!r}]: q_heads {read.op.q_heads} doesn't divide by "
                                f"{shards}, TP times SP, which a read sharded on both sides narrows its heads by")
    return problems


def declaration_digest(declarations: Sequence[Array | Sampler]) -> str:
    """A SHA-256 that matches across processes, which Python's `hash` doesn't, since it randomizes strings per process.

    Every field of every declaration goes in, with keys sorted and dtypes and classes by name.
    """
    ordered = sorted(declarations, key=lambda declared: declared.name)
    return hashlib.sha256(json.dumps(_canonical(ordered), sort_keys=True).encode()).hexdigest()


def _canonical(value):
    if is_dataclass(value):
        return {"class": type(value).__name__, **{f.name: _canonical(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Mapping):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, torch.dtype) or get_origin(value) is tuple:
        return str(value)
    if isinstance(value, type):
        return value.__name__
    return value
