"""What the engine works out from declarations: the checks that need the model, and their digest."""

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import fields, is_dataclass
from itertools import product
from typing import TYPE_CHECKING, get_origin

import torch

from mstar.engine.state.contract import EMPTY, Array, Block, Read, Sampler, read_layers

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
    problems += _host_problems(arrays, model.kv_labels())
    for sampler in samplers:
        problems += _sampler_problems(sampler, nodes)
    return problems


def position_key(array: Array) -> str:
    """The resource key of ``array``'s positions: ``thinker_pos`` for ``thinker_kv``, ``rope`` for any other name."""
    return f"{array.name.removesuffix('_kv')}_pos" if array.name.endswith("_kv") else "rope"


def query_array(array: Array, read: Read, arrays: Mapping[str, Array]) -> Array:
    """The array whose segments lay out the queries of ``array``'s ``read``: the one of ``arrays`` its ``queries``
    names, or else ``array``."""
    return arrays.get(read.queries, array)


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
            elif queried.max_len:
                problems.append(f"{where}: {read.queries!r} keeps entries, and queries are laid out by the segments "
                                "of an array with max_len 0")
    return problems


def _host_problems(arrays: Sequence[Array], kv_labels: Mapping[str, Mapping[str, object]]) -> list[str]:
    """A prefix hit restores a Value's snapshot, and a snapshot doesn't carry host fields yet."""
    holders = [array for array in arrays if array.entry.host]
    if not holders:
        return []
    keyed = {
        node for array in arrays if any(label.keys is not None for label in kv_labels.get(array.name, {}).values())
        for node in array.nodes
    }
    return [
        f"Array {array.name!r}.entry.host: nodes {sorted(array.nodes & keyed)} have prefix keys, and a prefix hit "
        "wouldn't restore host fields"
        for array in holders if array.nodes & keyed
    ]


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
        for layer, walk in product(sorted(read_layers(array, read)), sorted(read.walks or array_walks)):
            other = reader.setdefault((layer, walk), key)
            if other != key and (other, key) not in clashes:
                clashes.add((other, key))
                problems.append(f"{where}.layers: layer {layer} on walk {walk!r} is also read through reads[{other!r}]")
    return problems


def deployment_problems(
    model: "Model", resources: Mapping[str, Mapping[str, object]], *, sdpa_reads: Collection[str] = (),
    split: Mapping[str, Collection[str]] = EMPTY,
) -> list[str]:
    """What a deployment asks of the declarations that they can't do: its ``resources:`` block of a sampler, and SDPA
    (``sdpa_reads``, the reads its ``backend: sdpa`` blocks bind) over keys the sequence shard of a node on SP ranks
    splits (``split``: per such node, the labels its shard lists), which would attend one rank's share."""
    declarations = model.declare_state()
    problems = [
        f"Sampler {declared.name!r}.draws: enable_logprobs, and draws is {declared.draws}; log-probs take one token "
        "per step"
        for declared in declarations
        if isinstance(declared, Sampler) and declared.draws > 1
        and resources.get(declared.name, {}).get("enable_logprobs")
    ]
    for array in declarations:
        if not isinstance(array, Array) or array.max_len:
            continue
        for key in array.reads.keys() & set(sdpa_reads):
            nodes = sorted(node for node in array.nodes if array.labels & set(split.get(node, ())))
            if nodes:
                problems.append(f"Array {array.name!r}.reads[{key!r}]: backend sdpa over labels "
                                f"{sorted(array.labels)}, which the sequence shard of {nodes} splits over SP ranks; "
                                "SDPA would attend one rank's share")
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
