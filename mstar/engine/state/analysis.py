"""What the engine works out from declarations: the checks that need the model, and their digest."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from itertools import product
from typing import TYPE_CHECKING, get_origin

import torch

from mstar.engine.state.contract import Array, Block, Sampler, SeqShard

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
    problems = _name_problems(declarations)
    walks = {walk: set(graph.get_nodes()) for walk, graph in model.get_graph_walk_graphs().items()}
    nodes = sorted(set().union(*walks.values()))
    combined = {walk for by_name in model.get_combined_graph_walks().values() for walk in by_name}
    for declared in declarations:
        problems += [
            f"{type(declared).__name__} {declared.name!r}.nodes: {node!r} is not a node of the model, which has "
            f"{nodes}"
            for node in sorted(declared.nodes - set(nodes))
        ]
    for array in arrays:
        problems += _read_problems(array, walks, combined)
        problems += _array_problems(array)
    problems += _kv_label_problems(model.kv_labels(), arrays)
    for declared in declarations:
        if isinstance(declared, Sampler):
            problems += _sampler_problems(declared)
        elif isinstance(declared, SeqShard):
            problems += _shard_problems(declared, arrays, walks, combined)
    return problems


def position_key(array: Array) -> str:
    """The resource key of ``array``'s positions: ``thinker_pos`` for ``thinker_kv``, ``rope`` for any other name."""
    return f"{array.name.removesuffix('_kv')}_pos" if array.name.endswith("_kv") else "rope"


def _name_problems(declarations: Sequence[Array | Sampler | SeqShard]) -> list[str]:
    """Names share one namespace while they are resource keys, so a read can't reuse an array's name."""
    problems: list[str] = []
    declared: dict[str, str] = {}
    for field, name in (
        *(
            pair for array in declarations if isinstance(array, Array) for pair in (
                (f"Array {array.name!r}.name", array.name),
                *((f"Array {array.name!r}.reads", key) for key in array.reads),
                *((f"Array {array.name!r}.updates", key) for key in array.updates),
                *(() if array.positions is None else ((f"Array {array.name!r}.positions", position_key(array)),)),
            )
        ),
        *(
            (f"{type(other).__name__} {other.name!r}.name", other.name)
            for other in declarations if not isinstance(other, Array)
        ),
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


def _sampler_problems(sampler: Sampler) -> list[str]:
    where = f"Sampler {sampler.name!r}"
    problems = []
    if sampler.vocab_size < 1:
        problems.append(f"{where}.vocab_size must be at least 1, got {sampler.vocab_size}")
    if sampler.draws < 1:
        problems.append(f"{where}.draws must be at least 1, got {sampler.draws}")
    return problems


def _shard_problems(
    shard: SeqShard, arrays: Sequence[Array], walks: Mapping[str, set[str]], combined: set[str],
) -> list[str]:
    """``walks`` are rows' walks, as a read's are, and ``labels`` those an array on the shard's nodes opens: a label
    nothing opens would never split, and nothing would say so."""
    where = f"SeqShard {shard.name!r}"
    problems: list[str] = []
    for walk in sorted(shard.walks or ()):
        if walk in combined:
            problems.append(f"{where}.walks: {walk!r} is a combined walk; list the walks it combines, and it splits "
                            "where one of them does")
        elif walk not in walks:
            problems.append(f"{where}.walks: {walk!r} is not a walk of the model, which has {sorted(walks)}")
        elif not walks[walk] & shard.nodes:
            problems.append(f"{where}.walks: {walk!r} runs none of the shard's nodes {sorted(shard.nodes)}")
    opened = {label for array in arrays if array.nodes & shard.nodes for label in array.labels}
    if shard.labels - opened:
        problems.append(f"{where}.labels: {sorted(shard.labels - opened)} are opened by no array on nodes "
                        f"{sorted(shard.nodes)}, which open {sorted(opened)}")
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


def deployment_problems(model: "Model", resources: Mapping[str, Mapping[str, object]]) -> list[str]:
    """What a deployment's ``resources:`` block asks of a declared sampler that it can't do."""
    return [
        f"Sampler {declared.name!r}.draws: enable_logprobs, and draws is {declared.draws}; log-probs take one token "
        "per step"
        for declared in model.declare_state()
        if isinstance(declared, Sampler) and declared.draws > 1
        and resources.get(declared.name, {}).get("enable_logprobs")
    ]


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
