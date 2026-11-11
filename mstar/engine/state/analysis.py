"""What the engine works out from declarations: the checks that need the model, and their digest."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from itertools import product
from typing import TYPE_CHECKING, get_origin

import torch

from mstar.engine.state.contract import Array

if TYPE_CHECKING:
    from mstar.model.base import Model


def state_declaration_problems(model: "Model") -> list[str]:
    """Every problem with ``model``'s declarations that its nodes and walks show.

    A model that declares nothing returns before its walk graphs are built, since building them can read checkpoints.
    """
    arrays = model.declare_state()
    if not arrays:
        return []
    problems = _name_problems(arrays)
    walks = {walk: set(graph.get_nodes()) for walk, graph in model.get_graph_walk_graphs().items()}
    nodes = sorted(set().union(*walks.values()))
    combined = {walk for by_name in model.get_combined_graph_walks().values() for walk in by_name}
    for array in arrays:
        problems += [
            f"Array {array.name!r}.nodes: {node!r} is not a node of the model, which has {nodes}"
            for node in sorted(array.nodes - set(nodes))
        ]
        problems += _read_problems(array, walks, combined)
    return problems


def _name_problems(arrays: Sequence[Array]) -> list[str]:
    """Names share one namespace while they are resource keys, so a read can't reuse an array's name."""
    problems: list[str] = []
    declared: set[str] = set()
    for array in arrays:
        where = f"Array {array.name!r}"
        for field, name in (
            (f"{where}.name", array.name),
            *((f"{where}.reads", key) for key in array.reads),
            *((f"{where}.updates", key) for key in array.updates),
        ):
            if name in declared:
                problems.append(f"{field}: the name {name!r} is declared twice")
            declared.add(name)
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


def declaration_digest(arrays: Sequence[Array]) -> str:
    """A SHA-256 that matches across processes, which Python's `hash` doesn't, since it randomizes strings per process.

    Every field of every declaration goes in, with keys sorted and dtypes and classes by name.
    """
    ordered = sorted(arrays, key=lambda array: array.name)
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
