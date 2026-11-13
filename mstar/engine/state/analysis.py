"""What the engine works out from declarations: the checks that need the model, and their digest."""

import hashlib
import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from itertools import product
from typing import TYPE_CHECKING, get_origin

import torch

from mstar.engine.state.contract import (
    EMPTY,
    Array,
    Bidirectional,
    Block,
    Read,
    Sampler,
    SeqShard,
    is_value,
    read_layers,
)

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
    graphs = model.get_graph_walk_graphs()
    walks = {walk: set(graph.get_nodes()) for walk, graph in graphs.items()}
    nodes = sorted(set().union(*walks.values()))
    combined_by_node = model.get_combined_graph_walks()
    combined = {walk for by_name in combined_by_node.values() for walk in by_name}
    for declared in declarations:
        problems += [
            f"{type(declared).__name__} {declared.name!r}.nodes: {node!r} is not a node of the model, which has "
            f"{nodes}"
            for node in sorted(declared.nodes - set(nodes))
        ]
    for array in arrays:
        problems += _read_problems(array, walks, combined)
        problems += _combined_walk_problems(array, combined_walks(array, combined_by_node))
        problems += _array_problems(array)
    problems += _kv_label_problems(model.kv_labels(), arrays)
    problems += _cross_read_problems(arrays)
    problems += _regrouping_problems(model, arrays, graphs)
    problems += _value_problems(arrays, model.kv_labels())
    for declared in declarations:
        if isinstance(declared, Sampler):
            problems += _sampler_problems(declared)
        elif isinstance(declared, SeqShard):
            problems += _shard_problems(declared, arrays, walks, combined)
    return problems


def position_key(array: Array) -> str:
    """The resource key of ``array``'s positions: ``thinker_pos`` for ``thinker_kv``, ``rope`` for any other name."""
    return f"{array.name.removesuffix('_kv')}_pos" if array.name.endswith("_kv") else "rope"


def reads_differ_by_walk(array: Array) -> bool:
    return any(read.walks is not None for read in array.reads.values())


def runs_on_walk(read: Read, walk: str | None) -> bool:
    """Whether ``read`` runs on ``walk``, as a read that names no walk runs on every one."""
    return read.walks is None or walk in read.walks


def combined_walks(array: Array, by_node: Mapping[str, Mapping[str, Collection[str]]]) -> dict[str, Collection[str]]:
    """The combined walks of ``array``'s nodes, with the walks each batches, from each node's in ``by_node``."""
    return {walk: members for node in sorted(array.nodes) for walk, members in by_node.get(node, EMPTY).items()}


def host_field_holders(arrays: Iterable[Array]) -> list[Array]:
    """The arrays of ``arrays`` that hold host fields."""
    return [array for array in arrays if array.entry.host]


def written_by_step(array: Array) -> frozenset[str]:
    """The fields of a Value's entry that its step writes itself, the ones no update op writes in place, and none for
    any other array."""
    if not is_value(array):
        return frozenset()
    return frozenset(array.entry.tensors.keys() - {field for op in array.updates.values() for field in op.writes})


def query_array(array: Array, read: Read, arrays: Mapping[str, Array]) -> Array:
    """The array whose segments lay out the queries of ``array``'s ``read``: the one of ``arrays`` its ``queries``
    names, or else ``array``."""
    return arrays.get(read.queries, array)


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


def _regrouping_problems(model: "Model", arrays: Sequence[Array], graphs: Mapping[str, object]) -> list[str]:
    # A keyed walk whose node reads another tensor than the keyed one is checked per request instead, before the data
    # worker keys its spans (`PreprocessWorkerThread._check_whole_steps`).
    whole = [
        (array, key, read) for array in arrays for key, read in array.reads.items()
        if isinstance(read.mask, Block | Bidirectional)
    ]
    regrouped = [
        *((node, walk, "a chunking policy splits across steps", "leave it out of the policy")
          for node, policies in model.get_chunking_policies().items() for policy in policies if policy.chunkable
          for walk in policy.walks),
        *((node, walk, f"kv_labels()[{array.name!r}][{label!r}] keys for prefix reuse, and a hit can end inside a "
                       "step", "leave it out of the label's keys")
          for array in arrays for label, kv_label in model.kv_labels().get(array.name, {}).items()
          if kv_label.keys is not None for node in array.nodes
          for walk in (
              *(walk for walk in kv_label.keys.walks if _reads_input(graphs, walk, node, kv_label.keys.tensor)),
              *(grown for grown, _ in kv_label.growth if kv_label.keys.growth),
          )),
        *((node, walk, f"{node}'s rebuild policy runs it again with the ids sampled after the prompt",
           "drop the rebuild policy")
          # A rebuild reruns its steps as they ran, and only a walk that reads the ids input takes the ids sampled
          # since.
          for node, policy in model.rebuild_policies().items() if any(node in array.nodes for array, _, _ in whole)
          for walk in policy.ids_walks(node, graphs)),
    ]
    # once per array, which several of its nodes may report
    return list(dict.fromkeys(
        f"Array {array.name!r}.reads[{key!r}]: {type(read.mask).__name__} on walk {walk!r}, which {cause}. "
        f"{type(read.mask).__name__} sees each step's adds as a whole, so {fix}"
        for node, walk, cause, fix in regrouped
        for array, key, read in whole if node in array.nodes and runs_on_walk(read, walk)
    ))


def _reads_input(graphs: Mapping[str, object], walk: str, node: str, tensor: str) -> bool:
    """Whether ``node`` takes ``tensor`` as an input on ``walk``, or ``walk`` is no walk of the model."""
    nodes = graphs[walk].get_nodes() if walk in graphs else {}
    return walk not in graphs or (node in nodes and tensor in nodes[node].input_names)


def _value_problems(arrays: Sequence[Array], kv_labels: Mapping[str, Mapping[str, object]]) -> list[str]:
    """What a Value's entry can't hold yet. A prefix hit restores a Value's snapshot, and a snapshot carries neither its
    host fields nor a field its step writes itself, which no kernel writes mid-prefill."""
    plain = [array for array in arrays if written_by_step(array)]
    holders = host_field_holders(arrays)
    if not plain and not holders:
        return []
    keyed = {
        node for array in arrays if any(label.keys is not None for label in kv_labels.get(array.name, {}).values())
        for node in array.nodes
    }
    problems = [
        f"Array {array.name!r}.layers is empty, and the fields its step writes itself are read and written per layer"
        for array in plain if not array.layers
    ]
    problems += [
        f"Array {array.name!r}.entry.host: nodes {sorted(array.nodes & keyed)} have prefix keys, and a prefix hit "
        "wouldn't restore host fields"
        for array in holders if array.nodes & keyed
    ]
    problems += [
        f"Array {array.name!r}: nodes {sorted(array.nodes & keyed)} have prefix keys, and a prefix hit wouldn't "
        "restore its entry"
        for array in plain if array.nodes & keyed
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
    nothing opens would never split, and nothing would say so. The shard's handle brackets one label."""
    where = f"SeqShard {shard.name!r}"
    problems: list[str] = []
    if len(shard.labels) > 1:
        problems.append(f"{where}.labels: {sorted(shard.labels)} are several, and the handle brackets one")
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
    if len(_query_rows(shard, arrays)) > 1:
        problems.append(f"{where}.labels: reads on nodes {sorted(shard.nodes)} take queries on {sorted(shard.labels)} "
                        "both as rows of one span (packed=False) and as packed tokens or a segmented row, and the "
                        "shard cuts its label's tensors on one axis")
    return problems


def shard_axis(shard: SeqShard, arrays: Sequence[Array]) -> int:
    """The axis ``shard`` cuts its label's tensors on, from how the reads on its nodes take queries on the label: 1
    where they take rows of one span (`Read.packed` False, not `segmented`), every row cut alike; 0 where they take
    packed tokens or a request's segments in one row, cut contiguously. `_shard_problems` refuses a mix."""
    return 1 if _query_rows(shard, arrays) == {True} else 0


def _query_rows(shard: SeqShard, arrays: Sequence[Array]) -> set[bool]:
    """Whether each read on ``shard``'s nodes whose queries open its labels takes them as rows of one span."""
    by_name = {array.name: array for array in arrays}
    return {
        not read.packed and not read.segmented
        for array in arrays if array.nodes & shard.nodes
        for read in array.reads.values() if query_array(array, read, by_name).labels & shard.labels
    }


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


def _combined_walk_problems(array: Array, combined: Mapping[str, Collection[str]]) -> list[str]:
    """A combined walk's rows read by their own walks, and a combined step reads each layer for all its rows, with one
    mask per resource."""
    if not reads_differ_by_walk(array):
        return []
    problems: list[str] = []
    for walk, members in sorted(combined.items()):
        for layer in sorted(array.layers):
            reads = [
                (member, key, read.mask) for member in sorted(members) for key, read in array.reads.items()
                if layer in read_layers(array, read) and runs_on_walk(read, member)
            ]
            clash = next(((a, b) for a in reads for b in reads if a[2] != b[2]), None)
            if clash is not None:
                (one, key, mask), (other, other_key, other_mask) = clash
                problems.append(
                    f"Array {array.name!r}: combined walk {walk!r} runs {one!r} through reads[{key!r}] "
                    f"({type(mask).__name__}) and {other!r} through reads[{other_key!r}] ({type(other_mask).__name__}) "
                    f"on layer {layer}; a combined step plans one mask per resource"
                )
                break
            readers = {member for member, _, _ in reads}
            if readers and readers != set(members):
                problems.append(
                    f"Array {array.name!r}: combined walk {walk!r} runs {min(readers)!r}, which reads layer {layer}, "
                    f"and {min(set(members) - readers)!r}, which doesn't; a combined step reads each layer for all its "
                    "rows"
                )
                break
    return problems


def deployment_problems(model: "Model", resources: Mapping[str, Mapping[str, object]]) -> list[str]:
    """What a deployment's ``resources:`` block asks of the declarations that they can't do: of a sampler, and of an
    array whose step writes its state itself."""
    declarations = model.declare_state()
    problems = [
        f"Sampler {declared.name!r}.draws: enable_logprobs, and draws is {declared.draws}; log-probs take one token "
        "per step"
        for declared in declarations
        if isinstance(declared, Sampler) and declared.draws > 1
        and resources.get(declared.name, {}).get("enable_logprobs")
    ]
    for array in declarations:
        if not isinstance(array, Array):
            continue
        # a pool with no state block refuses state_dtype itself
        if "state_dtype" in resources.get(array.name, {}) and "state" in written_by_step(array):
            problems.append(f"Array {array.name!r}: resources.{array.name}.state_dtype, and its step writes state "
                            "itself, where only a field an update op writes takes a stored dtype")
    return problems


def split_read_problems(
    declarations: Sequence[Array | Sampler | SeqShard], sdpa_reads: Collection[str],
    split: Mapping[str, Collection[str]],
) -> list[str]:
    """What a read within a step asks of a node on SP ranks that no exchange serves, where ``split`` gives, per such
    node, the labels its sequence shard lists: SDPA (``sdpa_reads``, the reads a ``backend: sdpa`` block binds) over
    split keys, which runs outside the engine's resources and so would attend one rank's share, and a cross read whose
    queries split as well as its keys. The engine refuses these at load and ``validate_deployment`` reports them, both
    from the declarations, so neither waits for a step that splits."""
    arrays = {declared.name: declared for declared in declarations if isinstance(declared, Array)}
    problems = []
    for array in arrays.values():
        if array.max_len:
            continue
        nodes = [node for node in sorted(array.nodes) if array.labels & set(split.get(node, ()))]
        if not nodes:
            continue
        for key, read in array.reads.items():
            queries = query_array(array, read, arrays)
            where = f"Array {array.name!r}.reads[{key!r}]"
            if key in sdpa_reads:
                problems.append(f"{where}: backend sdpa over labels {sorted(array.labels)}, which the sequence shard "
                                f"of {nodes} splits over SP ranks; SDPA would attend one rank's share: run it on the "
                                "ragged resource (drop `backend: sdpa`)")
                continue
            both = [node for node in nodes if queries is not array and queries.labels & set(split[node])]
            if both:
                problems.append(f"{where}: its queries (Array {queries.name!r}, labels {sorted(queries.labels)}) and "
                                f"keys (labels {sorted(array.labels)}) both split over the SP ranks of {both}, which "
                                "no exchange serves: list only one side in the sequence shard")
    return problems


def declaration_digests(
    declarations: Sequence[Array | Sampler | SeqShard], kernel_settings: Mapping[str, Mapping[str, object]] = EMPTY,
) -> dict[str, str]:
    """Each declaration's SHA-256, by name, and with ``kernel_settings`` that of the hook's settings, under
    ``kernel_settings()``, since a pool size can follow the deployment too. A SHA-256 matches across processes, as
    Python's `hash` doesn't, since it randomizes strings per process.

    Every field goes in, with keys sorted and dtypes and classes by name.
    """
    digests = {declared.name: _sha256(declared) for declared in declarations}
    if kernel_settings:
        digests["kernel_settings()"] = _sha256(kernel_settings)
    return digests


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True).encode()).hexdigest()


def digest_mismatch(ours: Mapping[str, str], theirs: Mapping[str, str], *, here: str, there: str) -> str | None:
    """None when the two processes built the same declarations, or which ones differ."""
    differ = [
        name if name in ours and name in theirs else f"{name} (only on the {here if name in ours else there})"
        for name in sorted(ours.keys() | theirs.keys()) if ours.get(name) != theirs.get(name)
    ]
    if not differ:
        return None
    return (f"the {here} and the {there} declare different state: {', '.join(differ)}. A declaration that follows the "
            "deployment reads it in get_worker_graphs, which every process calls before declare_state")


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
