"""The value of a step, and the way an edge carries it.

A submodule of tier 1 does no matrix multiplication. It gives one number for
each output name. That number is a checksum.

The checksum covers three groups:

1. the identity of the step: the request, the forward pass, the node, the
   output name, and the number of times that node ran before
2. every input that the step reads, with the name of each input
3. ``extra``: the cache pages that the step reads, when the machine holds a
   cache

The groups decide what tier 1 finds. A step that covers group 1 alone reads
nothing. Its inputs and its pages then hold any value, and every oracle
passes. With groups 2 and 3, each of these faults changes a number:

* a lost edge
* an old buffer
* data that crosses between two requests
* loop iterations in the wrong order
* a page that holds what another step wrote

An edge of mstar carries a list of ``TensorPointerInfo``, one for each tensor
of that name. Tier 1 puts the checksum into the ``uuid`` field. That field
names a tensor in the transport layer. The other fields hold a constant, and
no machine reads them.
"""

from __future__ import annotations

import hashlib

from mstar.graph.base import GraphEdge, TensorPointerInfo

__all__ = [
    "carrier", "checksum", "prefix_checksum", "seed_value", "token_value",
    "values_of",
]

# A cache holds int32, so a token value stays inside that range.
_TOKEN_MASK = 0x3FFF_FFFF

_SEP = "|"


def checksum(
    request_id: str,
    pass_index: int,
    node_name: str,
    output_name: str,
    run_index: int,
    inputs: list[tuple[str, tuple[str, ...]]],
    extra: tuple[str, ...] = (),
) -> str:
    """The value of one output name of one step.

    ``inputs`` is the name of each input, with the values that arrived under
    it. This function sorts them, so the caller gives them in any order.
    ``extra`` is the contents of the cache that the step reads.
    """
    parts = [request_id, str(pass_index), node_name, output_name, str(run_index)]
    for name, values in sorted(inputs):
        parts.append(name)
        parts.extend(values)
    parts.extend(extra)
    digest = hashlib.blake2b(_SEP.join(parts).encode(), digest_size=8)
    return digest.hexdigest()


def seed_value(request_id: str, pass_index: int, name: str) -> str:
    """The value of one edge that starts a forward pass."""
    return checksum(request_id, pass_index, "<seed>", name, 0, [])


def token_value(value: str, index: int, layer: int) -> int:
    """The number that one token of one layer holds.

    ``value`` is the checksum of the step that wrote the token. ``index`` is
    the position of the token inside that step. All three arguments are part
    of the number. Thus a write at the wrong offset, or into the wrong layer,
    changes what a later step reads.
    """
    digest = hashlib.blake2b(
        f"{value}|{index}|{layer}".encode(), digest_size=4,
    ).digest()
    return int.from_bytes(digest, "big") & _TOKEN_MASK


def prefix_checksum(tokens: list[int]) -> str:
    """One number for every token that a step reads from a cache."""
    return hashlib.blake2b(
        "|".join(str(token) for token in tokens).encode(), digest_size=8,
    ).hexdigest()


def carrier(value: str) -> TensorPointerInfo:
    """Wrap one value so that an edge can carry it."""
    return TensorPointerInfo(
        dims=[1],
        dtype="int64",
        nbytes=8,
        address=0,
        stride=[1],
        uuid=value,
        source_session_id="fuzzer",
        source_entity="fuzzer",
    )


def values_of(edge: GraphEdge) -> tuple[str, ...]:
    """The values that one edge carries, in order."""
    return tuple(info.uuid for info in edge.tensor_info)
