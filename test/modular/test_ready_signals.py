"""A node's streaming readiness: every input still MISSING is a streaming one.

``input_names.issuperset(ready_names | streaming_inputs)`` expressed this as a
tautology -- ready_names is asserted a subset of input_names and
streaming_inputs is one by construction -- so every node with any input at
all read as streaming-ready.
"""
from mstar.graph.base import GraphEdge, ReadySignals


def test_a_node_missing_a_non_streaming_input_is_not_streaming_ready():
    """``ar_decode`` takes token and kv_cache, neither streaming, so one
    arriving must NOT make it streaming-ready."""
    sig = ReadySignals(
        node_name="ar_decode",
        input_names={"token", "kv_cache"},
        streaming_inputs=set(),
    )
    sig.update(GraphEdge(name="token", next_node="ar_decode"))
    assert not sig.is_ready
    assert not sig.is_ready_for_streaming, "a tautology made this always true"

    sig.update(GraphEdge(name="kv_cache", next_node="ar_decode"))
    assert sig.is_ready and sig.is_ready_for_streaming


def test_a_node_missing_only_a_streaming_input_is_streaming_ready():
    """The case the flag exists for: run on what has arrived."""
    sig = ReadySignals(
        node_name="snac",
        input_names={"text", "new_token"},
        streaming_inputs={"new_token"},
    )
    sig.update(GraphEdge(name="text", next_node="snac"))
    assert not sig.is_ready
    assert sig.is_ready_for_streaming


def test_removing_an_input_takes_streaming_readiness_back_down():
    """``remove`` is the speculation-rollback path, so the flag must fall."""
    sig = ReadySignals(
        node_name="snac",
        input_names={"text", "new_token"},
        streaming_inputs={"new_token"},
    )
    sig.update(GraphEdge(name="text", next_node="snac"))
    assert sig.is_ready_for_streaming
    sig.remove("text")
    assert not sig.is_ready_for_streaming
