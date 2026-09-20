"""The spec path must sync loop-iter counts before the GPU thread reads them.

The spec batch carries ``get_fwd_info``'s counts, which lag the graph io; only
the non-spec path resynced. N's routing advances the count after the spec
submit, so a new-iter speculation has to add the step itself.
"""

from types import SimpleNamespace

from mstar.worker.worker import Worker


def _spec(is_new_iter, loop_name, stale_counts):
    req_info = SimpleNamespace(dynamic_loop_iter_counts=dict(stale_counts))
    spec = SimpleNamespace(
        node_batch=SimpleNamespace(per_request_info={"X": req_info}),
        partition="p0",
        is_new_iter=is_new_iter,
        loop_name=loop_name,
    )
    return req_info, spec


def _worker(live_counts):
    w = Worker.__new__(Worker)
    w.worker_graphs_manager = SimpleNamespace(
        get_dynamic_loop_iters=lambda rid, partition: dict(live_counts),
    )
    return w


def test_new_iter_speculation_advances_the_stepped_loop():
    # io still reads 5 (routing deferred past the submit); the speculated step
    # is 6, not the stale 4 the batch carried.
    w = _worker({"decode_loop": 5})
    req_info, spec = _spec(True, "decode_loop", {"decode_loop": 4})
    w._sync_spec_loop_iters(spec)
    assert req_info.dynamic_loop_iter_counts == {"decode_loop": 6}


def test_same_iter_speculation_syncs_without_advancing():
    # No loop boundary crossed: correct the stale value to the live index, no +1.
    w = _worker({"decode_loop": 5})
    req_info, spec = _spec(False, None, {"decode_loop": 99})
    w._sync_spec_loop_iters(spec)
    assert req_info.dynamic_loop_iter_counts == {"decode_loop": 5}
