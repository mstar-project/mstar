"""The worker's side of the device loop-back: a step whose every routed signal
the node keeps on the device, with the client copy inline, stores and holds
nothing and routes zero tensors per (request, signal)."""
from types import SimpleNamespace

from mstar.communication.tensors import StoredOutputs
from mstar.worker.worker import Worker


class _Engine:
    def __init__(self, on_device):
        self.on_device = on_device
        self.calls = 0

    def device_loopback_signals(self, node, walk):
        self.calls += 1
        return self.on_device.get((node, walk), frozenset())


def _worker():
    w = Worker.__new__(Worker)
    w._device_loopback = {}
    return w


def test_all_on_device_is_per_step_signals_and_cached_per_node_and_walk():
    engine = _Engine({("LLM", "decode"): frozenset({"text_inputs"})})
    w = _worker()
    assert w._all_on_device(engine, "LLM", "decode", ["text_inputs"])
    assert w._all_on_device(engine, "LLM", "decode", ["text_inputs"])
    assert engine.calls == 1, "the node's answer is cached per (node, walk)"
    # a step that also routes a signal the node does not keep stays on tensors
    assert not w._all_on_device(engine, "LLM", "decode", ["text_inputs", "kv"])
    assert not w._all_on_device(engine, "LLM", "prefill_text", ["new_token"])
    assert not w._all_on_device(engine, "LLM", "decode", [])
    assert engine.calls == 2


def test_the_zero_tensor_store_shape_matches_the_route_layout():
    rids, signals = [3, 5, 8], ["text_inputs"]
    stored = StoredOutputs([], [], [], [0] * (len(rids) * len(signals)))
    assert stored.flat_uuids == [] and stored.flat_rids == [] and stored.signal_idxs == []
    assert stored.num_tensors == [0, 0, 0]
    assert SimpleNamespace(**stored._asdict()).num_tensors == [0, 0, 0] if hasattr(stored, "_asdict") else True
