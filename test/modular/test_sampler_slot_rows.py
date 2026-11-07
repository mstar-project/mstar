"""The sampler's per-request rows by slot: registration, growth, and the RNG
offset a captured sample gathers and scatters back in the graph."""
import threading

import numpy as np
import pytest
import torch

from mstar.engine.resources.sampler.utils import HostBuffer, SamplerBuffers, SamplingConfig


def _buffers(max_bs=4, cg_slots=2):
    return SamplerBuffers.allocate(
        max_batch_size=max_bs, device=torch.device("cpu"), cg_slots=cg_slots,
    )


def test_plan_thread_row_init_survives_a_concurrent_host_regrow(monkeypatch):
    """A plan-thread ``_init_slot`` between the regrow's copy and swap must still land."""
    bufs = _buffers(max_bs=2, cg_slots=1)
    copied, written = threading.Event(), threading.Event()
    paused = []

    def grow_master(self, new_capacity):
        new = np.full(new_capacity, self.default, dtype=self.master.dtype)
        new[: self.master.shape[0]] = self.master
        if not paused:
            paused.append(self)
            copied.set()
            written.wait(timeout=0.3)  # times out once the write waits on the lock
        self.master = new

    monkeypatch.setattr(HostBuffer, "grow_master", grow_master)
    bufs.register_request("a", SamplingConfig(temperature=0.0))
    bufs.register_request("b")

    def plan():
        copied.wait()
        bufs.gather_static(["a"], 1, 0)  # inits a's row on the plan thread
        written.set()

    t = threading.Thread(target=plan)
    t.start()
    bufs.register_request("c")  # grows the host masters on the main thread
    t.join()
    assert bufs.temperature.master[bufs.rows._rid_to_slot["a"]] == 0.0


def _ingraph(max_bs=4, cg_slots=2, slots=8):
    return SamplerBuffers.allocate(
        max_batch_size=max_bs, device=torch.device("cpu"), cg_slots=cg_slots,
        ingraph_scatter=True, slots=slots,
    )


def test_ingraph_scatter_rows_land_by_slot_and_padding_hits_the_trash_row():
    """What the captured sample does with the knob on, step by step on the
    CPU views: gather the offsets off the master, advance, scatter them back;
    padding rows read and write the trash row only."""
    bufs = _ingraph()
    assert bufs.ingraph_scatter and bufs._trash_slot == 0
    assert bufs.rows.offset.shape[0] == 9
    for rid in ("a", "b", "c"):
        bufs.register_request(rid)
    bufs.gather_static(["b", "a"], 4, 1)
    bufs.gather_dynamic(["b", "a"], 4, 1)  # uploads the index row, no offset gather
    # (a fresh slot's offset reset ran there; an advanced offset survives)
    bufs.rows.offset[bufs.rows._rid_to_slot["a"]] = 10
    assert bufs._slot_idx_gpu[1, :4].tolist()[2:] == [0, 0]
    s = bufs.sampler_for(4, 1)
    assert s.ingraph_scatter and s.slot_idx_view.tolist()[:2] == [bufs.rows._rid_to_slot[r] for r in ("b", "a")]
    s.gather_in_graph()
    assert s.offset_buf.tolist()[1] == 10
    s.offset_buf += 1
    s.scatter_in_graph()
    assert bufs.rows.offset[bufs.rows._rid_to_slot["a"]].item() == 11


def test_ingraph_scatter_graph_masters_never_grow():
    """The offset master a captured graph indexes is allocated in full and
    kept; the others start small and grow up to it, and a registration past
    it is refused."""
    bufs = SamplerBuffers.allocate(
        max_batch_size=2, device=torch.device("cpu"), vocab_size=16,
        ingraph_scatter=True, slots=4,
    )
    offset = bufs.rows.offset
    assert offset.shape[0] == 5
    assert bufs.temperature.master.shape[0] == 3
    masks = bufs.rows.seen_words
    assert masks.shape[0] == 5
    assert bufs.rows.free_rows == 4
    for rid in ("a", "b", "c", "d"):
        bufs.register_request(rid)
    assert bufs.rows.free_rows == 0
    bufs.rows.sync()  # a gpu-thread access grows the device rows
    assert bufs.temperature.master.shape[0] == 5 and bufs.rows.seen_words is masks
    assert bufs.rows.offset is offset
    with pytest.raises(ValueError, match="MSTAR_SAMPLER_SLOTS"):
        bufs.register_request("e")
    # the default path still grows
    plain = SamplerBuffers.allocate(max_batch_size=2, device=torch.device("cpu"))
    for rid in ("a", "b", "c"):
        plain.register_request(rid)
    plain.rows.sync()  # a gpu-thread access grows the device rows
    assert plain.rows.offset.shape[0] >= 3 and not plain.ingraph_scatter


def test_capture_padding_rows_take_no_slot():
    bufs = _buffers()
    free = len(bufs.rows._free_slots)
    for rid in (-2, -3, -4):
        bufs.register_request(rid)
    assert len(bufs.rows._free_slots) == free and not bufs.rows.has_slot(-2)
    # padding rows read slot 0
    bufs.register_request("a")
    assert bufs.rows.indices([-2, "a"]).tolist() == [0, bufs.rows._rid_to_slot["a"]]


def test_a_dropped_row_keeps_its_offset_in_graph():
    """``SamplerStep.kept_rids``: a dropped row still draws at its own offset,
    but the captured scatter sends its offset to the trash row."""
    bufs = _ingraph()
    for rid in ("a", "b"):
        bufs.register_request(rid)
    slot_a, slot_b = bufs.rows._rid_to_slot["a"], bufs.rows._rid_to_slot["b"]
    bufs.gather_static(["a", "b"], 4, 1)
    bufs.gather_dynamic(["a", "b"], 4, 1, kept=frozenset({"a"}))
    bufs.rows.offset[slot_a], bufs.rows.offset[slot_b] = 10, 20
    s = bufs.sampler_for(4, 1)
    assert s.slot_idx_view.tolist() == [slot_a, slot_b, 0, 0]
    assert s.scatter_idx_view.tolist() == [slot_a, 0, 0, 0]
    s.gather_in_graph()
    assert s.offset_buf.tolist()[:2] == [10, 20]
    s.offset_buf += 1
    s.scatter_in_graph()
    assert bufs.rows.offset[[slot_a, slot_b]].tolist() == [11, 20]
    # the next step keeps every row, and stages that into the same views
    bufs.gather_dynamic(["a", "b"], 4, 1)
    assert s.scatter_idx_view.tolist() == [slot_a, slot_b, 0, 0]
