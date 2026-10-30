"""The sampler's last-token master: the device-side loop-back's store. The
captured sample writes a per-step row, commit scatters it to the slot master,
the next step gathers its input ids by slot; an eager step writes and reads
the master by request."""
import threading

import numpy as np
import pytest
import torch

from mstar.engine.resources.sampler.utils import HostBuffer, SamplerBuffers, SamplingConfig
from mstar.utils.h2d import PinnedStager


def _buffers(max_bs=4, cg_slots=2):
    return SamplerBuffers.allocate(
        max_batch_size=max_bs, device=torch.device("cpu"), cg_slots=cg_slots,
    )


def test_eager_write_and_read_go_by_request():
    bufs = _buffers()
    for rid in ("a", "b", "c"):
        bufs.register_request(rid)
    bufs.write_last_tokens(["a", "b"], torch.tensor([11, 12]))
    assert bufs.last_tokens_for(["b", "a"]).tolist() == [12, 11]
    bufs.write_last_tokens(["c"], torch.tensor([13], dtype=torch.int32))
    assert bufs.last_tokens_for(["c", "a"]).tolist() == [13, 11]
    assert bufs.has_slot("a") and not bufs.has_slot("zz")


def test_a_routed_write_takes_its_slot_index_through_the_stager():
    bufs = _buffers()
    for rid in ("a", "b", "c"):
        bufs.register_request(rid)
    stager = PinnedStager(torch.long, numel=4)
    calls = []
    to_device = stager.to_device
    stager.to_device = lambda values, device: calls.append(list(values)) or to_device(values, device)
    bufs.write_last_tokens(["c", "a"], torch.tensor([31, 11]), stager)
    assert calls == [[bufs._rid_to_slot["c"], bufs._rid_to_slot["a"]]]
    assert bufs.last_tokens_for(["a", "c"]).tolist() == [11, 31]


def test_the_captured_row_scatters_to_the_slot_master_and_gathers_back():
    bufs = _buffers(max_bs=4, cg_slots=2)
    for rid in ("a", "b", "c"):
        bufs.register_request(rid)
    bufs.write_last_tokens(["a", "b", "c"], torch.tensor([1, 2, 3]))
    # a step of [b, a] padded to 4 on cg slot 1: static then dynamic gather
    bufs.gather_static(["b", "a"], 4, 1)
    bufs.gather_dynamic(["b", "a"], 4, 1)
    # what the captured sample does: write the step's tokens into the slot row
    bufs.slice_for_bs(4, 1)["last_token_buf"].copy_(torch.tensor([22, 21, 99, 99]))
    bufs.scatter_last_token(1)
    # real rows reached their masters, padding rows did not touch slot 0's
    assert bufs.last_tokens_for(["a", "b", "c"]).tolist() == [21, 22, 3]
    # the next step's input ids, in the slot row's order; padding rows read
    # slot 0 (whatever request holds it, or the default when it is free)
    slot0_tok = bufs.last_token.master[0].item()
    assert bufs.gather_last_tokens(1, 4).tolist() == [22, 21, slot0_tok, slot0_tok]
    # the other cg slot is untouched by this step
    bufs.gather_static(["c"], 4, 0)
    bufs.gather_dynamic(["c"], 4, 0)
    assert bufs.gather_last_tokens(0, 4).tolist()[0] == 3


def test_growth_and_slot_reuse_keep_the_tokens():
    bufs = _buffers(max_bs=2, cg_slots=1)
    bufs.register_request("a")
    bufs.register_request("b")
    bufs.write_last_tokens(["a", "b"], torch.tensor([5, 6]))
    bufs.register_request("c")  # beyond the capacity of 2: the masters grow
    bufs.write_last_tokens(["c"], torch.tensor([7]))
    assert bufs.last_tokens_for(["a", "b", "c"]).tolist() == [5, 6, 7]
    bufs.unregister_request("a")
    bufs.register_request("d")  # reuses a's slot; its first step writes it
    bufs.write_last_tokens(["d"], torch.tensor([8]))
    assert bufs.last_tokens_for(["d", "b", "c"]).tolist() == [8, 6, 7]


def test_registration_growth_leaves_the_device_masters_to_the_gpu_thread():
    """A scatter that read the master before a registration grew it must still land."""
    bufs = _buffers(max_bs=2, cg_slots=1)
    bufs.register_request("a")
    bufs.register_request("b")
    bufs.gather_static(["a", "b"], 2, 0)
    bufs.gather_dynamic(["a", "b"], 2, 0)
    bufs.slice_for_bs(2, 0)["last_token_buf"].copy_(torch.tensor([5, 6]))
    # commit's scatter, split where the GIL can switch: it reads the master...
    master = bufs.last_token.master
    idx, real_bs = bufs._scatter_idx(0)
    bufs.register_request("c")  # ...the main thread grows past the capacity of 2...
    assert bufs.last_token.master is master
    master.index_copy_(0, idx, bufs.last_token.slot_view(0, real_bs))  # ...then writes
    # the next step grows the device rows on the gpu thread, keeping the tokens
    bufs.gather_static(["a", "b"], 2, 0)
    bufs.gather_dynamic(["a", "b"], 2, 0)
    assert bufs.last_token.master.shape[0] >= 3 and bufs.offset.master.shape[0] >= 3
    assert bufs.gather_last_tokens(0, 2).tolist() == [5, 6]


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
    assert bufs.temperature.master[bufs._rid_to_slot["a"]] == 0.0


def _ingraph(max_bs=4, cg_slots=2, slots=8):
    return SamplerBuffers.allocate(
        max_batch_size=max_bs, device=torch.device("cpu"), cg_slots=cg_slots,
        ingraph_scatter=True, slots=slots,
    )


def test_ingraph_scatter_rows_land_by_slot_and_padding_hits_the_trash_row():
    """What the captured sample does with the knob on, step by step on the
    CPU views: gather the offsets off the master, advance, scatter offsets
    and tokens back; padding rows read and write the trash row only."""
    bufs = _ingraph()
    assert bufs.ingraph_scatter and bufs._trash_slot == 0
    assert bufs.last_token.master.shape[0] == 9 and bufs.offset.master.shape[0] == 9
    for rid in ("a", "b", "c"):
        bufs.register_request(rid)
    bufs.write_last_tokens(["a", "b", "c"], torch.tensor([1, 2, 3]))
    bufs.gather_static(["b", "a"], 4, 1)
    bufs.gather_dynamic(["b", "a"], 4, 1)  # uploads the index row, no offset gather
    # (a fresh slot's offset reset ran there; an advanced offset survives)
    bufs.offset.master[bufs._rid_to_slot["a"]] = 10
    assert bufs._slot_idx_gpu[1, :4].tolist()[2:] == [0, 0]
    s = bufs.sampler_for(4, 1)
    assert s.ingraph_scatter and s.slot_idx_view.tolist()[:2] == [bufs._rid_to_slot[r] for r in ("b", "a")]
    s.gather_in_graph()
    assert s.offset_buf.tolist()[1] == 10
    s.offset_buf += 1
    s.scatter_in_graph(torch.tensor([22, 21, 99, 98]))
    assert bufs.last_tokens_for(["a", "b", "c"]).tolist() == [21, 22, 3]
    assert bufs.offset.master[bufs._rid_to_slot["a"]].item() == 11
    assert bufs.last_token.master[0].item() in (99, 98)
    # the next step's input ids: real rows by slot, padding rows the trash row
    got = bufs.gather_last_tokens(1, 4).tolist()
    assert got[:2] == [22, 21] and got[2] == got[3] == bufs.last_token.master[0].item()


def test_ingraph_scatter_masters_never_grow():
    bufs = _ingraph(max_bs=2, slots=2)
    bufs.register_request("a")
    bufs.register_request("b")
    with pytest.raises(RuntimeError, match="MSTAR_SAMPLER_SLOTS"):
        bufs.register_request("c")
    # the default path still grows
    plain = SamplerBuffers.allocate(max_batch_size=2, device=torch.device("cpu"))
    for rid in ("a", "b", "c"):
        plain.register_request(rid)
    plain.write_last_tokens(["c"], torch.tensor([7]))  # a gpu-thread access grows the device rows
    assert plain.last_token.master.shape[0] >= 3 and not plain.ingraph_scatter


def test_capture_padding_rows_take_no_slot():
    bufs = _buffers()
    free = len(bufs._free_slots)
    for rid in (-2, -3, -4):
        bufs.register_request(rid)
    assert len(bufs._free_slots) == free and not bufs.has_slot(-2)
    bufs.register_request("a")
    bufs.write_last_tokens(["a"], torch.tensor([5]))
    # padding rows read slot 0
    assert bufs.last_tokens_for([-2, "a"]).tolist() == [
        bufs.last_token.master[0].item(), 5,
    ]
