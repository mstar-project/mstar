"""New-token counting in ``Worker._send_outputs`` happens on the first TP rank only: the
conductor takes the counts from the rank-0 WORKER_GRAPHS_DONE, and follower ranks never hold
the emitted tensors the new-token edges point at (only the leader emits to the client), so a
follower looking them up raised KeyError on every decode step of a TP>1 model."""
import types

import torch

from mstar.graph.base import GraphEdge
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.worker.node_manager_utils import NodeOutputRouting
from mstar.worker.worker import Worker


def _fake_worker(store: dict):
    counts: dict = {}
    sent: list = []
    wgm = types.SimpleNamespace(
        get_graph_walk=lambda rid, part: "decode",
        get_fwd_info=lambda rid, part: None,
        buffer_persist_signals=lambda rid, p: None,
        buffer_new_token_counts=lambda rid, c: counts.update(c),
        buffer_output_signals=lambda rid, e: None,
        register_output_loop_indices=lambda **kw: None,
        per_request_info={"r": types.SimpleNamespace(stream_buffers={}, per_partition_info={})},
    )
    tm = types.SimpleNamespace(get_tensor=lambda request_id, uuid: store[request_id][uuid])
    comm = types.SimpleNamespace(send=lambda dst, msg: sent.append((dst, msg)))
    return types.SimpleNamespace(worker_graphs_manager=wgm, tensor_manager=tm, communicator=comm), counts, sent


def _routing(first_tp_rank: bool) -> NodeOutputRouting:
    edge = GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", conductor_new_token=True)
    edge.tensor_info = [types.SimpleNamespace(uuid="u1")]
    return NodeOutputRouting(
        routed_to_this_worker_graph=[], is_first_tp_rank=first_tp_rank, persist=[], to_workers={},
        new_token_outputs=[edge],
    )


def test_follower_rank_skips_new_token_counting():
    # the follower's store has no tensor for the emitted edge: must neither raise nor count
    w, counts, sent = _fake_worker({"r": {}})
    Worker._send_outputs(w, "r", _routing(False), nested_loop_indices=None, graph_walk="decode", partition_name="p")
    assert counts == {} and sent == []


def test_first_rank_counts_new_tokens():
    w, counts, _ = _fake_worker({"r": {"u1": torch.zeros(3, dtype=torch.int64)}})
    Worker._send_outputs(w, "r", _routing(True), nested_loop_indices=None, graph_walk="decode", partition_name="p")
    assert counts == {"new_token": 3}


def test_symm_all_reduce_mode_parsing():
    from mstar.distributed.communication import _flashinfer_mode, _lamport_mode, _symm_mode

    assert _symm_mode("auto") == (True, True) and _symm_mode("multimem") == (True, True)
    assert _symm_mode("lamport") == (True, True) and _symm_mode("flashinfer") == (True, True)
    assert _symm_mode("1") == (True, False)
    assert _symm_mode("0") == (False, False) and _symm_mode("nccl") == (False, False)
    # the Lamport one-shot tier is the default for small 16-bit messages; multimem/1/0 opt out
    assert _lamport_mode("auto") and _lamport_mode(" Lamport ") and _lamport_mode("flashinfer")
    assert not _lamport_mode("multimem") and not _lamport_mode("1") and not _lamport_mode("0")
    # its all-reduces run on flashinfer's kernel by default; `lamport` pins M*'s own
    assert _flashinfer_mode("auto") and _flashinfer_mode("flashinfer") and not _flashinfer_mode("lamport")


def test_lamport_tier_selection(monkeypatch):
    """Which shapes the Lamport tier takes, and that the buffer path (``symm_applies``) steps aside
    for them so producers hand plain tensors to ``all_reduce``."""
    from mstar.distributed.communication import CommGroup

    cg = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    monkeypatch.setattr(cg, "_symm_available", lambda: True)
    cg._lamport_enabled = True
    cg._lamport_max_rows = 128
    cuda = torch.device("cuda", 0)
    assert cg.lamport_applies((1, 7168), torch.bfloat16, cuda)
    assert cg.lamport_applies((128, 3584), torch.float16, cuda)
    assert not cg.lamport_applies((129, 7168), torch.bfloat16, cuda)  # above the rows cap
    assert not cg.lamport_applies((1, 7168), torch.float32, cuda)  # 32-bit
    assert not cg.lamport_applies((7168,), torch.bfloat16, cuda)  # 1-D
    assert not cg.lamport_applies((1, 7168), torch.bfloat16, torch.device("cpu"))
    assert not cg.symm_applies((1, 7168), torch.bfloat16, cuda)  # taken by the Lamport tier
    assert cg.symm_applies((200, 7168), torch.bfloat16, cuda)  # 2.9 MB: the multicast ring
    cg._lamport_enabled = False
    assert not cg.lamport_applies((1, 7168), torch.bfloat16, cuda)
    assert cg.symm_applies((1, 7168), torch.bfloat16, cuda)
    assert not CommGroup.trivial().lamport_applies((1, 7168), torch.bfloat16, cuda)


def test_lamport_channel_kinds(monkeypatch):
    """All-reduces get flashinfer's kernel when selected, all-gathers always M*'s kernel; a
    flashinfer failure falls back to M*'s kernel for the reduces too."""
    import mstar.distributed.lamport_allreduce as la
    from mstar.distributed.communication import CommGroup

    made = []

    class FakeFI:
        def __init__(self, group, rank, world, max_rows, width, dtype, device):
            made.append(("fi", width))

    class FakeLamport:
        def __init__(self, group_name, rank, world, max_rows, width, dtype, device):
            made.append(("ours", width))

    monkeypatch.setattr(la, "FlashInferAllReduce", FakeFI)
    monkeypatch.setattr(la, "LamportAllReduce", FakeLamport)
    cg = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    cg.device_group = types.SimpleNamespace(group_name="g")
    monkeypatch.setattr(cg, "_symm_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    cg._lamport_enabled = cg._lamport_flashinfer = True
    cg._lamport_channels = {}
    x = types.SimpleNamespace(shape=(4, 7168), dtype=torch.bfloat16, device=torch.device("cuda", 0),
                              stride=lambda i: (7168, 1)[i])
    assert isinstance(cg._lamport_channel(x), FakeFI)
    assert isinstance(cg._lamport_channel(x, gather=True), FakeLamport)
    assert isinstance(cg._lamport_channel(x), FakeFI)  # cached, not rebuilt
    assert made == [("fi", 7168), ("ours", 7168)]
    # flashinfer refusing (missing module, workspace failure) -> M*'s kernel for the reduces
    def boom(*a, **k):
        raise ImportError("no flashinfer")
    monkeypatch.setattr(la, "FlashInferAllReduce", boom)
    cg._lamport_channels = {}
    assert isinstance(cg._lamport_channel(x), FakeLamport) and not cg._lamport_flashinfer
    # `lamport` mode: ours for everything
    cg2 = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    cg2.device_group = types.SimpleNamespace(group_name="g")
    monkeypatch.setattr(cg2, "_symm_available", lambda: True)
    cg2._lamport_enabled, cg2._lamport_flashinfer, cg2._lamport_channels = True, False, {}
    made.clear()
    assert isinstance(cg2._lamport_channel(x), FakeLamport) and made == [("ours", 7168)]
