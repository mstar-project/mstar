"""Decode attention over KV plans in their slots' kept buffers attends each row's
own tokens, as streams cross pages and slots take turns, on the real kernels."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("flashinfer")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from mstar.communication.tensors import LocalTransferEngine  # noqa: E402
from mstar.engine.resources import (  # noqa: E402
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    KVSpec,
    KVStep,
    PagedKVConfig,
    StepRunner,
)
from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo, build_resource  # noqa: E402
from mstar.engine.resources.kv.transfer import TransferEngineInfo  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segments, SlotLease, StepContext, SubmoduleStep  # noqa: E402

NODE, PAGE, QO, KV, HD = "dec", 16, 8, 2, 128
REAL, BS, SLOTS, STEPS = 3, 4, 2, 40


def _runner(tensor_cores: bool):
    kv_spec = KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=KV, head_dim=HD, max_seq_len=256, num_qo_heads=QO,
        page_size=PAGE, max_num_pages=64,
    ))
    attn_spec = AttentionSpec(resource_key="attn", nodes={NODE}, config=AttentionConfig(
        kv_cache="kv", decode_tensor_cores=tensor_cores,
    ))
    device = torch.device("cuda")
    resources = {}
    for spec, deps in ((kv_spec, {}), (attn_spec, {"kv": kv_spec})):
        resources[spec.resource_key] = build_resource(spec, EngineResourceInfo(
            device=device, dependencies=deps, nodes=frozenset({NODE}),
            transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
            needs_remote_transfer=False, compute_dtype=torch.bfloat16,
        ))
    return StepRunner(resources, node_resources={NODE: list(resources)}), resources


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``q`` [QO, HD] over one row's ``k``, ``v`` [tokens, KV, HD], a query head on its KV head."""
    k = k.float().repeat_interleave(QO // KV, dim=1)
    v = v.float().repeat_interleave(QO // KV, dim=1)
    scores = (q.float().unsqueeze(0) * k).sum(-1) / HD ** 0.5
    return (scores.softmax(0).unsqueeze(-1) * v).sum(0)


@pytest.mark.parametrize("tensor_cores", [False, True])
def test_each_row_attends_its_own_tokens(tensor_cores):
    torch.manual_seed(0)
    runner, resources = _runner(tensor_cores)
    kv, attn = resources["kv"], resources["attn"]
    bucket = BucketKey("decode", BS, BS)
    runner.build_accelerator_graph_buffers(
        [CGSlotSpec(bucket=bucket, slot=s, config=None) for s in range(SLOTS)], BS, BS,
    )
    cache = kv.kv_cache.tensor
    cache.copy_(torch.randn_like(cache, dtype=torch.float32).to(cache.dtype))
    rids = list(range(REAL))
    # each slot pads with rows of its own, as a replay's slots do
    pads = {s: [-(100 * (s + 1) + i) for i in range(BS - REAL)] for s in range(SLOTS)}
    for slot_pads in pads.values():
        for rid in slot_pads:
            runner.ingest_capture_row(rid)
    for rid in rids:
        runner.ingest_request(rid, {})
    for i in range(STEPS):
        slot = i % SLOTS
        padded = [*rids, *pads[slot]]
        segments = Segments(padded)
        step = SubmoduleStep(segments=segments, steps={"kv": KVStep(), "attn": AttentionStep(causal=False)})
        ctx = StepContext(list(rids), "decode", slot, False, slot_lease=SlotLease(slot=slot, bucket=bucket))
        ctx.set_padded_rids(padded)
        step.set_ctx(ctx)
        assert runner.admit(step).ok
        runner.plan(step)
        q = torch.randn(BS, QO, HD, device="cuda").bfloat16()
        out = attn.run(q, "main", cache[0])
        for row, rid in enumerate(rids):
            stream = kv._streams[rid]["main"]
            tokens = stream.stored_len + 1
            whole, rest = divmod(tokens, PAGE)
            pages = stream.page_indices.tolist()
            k = cache[0][pages, 0].reshape(-1, KV, HD)[:tokens]
            v = cache[0][pages, 1].reshape(-1, KV, HD)[:tokens]
            assert len(pages) == whole + bool(rest)
            torch.testing.assert_close(
                out[row].float(), _reference(q[row], k, v), atol=0.02, rtol=0.02,
                msg=f"row {row} at step {i}",
            )
        runner.commit(step)
        torch.cuda.synchronize()
    assert kv._slot_columns, "planned in the slots' buffers"
