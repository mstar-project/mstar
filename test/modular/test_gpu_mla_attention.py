"""GPU: the FlashInfer MLA attention resource over a ``KVLayout.MLA`` cache vs. a dense
reference, prefill then decode, built and planned through the resource runner. Skipped
without CUDA."""
import pytest
import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    KVConfig,
    KVLayout,
    KVSpec,
    KVStep,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
    resolve_spec_dependencies,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


def _build(specs, device=DEV):
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo

    by_key = resolve_spec_dependencies(specs)
    resources = {}
    transfer = TransferEngineInfo("local", "local", LocalTransferEngine("localhost"))
    for key, spec in by_key.items():
        info = EngineResourceInfo(device=device, kv_dtype=torch.bfloat16, transfer_engine_info=transfer,
                                  dependencies={d: by_key[d] for d in spec.depends_on()})
        resources[key] = build_resource(spec, info)
    return resources, StepRunner(resources)


def _step(runner, keys_steps, segs, walk):
    step = SubmoduleStep(segments=segs, steps=keys_steps)
    ctx = StepContext(request_ids=[s.request_id for s in segs], graph_walk=walk, slot=0, capture=False)
    step.set_ctx(ctx)
    assert runner.admit(step).ok
    runner.plan(step)
    return step


@pytest.mark.parametrize(
    "heads,lora,rope,page", [(4, 512, 64, 64), (8, 512, 64, 64), (12, 512, 64, 16), (4, 64, 16, 16)])
def test_mla_wrapper_matches_dense_reference(heads, lora, rope, page):
    """FlashInfer's Hopper MLA kernel at the real latent shape (two head counts / page
    sizes) and the torch fallback at a small shape, both against dense attention."""
    torch.manual_seed(0)
    kv_cfg = KVConfig(num_layers=1, num_kv_heads=1, head_dim=lora + rope, max_seq_len=1024, max_num_pages=64,
                      page_size=page, num_qo_heads=heads, layout=KVLayout.MLA, kv_lora_rank=lora, qk_rope_head_dim=rope)
    scale = (128 + rope) ** -0.5
    specs = [
        KVSpec(resource_key="kv", nodes={"n"}, config=kv_cfg),
        AttentionSpec(resource_key="attn", nodes={"n"},
                      config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLASHINFER_MLA, sm_scale=scale)),
    ]
    res, runner = _build(specs)
    kv, attn = res["kv"], res["attn"]
    runner.ingest_request("a")
    runner.ingest_request("b")
    lens = [37, 5]
    t = sum(lens)
    step = _step(runner, {"kv": KVStep(), "attn": AttentionStep(causal=True)},
                 [Segment("a", "main", lens[0]), Segment("b", "main", lens[1])], "prefill")
    latent = torch.randn(t, lora + rope, device=DEV, dtype=torch.bfloat16)
    q_lat = torch.randn(t, heads, lora, device=DEV, dtype=torch.bfloat16)
    q_pe = torch.randn(t, heads, rope, device=DEV, dtype=torch.bfloat16)
    kv.set_default_layer_idx(0)
    attn.set_default_layer_idx(0)
    kv.set_default_label("main")
    attn.set_default_label("main")
    kv.write_kv(latent)
    out = attn.run(q_lat, kv_cache_layer=kv.layer_view(), q_pe=q_pe)
    runner.commit(step)

    def dense(ql, qp, lat):
        c, kp = lat.float().split([lora, rope], dim=-1)
        s = (torch.einsum("qhl,kl->hqk", ql.float(), c) + torch.einsum("qhr,kr->hqk", qp.float(), kp)) * scale
        n, m = ql.shape[0], lat.shape[0]
        mask = torch.arange(m, device=DEV)[None, :] <= (torch.arange(n, device=DEV)[:, None] + (m - n))
        s = s.masked_fill(~mask[None], float("-inf"))
        return torch.einsum("hqk,kl->qhl", torch.softmax(s, -1), c)

    ref_a = dense(q_lat[: lens[0]], q_pe[: lens[0]], latent[: lens[0]])
    ref_b = dense(q_lat[lens[0]:], q_pe[lens[0]:], latent[lens[0]:])
    torch.testing.assert_close(out[: lens[0]].float(), ref_a, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out[lens[0]:].float(), ref_b, rtol=2e-2, atol=2e-2)
    # decode: one new token per request against the resident context
    step2 = _step(runner, {"kv": KVStep(), "attn": AttentionStep(causal=True)},
                  [Segment("a", "main", 1), Segment("b", "main", 1)], "decode")
    lat2 = torch.randn(2, lora + rope, device=DEV, dtype=torch.bfloat16)
    ql2 = torch.randn(2, heads, lora, device=DEV, dtype=torch.bfloat16)
    qp2 = torch.randn(2, heads, rope, device=DEV, dtype=torch.bfloat16)
    kv.write_kv(lat2)
    out2 = attn.run(ql2, kv_cache_layer=kv.layer_view(), q_pe=qp2)
    runner.commit(step2)
    ref2a = dense(ql2[:1], qp2[:1], torch.cat([latent[: lens[0]], lat2[:1]]))
    ref2b = dense(ql2[1:], qp2[1:], torch.cat([latent[lens[0]:], lat2[1:]]))
    torch.testing.assert_close(out2[:1].float(), ref2a, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out2[1:].float(), ref2b, rtol=2e-2, atol=2e-2)
