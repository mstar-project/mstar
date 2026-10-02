"""GPU check for FA3MLAWrapper (MSTAR_MLA_DECODE_BACKEND=fa3): the same MLA
decode as FlashInferMLAWrapper over one latent cache, eagerly and replayed
from a CUDA graph re-planned with new lengths and pages after capture."""
import math

import pytest
import torch

from mstar.engine.resources.attn.wrappers import FA3MLAWrapper, FlashInferMLAWrapper, load_fa3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9 or load_fa3() is None,
    reason="needs sm90 and FA3",
)

HEADS, CKV, KPE, PAGE, MAX_SEQ = 16, 512, 64, 128, 16384
SCALE = 1.0 / math.sqrt(192)


def _plan_inputs(lengths, num_pages, gen):
    pages = [math.ceil(n / PAGE) for n in lengths]
    perm = torch.randperm(num_pages, generator=gen)[: sum(pages)].to(torch.int32)
    kv_indptr = torch.tensor([0] + list(torch.tensor(pages).cumsum(0)), dtype=torch.int32)
    qo_indptr = torch.arange(len(lengths) + 1, dtype=torch.int32)
    return qo_indptr, kv_indptr, perm, torch.tensor(lengths, dtype=torch.int32)


def _reference(latent, q_nope, q_pe, plan):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    fi = FlashInferMLAWrapper(ws, num_heads=HEADS, head_dim_ckv=CKV, head_dim_kpe=KPE, page_size=PAGE,
                              sm_scale=SCALE, backend="fa3")
    fi.plan(*plan)
    return fi.run(q_nope, q_pe, latent[..., :CKV], latent[..., CKV:])


@torch.no_grad()
def test_fa3_mla_matches_flashinfer_eager_and_replayed():
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    num_pages, bs = 512, 4
    latent = torch.randn(num_pages, PAGE, CKV + KPE, device="cuda", dtype=torch.bfloat16)
    q_nope = torch.randn(bs, HEADS, CKV, device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn(bs, HEADS, KPE, device="cuda", dtype=torch.bfloat16)
    ckv, kpe = latent[..., :CKV], latent[..., CKV:]

    plan = _plan_inputs([100, 1000, 5000, 9000], num_pages, gen)
    ref = _reference(latent, q_nope, q_pe, plan)
    eager = FA3MLAWrapper(num_heads=HEADS, head_dim_ckv=CKV, head_dim_kpe=KPE, page_size=PAGE,
                          sm_scale=SCALE, max_seq_len=MAX_SEQ)
    eager.plan(*plan)
    torch.testing.assert_close(eager.run(q_nope, q_pe, ckv, kpe), ref, rtol=2e-2, atol=2e-2)

    graphed = FA3MLAWrapper(num_heads=HEADS, head_dim_ckv=CKV, head_dim_kpe=KPE, page_size=PAGE,
                            sm_scale=SCALE, max_seq_len=MAX_SEQ, batch_size=bs, use_cuda_graph=True)
    graphed.plan(*plan)
    graphed.run(q_nope, q_pe, ckv, kpe)  # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = graphed.run(q_nope, q_pe, ckv, kpe)
    for lengths in ([100, 1000, 5000, 9000], [9000, 1, 300, 16384], [129, 128, 127, 4096]):
        plan = _plan_inputs(lengths, num_pages, gen)
        graphed.plan(*plan)
        graph.replay()
        torch.testing.assert_close(out, _reference(latent, q_nope, q_pe, plan), rtol=2e-2, atol=2e-2)
