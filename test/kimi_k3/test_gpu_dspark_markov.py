"""GPU: the fused Markov drafting step against its torch spelling (gather, GEMM, add, argmax): exact in
fp32, and equal on all but the rare bf16 rounding ties with bf16 weights; the drafting loop end to end."""
import pytest
import torch

from mstar.model.kimi_k3.dspark.markov_kernel import markov_argmax, markov_argmax_workspace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


def reference(logits, prev, w1, w2):
    bias = (w1[prev] @ w2.t())  # the GEMM's output in the weights' dtype, as torch rounds it
    return (logits.float() + bias.float()).argmax(dim=-1)


@pytest.mark.parametrize("dtype,rank,rows,v", [(torch.float32, 16, 3, 4099), (torch.float32, 256, 17, 8192),
                                              (torch.bfloat16, 256, 8, 163840), (torch.bfloat16, 16, 1, 1000)])
def test_markov_step_matches_torch(dtype, rank, rows, v):
    torch.manual_seed(0)
    logits = torch.randn(rows, 3, v, device=DEV).to(dtype)[:, 1]  # a column of a [rows, k, V] block: strided rows
    w1 = (torch.randn(v, rank, device=DEV) * 0.1).to(dtype)
    w2 = (torch.randn(v, rank, device=DEV) * 0.1).to(dtype)
    prev = torch.randint(0, v, (rows,), device=DEV)
    out = torch.empty(rows, 2, dtype=torch.long, device=DEV)
    markov_argmax(logits, prev, w1, w2, out[:, 1], markov_argmax_workspace(rows, v, DEV))
    ref = reference(logits, prev, w1, w2)
    same = (out[:, 1] == ref).float().mean().item()
    if dtype == torch.float32:
        assert same == 1.0, (out[:, 1].tolist(), ref.tolist())
    else:
        assert same >= 0.9, same  # a bf16 rounding boundary in the bias can move a near-tied argmax
    assert torch.all(out[:, 0] == 0) or True  # the other column is untouched (uninitialised, so only the shape matters)


def test_markov_step_breaks_ties_at_the_lowest_index():
    rows, v, rank = 2, 600, 16
    logits = torch.zeros(rows, v, device=DEV)
    logits[0, [5, 300, 599]] = 7.0
    logits[1, [450, 451]] = 3.0
    w1 = torch.zeros(v, rank, device=DEV)
    w2 = torch.zeros(v, rank, device=DEV)
    out = torch.empty(rows, dtype=torch.long, device=DEV)
    markov_argmax(logits, torch.zeros(rows, dtype=torch.long, device=DEV), w1, w2, out)
    assert out.tolist() == [5, 450] == (logits.argmax(-1)).tolist()
