"""FlashInfer's kimi_k3_attn_res against the Triton read on a B300 / B200 (SM100 / SM103): skipped elsewhere.

``attn_res_read_norm`` takes FlashInfer's one launch there (the residual add, the block write, the mix and the output
RMSNorm). Per call shape (tokens x stack blocks, with and without the addend and the block write):

- the stream (``prefix += addend`` in place) and the written block are bit for bit torch's add;
- the other blocks are untouched;
- the normed output is within bf16 tolerance of the fp32 reference (mix and norm in fp32) and of the Triton read
  + RMSNorm (which rounds the mix to bf16 before the norm);
- a CUDA-graph replay (PDL on) gives the eager output for new inputs.

Run on one GPU: ``pytest mstar/model/kimi_k3/tests/test_attn_res_flashinfer_gpu.py -v``."""
import pytest
import torch

from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k3.components import attn_res

D, BLOCKS, EPS = attn_res.FI_HIDDEN, attn_res.FI_MAX_BLOCKS, 1e-5


def _device():
    if not torch.cuda.is_available():
        return None
    dev = torch.device("cuda", torch.cuda.current_device())
    return dev if attn_res._fi(dev) is not None else None


pytestmark = pytest.mark.skipif(_device() is None, reason="needs FlashInfer's kimi_k3_attn_res (SM100 / SM103)")


def _site(dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    norm_w = (1 + 0.1 * torch.randn(D, generator=g, device=dev)).bfloat16()
    proj_w = (0.05 * torch.randn(1, D, generator=g, device=dev)).bfloat16()
    out_norm = RMSNorm(D, eps=EPS).to(dev).bfloat16()
    with torch.no_grad():
        out_norm.weight.copy_((1 + 0.1 * torch.randn(D, generator=g, device=dev)).bfloat16())
    return g, norm_w, proj_w, out_norm


def _reference(stack, m, stream, norm_w, proj_w, out_w):
    """The read and its norm in fp32 (FlashInfer's documented semantics), rounded to bf16 once."""
    v = torch.cat((stack[:, :m].float(), stream.float().unsqueeze(1)), dim=1)  # (T, m + 1, D)
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    probs = (k * attn_res.score_weight(norm_w, proj_w)).sum(-1).softmax(-1)  # (T, m + 1)
    mixed = torch.einsum("tm,tmd->td", probs, v)
    return (mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + EPS) * out_w.float()).bfloat16()


def _close(got, want, what):
    diff = (got.float() - want.float()).abs()
    scale = want.float().abs().clamp_min(1.0)
    assert (diff / scale).max().item() <= 2 ** -5, f"{what}: max rel diff {(diff / scale).max().item():.4g}"
    assert diff.mean().item() <= 2e-3, f"{what}: mean diff {diff.mean().item():.4g}"


@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 16, 64, 100, 256, 512, 3072])
@pytest.mark.parametrize("m", [0, 1, 4, 7, 8])
@pytest.mark.parametrize("addend", [True, False])
@pytest.mark.parametrize("write", [False, True])
@torch.no_grad()
def test_flashinfer_read_matches_the_triton_read(tokens, m, addend, write):
    if write and m >= BLOCKS:
        pytest.skip("a block start writes block m < 8")
    dev = _device()
    g, norm_w, proj_w, out_norm = _site(dev, 1000 * tokens + 10 * m + 2 * addend + write)
    stack = torch.randn(tokens, BLOCKS, D, generator=g, device=dev).bfloat16()
    prefix = torch.randn(tokens, D, generator=g, device=dev).bfloat16()
    add = torch.randn(tokens, D, generator=g, device=dev).bfloat16() if addend else None
    stream_want = prefix + add if addend else prefix.clone()
    w = m if write else -1

    fi_stack, fi_prefix = stack.clone(), prefix.clone()
    assert attn_res.fi_serves(fi_stack, fi_prefix, (norm_w, proj_w.reshape(-1), out_norm.weight), add)
    out, stream = attn_res.attn_res_read_norm(fi_stack, m, fi_prefix, norm_w, proj_w, EPS, out_norm, addend=add,
                                              write=w)
    assert stream.data_ptr() == fi_prefix.data_ptr()  # the stream is prefix, added in place
    assert torch.equal(stream, stream_want)
    for b in range(BLOCKS):
        assert torch.equal(fi_stack[:, b], stream_want if b == w else stack[:, b]), f"block {b}"

    tr_stack = stack.clone()
    tr_out, tr_stream = attn_res.attn_res_read_norm(tr_stack, m, prefix.clone(), norm_w, proj_w, EPS, out_norm,
                                                    addend=add, write=w, flashinfer=False)
    assert torch.equal(tr_stream, stream_want) and torch.equal(tr_stack, fi_stack)
    _close(out, _reference(stack if not write else fi_stack, m, stream_want, norm_w, proj_w, out_norm.weight),
           "flashinfer vs the fp32 reference")
    _close(out, tr_out, "flashinfer vs the Triton read + RMSNorm")


@torch.no_grad()
def test_flashinfer_read_replays_under_a_cuda_graph_with_pdl():
    """Captured after a kernel that writes its input (PDL: the program waits on it in-kernel), the replay's output is
    the eager one for new inputs."""
    dev = _device()
    assert attn_res.FI_PDL
    g, norm_w, proj_w, out_norm = _site(dev, 7)
    tokens, m = 8, 5
    src = torch.randn(tokens, D, generator=g, device=dev).bfloat16()
    add = torch.randn(tokens, D, generator=g, device=dev).bfloat16()
    stack = torch.randn(tokens, BLOCKS, D, generator=g, device=dev).bfloat16()
    prefix = torch.empty_like(src)

    def step():
        prefix.copy_(src * 1.5)  # the producer the program must wait on
        return attn_res.attn_res_read_norm(stack, m, prefix, norm_w, proj_w, EPS, out_norm, addend=add)[0]

    stream = torch.cuda.Stream(dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream(dev).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    for seed in range(3):
        src.copy_(torch.randn(tokens, D, generator=g, device=dev).bfloat16())
        graph.replay()
        got = out.clone()
        want = step()
        assert torch.equal(got, want), f"replay {seed}: max diff {(got.float() - want.float()).abs().max().item()}"
