"""Kimi K3.1's latent-MoE front at one TP8 rank (B300 / B200), per call, in a captured graph, up to z quantized
for the routed experts (MXFP8) and the router's logits ready:

- ``packed4480``: the packed fp32 GEMM over router | latent (``small_m_gemm``, with ``--table`` the backend the B300
  table names) and the latent's cast + MXFP8 quantization (``mxfp8_cast``): the overlapped path's main stream
  (``latent_moe_front: gemm``; the shared experts run beside it on the side stream).
- ``split``: the router's fp32 GEMM forked to the side stream, the latent's bf16 GEMM and FlashInfer's
  ``mxfp8_quantize`` on the main one, joined (``latent_moe_front: split``).
- ``packed6016``: the packed fp32 GEMM over router | latent | shared gate_up, the cast of its latent and gate_up
  columns, the SiTU kernel: the one-stream path.
- ``front``: FlashInfer's one-launch front (``mstar::k31_latent_front``), where its programs build.

    python benchmark/kimi_k3/latent_front.py [--table /workspace/results/k31-gemm/table_b300.json]
"""
import argparse

import torch

from mstar.engine import side_stream
from mstar.model.kimi_k3.components import latent_front as LF
from mstar.utils import small_m_gemm
from mstar.utils.fused_moe import mxfp8_cast  # noqa: F401  (registers mstar::split_cast_mxfp8)
from mstar.utils.situ import situ_and_mul

I_LOCAL = LF.SHARED // 8


def time_calls(fn, calls: int = 20, replays: int = 50) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(replays):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / (replays * calls)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    p.add_argument("--table", default=None)
    args = p.parse_args()
    if args.table:
        small_m_gemm.load_table(args.table)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    e, lat = LF.EXPERTS, LF.LATENT
    w = (torch.randn(e + lat + 2 * I_LOCAL, LF.HIDDEN, device=dev, generator=g) * 0.02).bfloat16()
    has_front = LF.unavailable(w, e, lat, LF.situ_spec(), 8) is None
    if has_front:
        LF.warm(w)
    # MXFP8 for the routed experts where they take it (SM100: TRT-LLM's MoE); elsewhere the casts alone
    quant = torch.cuda.get_device_capability()[0] >= 10
    if quant:
        from flashinfer.quantization.fp8_quantization import mxfp8_quantize
    print(f"# {torch.cuda.get_device_name()} table={args.table} front={'yes' if has_front else 'no'} "
          f"mxfp8={'yes' if quant else 'no'}  us per call")
    print(f"{'T':>5} {'packed4480':>11} {'split':>8} {'packed6016':>11} {'front':>8}")
    for T in map(int, args.tokens.split(",")):
        x = torch.randn(T, LF.HIDDEN, device=dev, generator=g).bfloat16()

        def front(x=x):
            torch.ops.mstar.k31_latent_front(x, w, e, lat)

        def split(x=x):
            logits, token = side_stream.fork(lambda t: small_m_gemm.linear(t, w[:e], out_dtype=torch.float32), x)
            z = small_m_gemm.linear(x, w[e:e + lat])
            if quant:
                mxfp8_quantize(z, is_sf_swizzled_layout=False)
            side_stream.join(token)

        def packed6016(x=x):
            h = small_m_gemm.linear(x, w, out_dtype=torch.float32)
            zg = h[:, e:].to(torch.bfloat16)
            gate, up = zg[:, lat:lat + I_LOCAL], zg[:, lat + I_LOCAL:]
            situ_and_mul(gate, up, 4.0, 25.0)

        def packed4480(x=x):
            h = small_m_gemm.linear(x, w[:e + lat], out_dtype=torch.float32)
            if quant:
                torch.ops.mstar.split_cast_mxfp8(h, e, lat)
            else:
                h[:, e:].to(torch.bfloat16)

        f = f"{time_calls(front):8.2f}" if has_front else f"{'-':>8}"
        print(f"{T:5d} {time_calls(packed4480):11.2f} {time_calls(split):8.2f} {time_calls(packed6016):11.2f} {f}",
              flush=True)


if __name__ == "__main__":
    main()
