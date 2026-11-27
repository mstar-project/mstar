"""What one draft token costs on the GPU: the Qwen3.5 MTP head at batch 1 plus the shared lm_head.

    python -m benchmark.spec_decode.draft_cost --model Qwen/Qwen3.5-9B --context 1024

Times, with CUDA events over repeated runs: the lm_head GEMM at 1..6 rows (a verify step's rows at concurrency 1),
each GEMM of the head at one row, and one complete draft step (fc, the full-attention decoder layer over a
fixed-length KV cache, final norm, lm_head, argmax) both eager and replayed from a HIP graph. The graph number is the
launch-free cost a captured engine step would pay; the eager number is what the acceptance harness pays.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

from benchmark.spec_decode.qwen3_5_mtp import BaseModel, MtpHead, load_base


def time_fn(fn, iters: int, repeats: int = 3) -> float:
    """Mean milliseconds per call, best of `repeats` blocks of `iters` calls."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best


def gemm_row(name: str, weight: torch.Tensor, rows: int, iters: int) -> dict:
    x = torch.randn(rows, weight.shape[1], device=weight.device, dtype=weight.dtype)
    ms = time_fn(lambda: F.linear(x, weight), iters)
    byte_count = weight.numel() * weight.element_size()
    return {"name": name, "rows": rows, "shape": list(weight.shape), "ms": ms, "mb": byte_count / 1e6,
            "tb_s": byte_count / (ms * 1e-3) / 1e12}


class DraftStep:
    """One draft token through the head with static buffers, so it can be captured into a HIP graph."""

    def __init__(self, head: MtpHead, base: BaseModel, context: int):
        self.head, self.base = head, base
        layer = head.layers[0]
        attn = layer.self_attn
        self.attn, self.layer = attn, layer
        cfg = base.config
        device, dtype = base.device, base.dtype
        self.kv_heads = cfg.num_key_value_heads
        self.groups = attn.num_key_value_groups
        self.head_dim = attn.head_dim
        hidden = cfg.hidden_size
        self.h = torch.randn(1, hidden, device=device, dtype=dtype)
        self.tok = torch.tensor([1], device=device)
        # Keys and values of `context` earlier positions; the new token lands in the last slot.
        self.K = torch.randn(1, self.kv_heads, context, self.head_dim, device=device, dtype=dtype)
        self.V = torch.randn(1, self.kv_heads, context, self.head_dim, device=device, dtype=dtype)
        pos = torch.tensor([[[context - 1]]], device=device).expand(3, 1, 1)
        self.cos, self.sin = base.rotary(self.h[None], pos)
        self.gqa = True
        self.out = None

    def attention(self, q, k, v):
        if self.gqa:
            try:
                return F.scaled_dot_product_attention(q, k, v, scale=self.attn.scaling, enable_gqa=True)
            except (RuntimeError, TypeError):
                self.gqa = False
        return F.scaled_dot_product_attention(
            q, repeat_kv(k, self.groups), repeat_kv(v, self.groups), scale=self.attn.scaling
        )

    def layer_forward(self) -> torch.Tensor:
        head, layer, attn = self.head, self.layer, self.attn
        emb = head.pre_fc_norm_embedding(self.base.embed(self.tok))
        x = head.fc(torch.cat((emb, head.pre_fc_norm_hidden(self.h)), dim=-1))[None]  # (1, 1, hidden)
        residual = x
        y = layer.input_layernorm(x)
        q, gate = torch.chunk(attn.q_proj(y).view(1, 1, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(1, 1, -1)
        q = attn.q_norm(q.reshape(1, 1, -1, self.head_dim)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(y).view(1, 1, -1, self.head_dim)).transpose(1, 2)
        v = attn.v_proj(y).view(1, 1, -1, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        self.K[:, :, -1:] = k
        self.V[:, :, -1:] = v
        o = self.attention(q, self.K, self.V).transpose(1, 2).reshape(1, 1, -1)
        x = residual + attn.o_proj(o * torch.sigmoid(gate))
        residual = x
        x = residual + layer.mlp(layer.post_attention_layernorm(x))
        return head.norm(x)[0, -1]

    def full(self) -> None:
        self.out = self.base.lm_head(self.layer_forward()).argmax()

    def layer_only(self) -> None:
        self.out = self.layer_forward()


def capture(fn) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--context", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args(argv)

    base, snapshot = load_base(args.model, args.device)
    head = MtpHead.from_snapshot(snapshot, base.config, base.device, base.dtype)
    print(f"{args.model} on {torch.cuda.get_device_name(0)}, torch {torch.__version__}, {base.dtype}")
    lm_head = base.lm_head.weight
    mb = lambda w: w.numel() * w.element_size() / 1e6  # noqa: E731
    layer_bytes = sum(mb(p) for p in head.parameters())
    print(f"mtp head weights {layer_bytes:.1f} MB, lm_head {mb(lm_head):.1f} MB")

    with torch.inference_mode():
        print("\nlm_head GEMM (bf16, torch F.linear):")
        for rows in range(1, 7):
            r = gemm_row("lm_head", lm_head, rows, args.iters)
            print(f"  rows={rows}: {r['ms']:.3f} ms  {r['tb_s']:.2f} TB/s")

        print("\nmtp head GEMMs at one row:")
        layer = head.layers[0]
        attn, mlp = layer.self_attn, layer.mlp
        weights = [
            ("fc", head.fc.weight), ("q_proj", attn.q_proj.weight), ("k_proj", attn.k_proj.weight),
            ("v_proj", attn.v_proj.weight), ("o_proj", attn.o_proj.weight), ("gate_proj", mlp.gate_proj.weight),
            ("up_proj", mlp.up_proj.weight), ("down_proj", mlp.down_proj.weight),
        ]
        total_ms = 0.0
        for name, w in weights:
            r = gemm_row(name, w, 1, args.iters)
            total_ms += r["ms"]
            print(f"  {name:10s} {str(r['shape']):16s} {r['ms']:.3f} ms  {r['mb']:7.1f} MB  {r['tb_s']:.2f} TB/s")
        lm = gemm_row("lm_head", lm_head, 1, args.iters)
        print(f"  GEMM sum: head {total_ms:.3f} ms + lm_head {lm['ms']:.3f} ms = {total_ms + lm['ms']:.3f} ms")

        for context in args.context:
            step = DraftStep(head, base, context)
            eager_full = time_fn(step.full, args.iters)
            eager_layer = time_fn(step.layer_only, args.iters)
            print(
                f"\ndraft step, context {context} (gqa sdpa={step.gqa}):\n"
                f"  eager: full {eager_full:.3f} ms, layer only {eager_layer:.3f} ms"
            )
            try:
                g_full = capture(step.full)
                graph_full = time_fn(g_full.replay, args.iters)
                g_layer = capture(step.layer_only)
                graph_layer = time_fn(g_layer.replay, args.iters)
            except RuntimeError as exc:
                print(f"  graph capture failed: {exc!r}")
                continue
            print(
                f"  graph: full {graph_full:.3f} ms, layer only {graph_layer:.3f} ms, "
                f"lm_head share {(graph_full - graph_layer) / graph_full * 100:.0f}%"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
