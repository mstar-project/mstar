"""Decode-step host-vs-GPU profile for Zonos2 (Phase 2/3 ROI gate).

Question this answers: is a Zonos2 decode step launch/host-bound (CPU spends
longer *issuing* kernels than the GPU spends running them) or compute-bound?
CUDA-graph capture collapses per-step kernel launches + host syncs into ~one
replay, so it only pays off when decode is launch-bound.

Method (per batch size B):
  * Build ``Zonos2ForCausalLM`` at the RELEASED dims (from params.json), bf16,
    CUDA, random weights (timing is data-shape-driven, not weight-driven).
  * Drive a decode step = full 28-layer forward + multi-codebook head, with a
    fixed-length static KV stub for attention (real serving uses FlashInfer;
    one attention kernel per layer either way, so the *launch profile* — what
    determines launch-boundness — is representative; GEMM shapes are exact).
  * Pipeline N steps with NO per-step sync and measure:
      - t_issue : CPU wall to enqueue all N steps (returns before GPU drains)
      - gpu     : CUDA-event elapsed over the same region (GPU busy+gaps)
    cpu_issue/step vs gpu/step is the verdict:
      cpu_issue >> gpu  -> launch-bound  -> capture is a big win
      cpu_issue << gpu  -> compute-bound -> capture ROI low
  * Separately time the eager sampler (gather + ``_sample_in_graph``, i.e. its
    Python + host syncs) to size what capture eliminates independently of the
    transformer.

Phase 3: the multi-codebook sampler is now folded INTO the captured region
(``_sample_in_graph``), with only the per-step gather left host-side — so the
``captured`` column below is the full decode step (forward + head + sampler),
not just the transformer.

Throwaway scratch profiling; not a test. Run on a GPU box:
    python test/scratch/zonos2_decode_profile.py
"""
from __future__ import annotations

import glob
import json
import time
from types import SimpleNamespace

import torch

from mstar.model.zonos2.config import load_zonos2_config
from mstar.model.zonos2.components.language_model import Zonos2ForCausalLM
from mstar.model.zonos2.submodules import Zonos2LLMSubmodule
from mstar.model.zonos2.tts_sampling import TTSSamplingParams

DEVICE = "cuda"
DTYPE = torch.bfloat16
CTX_LEN = 512          # representative decode context length for the KV stub
N_STEPS = 100          # timed decode steps
WARMUP = 20
BATCH_SIZES = [1, 4, 16, 32]


class _DecodeStubCache:
    """Minimal cache_handle: interleaved RoPE + SDPA over a fixed static KV.

    Not the real FlashInfer path, but issues one attention kernel per layer
    against a fixed [B, kv_heads, CTX_LEN, head_dim] context, so the per-step
    kernel-launch count matches real decode. Numerics are irrelevant here.
    """

    def __init__(self, cfg, batch: int):
        self.n_layers = cfg.num_layers
        self.head_dim = cfg.head_dim
        self.n_kv = cfg.num_kv_heads
        self.n_q = cfg.num_qo_heads
        self.rep = self.n_q // self.n_kv
        self.layer_idx = 0
        # Static KV context per layer (never grown — fixed-shape decode proxy).
        self.k_ctx = [
            torch.randn(batch, self.n_kv, CTX_LEN, self.head_dim, device=DEVICE, dtype=DTYPE)
            for _ in range(self.n_layers)
        ]
        self.v_ctx = [
            torch.randn(batch, self.n_kv, CTX_LEN, self.head_dim, device=DEVICE, dtype=DTYPE)
            for _ in range(self.n_layers)
        ]
        # Precompute a rope table for CTX_LEN+1 positions.
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.head_dim, 2, device=DEVICE).float() / self.head_dim))
        t = torch.arange(CTX_LEN + 1, device=DEVICE).float()
        freqs = torch.outer(t, inv)
        self.cos = freqs.cos().to(DTYPE)
        self.sin = freqs.sin().to(DTYPE)

    def set_layer_idx(self, i):
        self.layer_idx = i

    def apply_rope(self, q, k, rope_theta, interleave=True):
        # q: (B, n_q, head_dim), k: (B, n_kv, head_dim) at the current position.
        pos = CTX_LEN
        cos = self.cos[pos].view(1, 1, -1)
        sin = self.sin[pos].view(1, 1, -1)
        def rot(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos
            return out
        return rot(q), rot(k)

    def run_attention(self, q, k, v):
        # q,k,v: (B, heads, head_dim) for the single decode token.
        B = q.shape[0]
        i = self.layer_idx
        kc, vc = self.k_ctx[i], self.v_ctx[i]  # (B, n_kv, CTX, dim)
        # append current token's k/v (fixed shape: drop oldest, keep CTX)
        k_all = torch.cat([kc[:, :, 1:, :], k.unsqueeze(2)], dim=2)  # (B, n_kv, CTX, dim)
        v_all = torch.cat([vc[:, :, 1:, :], v.unsqueeze(2)], dim=2)
        # GQA expand
        k_all = k_all.repeat_interleave(self.rep, dim=1)  # (B, n_q, CTX, dim)
        v_all = v_all.repeat_interleave(self.rep, dim=1)
        qh = q.unsqueeze(2)  # (B, n_q, 1, dim)
        o = torch.nn.functional.scaled_dot_product_attention(qh, k_all, v_all)  # (B, n_q, 1, dim)
        return o.squeeze(2)  # (B, n_q, dim)

    def advance_seq_lens(self):
        pass


def _build_model(cfg):
    torch.manual_seed(0)
    model = Zonos2ForCausalLM(cfg).eval()
    model = model.to(DEVICE, DTYPE)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02)
    return model


def _decode_inputs(cfg, batch):
    # (B, n_codebooks + 1): audio codebook columns then a text-placeholder col.
    # Audio cols in [0, codebook_size); the final text col must be < text_vocab+1
    # (the text embedder's row count), so use the text_placeholder id itself.
    audio = torch.randint(0, cfg.codebook_size, (batch, cfg.n_codebooks), device=DEVICE, dtype=torch.long)
    text = torch.full((batch, 1), cfg.text_vocab, device=DEVICE, dtype=torch.long)
    return torch.cat([audio, text], dim=1)


def _bench(fn, n=N_STEPS, w=WARMUP):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t) / n


def _make_sub(model, cfg):
    return Zonos2LLMSubmodule(
        model=model, n_codebooks=cfg.n_codebooks, text_vocab=cfg.text_vocab,
        eoa_id=cfg.eoa_id, params=TTSSamplingParams(seed=0),
    )


def _engine_like(rids):
    # Stand-in for ModelInputsFromEngine: eager path (real_request_ids=None),
    # so _prepare_sampler_step recovers rids straight from request_ids.
    return SimpleNamespace(request_ids=rids, real_request_ids=None, cache_manager=None)


@torch.no_grad()
def profile_forward(model, cfg, batch):
    """Eager vs CUDA-graph-captured decode step (Phase 3: sampler in-graph).

    The eager/graph ratio is the launch-boundness verdict: a large ratio means
    the step is dominated by per-kernel launch + host dispatch overhead (which
    capture collapses), i.e. capture has high ROI. Capture succeeding at all
    also proves the forward (incl. the Phase-1 fused-MoE dispatch) AND the
    multi-codebook sampler are host-sync-free and thus graph-safe.

    The captured region is ``forward + head + _sample_in_graph``; only the
    per-step gather (pinned H2D slot-index copy) is left host-side, exactly as
    the engine's preprocess/replay split does it.

    NOTE: the eager baseline is un-compiled. The engine torch.compiles the real
    forward, so this ratio is an upper bound on the *incremental* value of
    graphs over compile — but compile does not remove kernel-launch overhead the
    way graphs do (the codebase runs both together for captured models).
    """
    cache = _DecodeStubCache(cfg, batch)
    ids = _decode_inputs(cfg, batch).clone()  # static input buffer for capture

    sub = _make_sub(model, cfg)
    ei = _engine_like([f"r{i}" for i in range(batch)])
    sub._prepare_sampler_step(ei, padded_bs=batch)  # allocate + register + gather

    def gather():
        # Host-side per-step gather (outside the captured graph).
        sub._prepare_sampler_step(ei, padded_bs=batch)

    def step():
        h = model(ids, cache)
        logits = model.compute_logits(h[-batch:])   # (B, C, V)
        return sub._sample_in_graph(logits)          # (B, C + 1)

    def eager_step():
        gather()
        step()

    eager_ms = _bench(eager_step)

    # Warm up on a side stream (Triton autotune / cuBLAS workspaces), then capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    def replay_step():
        gather()          # host-side, eager
        g.replay()        # captured forward + head + sampler

    graph_ms = _bench(replay_step)

    return {"eager_ms": eager_ms, "graph_ms": graph_ms}


@torch.no_grad()
def profile_sample(model, cfg, batch):
    """Eager sampler host overhead: per-step gather + ``_sample_in_graph``.

    This is the work capture removes from the launch path once the sampler is
    folded into the graph (Phase 3)."""
    sub = _make_sub(model, cfg)
    ei = _engine_like([f"r{i}" for i in range(batch)])
    logits = torch.randn(batch, cfg.n_codebooks, cfg.audio_vocab, device=DEVICE, dtype=DTYPE)

    def samp():
        sub._prepare_sampler_step(ei, padded_bs=batch)
        sub._sample_in_graph(logits)

    return {"sample_ms": _bench(samp)}


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    p = glob.glob("/home/stephenduan/.cache/huggingface/models--Zyphra--ZONOS2/snapshots/*/params.json")[0]
    cfg = load_zonos2_config(json.load(open(p)))
    print(f"device={torch.cuda.get_device_name()}  dtype={DTYPE}  ctx={CTX_LEN}  N={N_STEPS}")
    print(f"dims: L={cfg.num_layers} d={cfg.hidden_size} inter={cfg.moe_inter} "
          f"E={cfg.moe_n_experts} C={cfg.n_codebooks} V={cfg.audio_vocab}\n")
    model = _build_model(cfg)

    hdr = (f"{'B':>3} | {'eager':>8} {'captured':>9} {'speedup':>8} | "
           f"{'_sample':>8} {'samp/captured':>14}")
    print(hdr); print("-" * len(hdr))
    for B in BATCH_SIZES:
        f = profile_forward(model, cfg, B)
        s = profile_sample(model, cfg, B)
        speedup = f["eager_ms"] / f["graph_ms"]
        # _sample host cost relative to the *captured* forward — i.e. how much it
        # would dominate once Phase 3 makes the forward cheap.
        samp_frac = s["sample_ms"] / (f["graph_ms"] + s["sample_ms"])
        print(f"{B:>3} | {f['eager_ms']:>7.2f}m {f['graph_ms']:>8.2f}m {speedup:>7.1f}x | "
              f"{s['sample_ms']:>7.2f}m {samp_frac*100:>13.1f}%")
    print("\nspeedup = eager/captured per step (sampler now in-graph): how "
          "launch-bound decode is (higher => more gained from capture).")
    print("samp/captured = eager sampler host time vs the captured full step: "
          "the launch-path work Phase 3 folds into the graph.")


if __name__ == "__main__":
    main()
