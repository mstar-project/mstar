"""Replay a captured GDN stack across several requests, the way serving does.

Standalone because the server path costs a couple of minutes of inductor
autotune per attempt, and the failure only shows from the *second* request: the
first one looks perfect.

Three things turned out to be necessary to see it at all, which is why this is
not a two-line test:

* **Depth.** A captured graph keeps its intermediates in a private pool, so a
  kernel that allocates with ``empty`` and writes only the ``cu_seqlens`` range
  leaves a bucket's padded tail holding the last replay's values. With one
  layer that goes nowhere; the residual stack is what compounds it.
* **Both walks.** Serving captures one graph per walk and alternates them.
* **Request turnover.** Slots are freed and handed back out between requests.

Run directly for a report, or under pytest for the assertion:

    python test/modular/test_gdn_cuda_graph.py
"""

from __future__ import annotations

import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.linear_attn import (
    LinearAttnConfig,
    LinearAttnManager,
    LinearAttnSpec,
    LinearAttnStep,
)
from mstar.engine.resources.linear_attn.config import LinearAttnVariant
from mstar.engine.resources.recurrent import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStatePool,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext
from mstar.model.components.linear_attn import GatedDeltaNet, GDNProjLayout
from mstar.model.components.norm import RMSNorm

GDN_STATE, LINEAR_ATTN = "gdn_state", "linear_attn"
BUCKET_TOKENS = 32  # a capture bucket's token slots

# Two shapes. The small one runs anywhere in a second and shows whether a
# bucket's padded tail goes stale at all; the real one is Qwen3.5-4B's, and is
# what makes a stale tail actually blow up — randomly initialised layers are
# contractive, trained ones are not.
SMALL = dict(
    geometry=DeltaNetGeometry(2, 4, 128, 128, 4), hidden=512, num_layers=24,
)
REAL_4B = dict(
    geometry=DeltaNetGeometry(16, 32, 128, 128, 4), hidden=2560, num_layers=24,
)


def build(device: torch.device, shape: dict):
    pool_spec = RecurrentStateSpec(
        GDN_STATE, {"llm"},
        RecurrentStateConfig(
            num_layers=shape["num_layers"], blocks=shape["geometry"].to_blocks(),
            max_slots=8,
        ),
    )
    pool = RecurrentStatePool.build(pool_spec, EngineResourceInfo(device=device))
    manager = LinearAttnManager.build(
        LinearAttnSpec(
            LINEAR_ATTN, {"llm"},
            LinearAttnConfig(
                recurrent_state=GDN_STATE, variant=LinearAttnVariant.GDN,
            ),
        ),
        EngineResourceInfo(device=device, dependencies={GDN_STATE: pool_spec}),
    )
    return pool, manager


class Stack(torch.nn.Module):
    """Real ``GatedDeltaNet`` layers, residually chained.

    The model's attention and MLP blocks are left out — this is about the
    recurrent path — but the depth and the residual are kept, since those are
    what let a poisoned padded tail grow from one layer to the next.
    """

    def __init__(self, pool, manager, device: torch.device, shape: dict):
        super().__init__()
        g, hidden, depth = shape["geometry"], shape["hidden"], shape["num_layers"]
        self.layers = torch.nn.ModuleList(
            GatedDeltaNet(
                hidden_size=hidden,
                num_k_heads=g.num_k_heads,
                num_v_heads=g.num_v_heads,
                head_k_dim=g.head_k_dim,
                head_v_dim=g.head_v_dim,
                conv_kernel_size=g.conv_kernel_size,
                layout=GDNProjLayout.SPLIT,
                linear_attn_key=LINEAR_ATTN,
                state_key=GDN_STATE,
            )
            for _ in range(depth)
        )
        # gemma-style, as Qwen3.5's plain norms are
        self.norms = torch.nn.ModuleList(
            RMSNorm(hidden, eps=1e-6, gemma_mode=True) for _ in range(depth)
        )
        self.to(device=device, dtype=torch.bfloat16)
        self.requires_grad_(False).eval()
        for layer in self.layers:
            layer.bind_resources({GDN_STATE: pool, LINEAR_ATTN: manager})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            layer.mix.bind_step("main")
            layer.mix.set_layer_idx(i)
            x = x + layer(norm(x))
        return x

    def probe(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Same forward, but inlined so each layer's two kernels are visible.

        Returns the output plus a ``[depth, 3]`` finite flag per layer — after
        the conv, after the delta rule, after the residual — recorded on device
        so it survives being captured and read back after a replay.
        """
        flags = []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            layer.mix.bind_step("main")
            layer.mix.set_layer_idx(i)
            h = norm(x)
            tokens = h.shape[0]
            qkv = layer.mix.conv(
                layer.in_proj_qkv(h),
                weight=layer.conv1d.weight.squeeze(1),
                bias=layer.conv1d.bias,
            )
            after_conv = torch.isfinite(qkv).all()
            q, k, v = torch.split(
                qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1,
            )
            core = layer.mix(
                q.view(tokens, layer.num_k_heads, layer.head_k_dim),
                k.view(tokens, layer.num_k_heads, layer.head_k_dim),
                v.view(tokens, layer.num_v_heads, layer.head_v_dim),
                layer.in_proj_a(h), layer.in_proj_b(h),
                layer.A_log, layer.dt_bias,
            )
            after_rule = torch.isfinite(core).all()
            z = layer.in_proj_z(h).view(tokens, layer.num_v_heads, layer.head_v_dim)
            x = x + layer.out_proj(
                layer.norm(core, z).reshape(tokens, layer.value_dim)
            )
            flags.append(torch.stack([after_conv, after_rule, torch.isfinite(x).all()]))
        return x, torch.stack(flags)


def plan_step(pool, manager, rid: str, span: int, lease: SlotLease | None, walk: str):
    """One step's planning, as the runner does it: pool first, then the backend
    off its result. Always outside capture."""
    ctx = StepContext(
        request_ids=[rid], graph_walk=walk, slot=0,
        capture=False, slot_lease=lease,
    )
    segments = (Segment(rid, "main", span),)
    rstep = RecurrentStep(segments=segments)
    assert pool.admit(rstep, ctx).ok
    ctx.plan_results[GDN_STATE] = pool.plan(rstep, ctx)
    manager.plan(LinearAttnStep(segments=segments), ctx)
    return rstep, ctx


def _capture(stack, x, probe: bool = False):
    """Warm up on a side stream, then record — what the runner does."""
    fn = stack.probe if probe else stack
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(x)
    return graph, out


def load_real_weights(stack, repo) -> int:
    """Fill the stack from a Qwen3.5 checkpoint's gated-delta-net layers.

    Only the recurrent layers and the norm in front of each; the stack has no
    attention or MLP to fill. Returns how many tensors landed.
    """
    from mstar.model.loader.iterators import iter_safetensors_shards
    from mstar.model.qwen3_5.config import Qwen3_5Config

    cfg = Qwen3_5Config.from_hf(repo)
    # stack position -> our index, for the linear-attention layers only
    where = {
        f"model.language_model.layers.{pos}.": i
        for i, pos in enumerate(cfg.linear_layer_indices)
    }
    params = dict(stack.named_parameters())
    loaded = 0
    for key, tensor in iter_safetensors_shards(repo, device="cpu"):
        for prefix, i in where.items():
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if rest.startswith("linear_attn."):
                name = f"layers.{i}.{rest[len('linear_attn.'):]}"
            elif rest.startswith("input_layernorm."):
                name = f"norms.{i}.weight"
            else:
                continue
            target = params.get(name)
            if target is not None:
                target.data.copy_(tensor.to(target.device, target.dtype))
                loaded += 1
            break
    return loaded


def run_case(
    capture: bool, spans=(14, 9, 20), decode_steps: int = 4, seed: int = 0,
    shape: dict | None = None, repo=None,
) -> list[dict]:
    """Each span is one request: prefill, a few decode steps, then remove."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    shape = shape or SMALL
    hidden = shape["hidden"]
    pool, manager = build(device, shape)
    manager.build_cuda_graph_buffers([], max_bs=1, max_seq_len=BUCKET_TOKENS)
    stack = Stack(pool, manager, device, shape)
    if repo is not None:
        load_real_weights(stack, repo)

    pre_lease = SlotLease(
        slot=0,
        bucket=BucketKey(graph_walk="prefill_text", bs=1, num_tokens=BUCKET_TOKENS),
    ) if capture else None
    dec_lease = SlotLease(
        slot=0, bucket=BucketKey(graph_walk="decode", bs=1, num_tokens=1),
    ) if capture else None

    # Static inputs padded to the bucket: replay writes the real prefix and
    # leaves the tail alone, as `CudaGraphRunner._stage` does.
    sx = torch.zeros(BUCKET_TOKENS, hidden, device=device, dtype=torch.bfloat16)
    dx = torch.zeros(1, hidden, device=device, dtype=torch.bfloat16)

    pre_graph = pre_out = dec_graph = None
    if capture:
        # Capture against a dummy request and hand its slot back, as
        # `CudaGraphRunner` does with its `__cg_*` rows. Recording against a
        # real request would leave that request's first step looking like a
        # capture rather than a replay.
        dummy = "__cg_dummy__"
        pool.ingest_request(dummy)
        drs, dcs = plan_step(pool, manager, dummy, BUCKET_TOKENS, pre_lease, "prefill_text")
        pre_graph, pre_out = _capture(stack, sx)
        pool.commit(drs, dcs)
        drs, dcs = plan_step(pool, manager, dummy, 1, dec_lease, "decode")
        dec_graph, _ = _capture(stack, dx)
        pool.commit(drs, dcs)
        pool.remove_request(dummy)

    results = []
    for i, span in enumerate(spans):
        rid = f"r{i}"
        pool.ingest_request(rid)
        rstep, ctx = plan_step(pool, manager, rid, span, pre_lease, "prefill_text")
        sx[:span].copy_(torch.randn(span, hidden, device=device).bfloat16())

        if not capture:
            out = stack(sx[:span])
        else:
            pre_graph.replay()
            out = pre_out
        pool.commit(rstep, ctx)

        for _ in range(decode_steps):
            drstep, dctx = plan_step(pool, manager, rid, 1, dec_lease, "decode")
            dx.copy_(torch.randn(1, hidden, device=device).bfloat16())
            if not capture:
                stack(dx)
            else:
                dec_graph.replay()
            pool.commit(drstep, dctx)

        slot = pool._slots[rid]["main"].index
        state = pool.block("state", shape["num_layers"] - 1)
        results.append({
            "request": i,
            "span": span,
            "slot": slot,
            "out_finite": bool(torch.isfinite(out[:span]).all()),
            "state_finite": bool(torch.isfinite(state[slot]).all()),
            "state_absmax": round(state[slot].abs().max().item(), 4),
            "tail_absmax": round(out[span:].float().abs().max().item(), 4)
            if out.shape[0] > span else 0.0,
        })
        pool.remove_request(rid)
    return results


def test_gdn_stack_survives_replay():
    for row in run_case(capture=True, shape=SMALL):
        assert row["state_finite"], (
            f"recurrent state went non-finite on request {row['request']}: {row}"
        )


def bisect_layers(spans=(14, 9), seed: int = 0) -> None:
    """Replay a probed capture and report the first kernel that goes non-finite."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    shape = SMALL
    hidden = shape["hidden"]
    pool, manager = build(device, shape)
    manager.build_cuda_graph_buffers([], max_bs=1, max_seq_len=BUCKET_TOKENS)
    stack = Stack(pool, manager, device, shape)

    pre_lease = SlotLease(
        slot=0,
        bucket=BucketKey(graph_walk="prefill_text", bs=1, num_tokens=BUCKET_TOKENS),
    )
    sx = torch.zeros(BUCKET_TOKENS, hidden, device=device, dtype=torch.bfloat16)

    dummy = "__cg_dummy__"
    pool.ingest_request(dummy)
    drs, dcs = plan_step(pool, manager, dummy, BUCKET_TOKENS, pre_lease, "prefill_text")
    graph, (out, flags) = _capture(stack, sx, probe=True)
    pool.commit(drs, dcs)
    pool.remove_request(dummy)

    for i, span in enumerate(spans):
        rid = f"r{i}"
        pool.ingest_request(rid)
        rstep, ctx = plan_step(pool, manager, rid, span, pre_lease, "prefill_text")
        sx[:span].copy_(torch.randn(span, hidden, device=device).bfloat16())
        graph.replay()
        f = flags.cpu()
        bad = [
            (layer, ["conv", "delta_rule", "residual"][slot])
            for layer in range(f.shape[0]) for slot in range(3) if not f[layer, slot]
        ]
        print(f"  replay {i} (span={span}): first non-finite = {bad[0] if bad else None}"
              f"  (of {len(bad)} points)")
        pool.commit(rstep, ctx)
        pool.remove_request(rid)


if __name__ == "__main__":
    import sys

    repo = None
    shape = SMALL
    if "--real" in sys.argv:
        from huggingface_hub import snapshot_download

        repo = snapshot_download("Qwen/Qwen3.5-4B")
        shape = REAL_4B
        print("using Qwen3.5-4B weights and geometry")
    if "--bisect" in sys.argv:
        print("=== which kernel goes non-finite first ===")
        bisect_layers()
        sys.exit(0)
    for mode in (False, True):
        print(f"=== capture={mode} ===")
        for row in run_case(capture=mode, shape=shape, repo=repo):
            print("   ", row)
