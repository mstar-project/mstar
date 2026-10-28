"""The Triton LoRA kernels in bf16 on a GPU (skipped elsewhere), at Qwen3.5-0.8B's shapes: deltas
against fp32 references over batches mixing adapter ranks, base rows and slot reuse, eagerly and
inside one captured CUDA graph replayed for changing adapter mixes.

    pytest test/integration/test_lora_triton_gpu.py -v
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
pytest.importorskip("triton")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from mstar.engine.resources.lora import LoraReqConfig, ops  # noqa: E402
from mstar.engine.resources.lora import backend as ref_ops  # noqa: E402
from mstar.engine.resources.lora.checkpoint import read_lora_file  # noqa: E402
from mstar.engine.resources.lora.identity import adapter_digest  # noqa: E402
from mstar.engine.resources.lora.pool import LoraAdapterPool  # noqa: E402
from mstar.engine.resources.lora.triton_backend import TritonLoraBackend  # noqa: E402
from mstar.engine.resources.step import BucketKey, ResourceStep, Segment, SlotLease, StepContext  # noqa: E402
from mstar.model.qwen3_5.config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5Config  # noqa: E402
from mstar.model.qwen3_5.lora import qwen3_5_lora_spec  # noqa: E402

# Qwen3.5-0.8B's geometry, two layers (a delta-net one and a full-attention one)
HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM = 1024, 3584, 8, 2, 256
LIN_HEADS, LIN_DIM = 16, 128
KEY_DIM = VALUE_DIM = LIN_HEADS * LIN_DIM
LAYERS = [LINEAR_ATTENTION, FULL_ATTENTION]
VOCAB = 248320  # every Qwen3.5 size's
BF16 = dict(device="cuda", dtype=torch.bfloat16)
BUCKET_BS, BUCKET_TOKENS = 4, 64
SHAPES = {  # HF projection -> (out, in)
    "self_attn.q_proj": (2 * HEADS * HEAD_DIM, HIDDEN), "self_attn.k_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.v_proj": (KV_HEADS * HEAD_DIM, HIDDEN), "self_attn.o_proj": (HIDDEN, HEADS * HEAD_DIM),
    "mlp.gate_proj": (INTER, HIDDEN), "mlp.up_proj": (INTER, HIDDEN), "mlp.down_proj": (HIDDEN, INTER),
    "linear_attn.in_proj_qkv": (2 * KEY_DIM + VALUE_DIM, HIDDEN), "linear_attn.in_proj_z": (VALUE_DIM, HIDDEN),
    "linear_attn.in_proj_a": (LIN_HEADS, HIDDEN), "linear_attn.in_proj_b": (LIN_HEADS, HIDDEN),
    "linear_attn.out_proj": (HIDDEN, VALUE_DIM), "lm_head": (VOCAB, HIDDEN),
}
TARGETS = {  # target -> its HF projections, in slice order
    "qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "o_proj": ("self_attn.o_proj",),
    "in_proj_fused": tuple(f"linear_attn.in_proj_{p}" for p in ("qkv", "z", "a", "b")),
    "out_proj": ("linear_attn.out_proj",), "gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
    "down_proj": ("mlp.down_proj",), "lm_head": ("lm_head",),
}


def write_adapter(path, rank, alpha, seed, projs):
    g = torch.Generator().manual_seed(seed)
    sd, ref = {}, {}
    for layer, kind in enumerate(LAYERS):
        attn = "self_attn." if kind == FULL_ATTENTION else "linear_attn."
        for proj in (p for p in projs if p.startswith(("mlp.", attn))):
            out, inp = SHAPES[proj]
            a, b = torch.randn(rank, inp, generator=g) / inp ** 0.5, torch.randn(out, rank, generator=g) / rank ** 0.5
            module = f"base_model.model.model.language_model.layers.{layer}.{proj}"
            sd[f"{module}.lora_A.weight"], sd[f"{module}.lora_B.weight"] = a, b
            sd[f"{module}.alpha"] = torch.tensor(float(alpha))
            ref[(layer, proj)] = (a, b, alpha / rank)
    save_file(sd, str(path))
    return ref


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    d = tmp_path_factory.mktemp("adapters")
    specs = {"r16": (16, 32.0, 1, tuple(SHAPES)), "r4": (4, 4.0, 2, ("self_attn.q_proj", "mlp.down_proj")),
             "r8": (8, 16.0, 3, tuple(SHAPES))}
    paths = {name: str(d / f"{name}.safetensors") for name in specs}
    return paths, {name: write_adapter(paths[name], *spec) for name, spec in specs.items()}


@pytest.fixture(scope="module")
def head_adapters(tmp_path_factory):
    """PEFT ``lm_head`` LoRAs at the real vocabulary, scaling 2."""
    d, paths, refs = tmp_path_factory.mktemp("heads"), {}, {}
    for name, rank, seed in (("h16", 16, 41), ("h4", 4, 42)):
        g = torch.Generator().manual_seed(seed)
        a, b = torch.randn(rank, HIDDEN, generator=g) / 32, torch.randn(VOCAB, rank, generator=g) / rank ** 0.5
        paths[name], refs[name] = str(d / f"{name}.safetensors"), {(0, "lm_head"): (a, b, 2.0)}
        save_file({"base_model.model.lm_head.lora_A.weight": a, "base_model.model.lm_head.lora_B.weight": b,
                   "base_model.model.lm_head.alpha": torch.tensor(2.0 * rank)}, paths[name])
    return paths, refs


def make_pool(paths, *, head=False, rank=0, world=1):
    """Two slots, max rank 16; ``head`` keeps only the LM head and embedding, at the real vocabulary."""
    spec = qwen3_5_lora_spec(Qwen3_5Config(
        num_hidden_layers=len(LAYERS), hidden_size=HIDDEN, intermediate_size=INTER,
        layer_types=list(LAYERS), num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM, linear_num_key_heads=LIN_HEADS, linear_num_value_heads=LIN_HEADS,
        linear_key_head_dim=LIN_DIM, linear_value_head_dim=LIN_DIM, linear_conv_kernel_dim=4,
        vocab_size=VOCAB if head else 1000, rms_norm_eps=1e-6, max_position_embeddings=4096,
        attn_output_gate=True,
    ))
    targets = ["lm_head", "embed_tokens"] if head else None
    spec.apply_yaml_overrides(max_slots=2, max_rank=16, adapters=dict(paths), targets=targets)
    # pinned as the API server pins a serving config's adapters
    spec.adapter_digests = {name: adapter_digest(read_lora_file(path, convert_keys=spec.convert_keys))
                           for name, path in paths.items()}
    spec.backend = "triton"
    pool = LoraAdapterPool(spec, "cuda", dtype=torch.bfloat16, rank=rank, world_size=world)
    assert isinstance(pool.backend, TritonLoraBackend)
    return pool


def expected_delta(x, ref, layer, target):
    """fp32 per token: each slice's ``scaling * x @ A^T @ B^T``, zeros where the adapter lacks it."""
    parts = []
    for proj in TARGETS[target]:
        if (layer, proj) not in ref:
            parts.append(torch.zeros(x.shape[0], SHAPES[proj][0], device=x.device))
            continue
        a, b, scaling = ref[(layer, proj)]
        parts.append(scaling * (x.float() @ a.cuda().t()) @ b.cuda().t())
    return torch.cat(parts, dim=1)


def check(got, want, what):  # bf16 inputs and weights, plus a bf16 rounding of the output
    err = (got.float() - want).norm() / want.norm().clamp_min(1e-6)
    assert err < 3e-2, f"{what}: relative error {err:.3g}"


def eager_step(pool, rows, walk="prefill_text", ingest=True):
    """Admit and plan ``rows`` of (rid, adapter, tokens) as one eager step."""
    for rid, name, _ in rows if ingest else ():
        pool.ingest_request(rid, None if name is None else LoraReqConfig(adapter=name))
    step = ResourceStep(segments=tuple(Segment(rid, "main", n) for rid, _, n in rows))
    # idle as the engine sets it: no row names an adapter
    ctx = StepContext(request_ids=[rid for rid, *_ in rows], graph_walk=walk, slot=0, capture=False,
                      idle=all(name is None for _, name, _ in rows))
    assert pool.admit(step, ctx).ok
    pool.plan(step, ctx)
    return step, ctx


def capture(fn):
    """Warm ``fn`` on a side stream (Triton JIT and autotuning stay out of the graph), then capture it."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def bucket_step(pool, rows, lease, capturing=False):
    """A step of (rid, tokens) rows padded to the bucket as the graph runner pads a replay."""
    pad = [-(i + 1) for i in range(len(rows), BUCKET_BS)]
    rids = [rid for rid, _ in rows]
    step = ResourceStep(segments=tuple([Segment(r, "main", n) for r, n in rows] + [Segment(r, "main", 0) for r in pad]))
    ctx = StepContext(request_ids=rids, graph_walk="prefill_text", slot=lease.slot, capture=capturing,
                      slot_lease=lease)
    ctx.set_padded_rids([*rids, *pad])
    assert pool.admit(step, ctx).ok
    pool.plan(step, ctx)
    pool.commit(step, ctx)


def slot_layout(token_slots, slots):
    order = torch.argsort(token_slots, stable=True).to(torch.int32)
    indptr = torch.zeros(slots + 1, dtype=torch.int32, device=token_slots.device)
    indptr[1:] = torch.cumsum(torch.bincount(token_slots, minlength=slots), 0)
    return order, indptr


@pytest.mark.parametrize("spans", [[1, 1, 1, 1, 1], [7, 1, 64, 1, 300], [513]])
def test_bf16_deltas_against_fp32(adapters, spans):
    """Every target, decode to long prefill, mixed adapters; base rows left bitwise untouched."""
    paths, refs = adapters
    pool = make_pool(paths)
    names = ["r16", None, "r4", "r16", None][:len(spans)]
    eager_step(pool, [(f"q{i}", n, s) for i, (n, s) in enumerate(zip(names, spans, strict=True))])
    token_names = [n for n, s in zip(names, spans, strict=True) for _ in range(s)]
    g = torch.Generator(device="cuda").manual_seed(5)
    for target, layer in [("qkv_proj", 1), ("o_proj", 1), ("in_proj_fused", 0), ("out_proj", 0),
                          ("gate_up_proj", 0), ("down_proj", 0)]:
        t = pool.targets[target]
        assert t.out_features == sum(SHAPES[p][0] for p in TARGETS[target])  # no `a` padding on one GPU
        x = torch.randn(len(token_names), t.in_features, generator=g, **BF16)
        base = torch.randn(len(token_names), t.out_features, generator=g, **BF16)
        y = pool.layer_delta(target, layer)(x, base.clone())
        for row, name in enumerate(token_names):
            if name is None:
                assert torch.equal(y[row], base[row]), f"{target}: a base row changed"
            else:
                want = expected_delta(x[row:row + 1], refs[name], layer, target)[0]
                check(y[row].float() - base[row].float(), want, f"{target} row {row} ({name})")


@pytest.mark.parametrize("world", [2, 4])
def test_delta_net_in_proj_rank_shards(adapters, world):
    """Each rank's share of q, k, v, z, a, b, with zeros on the rows that pad ``a``."""
    paths, refs = adapters
    for rank in range(world):
        pool = make_pool(paths, rank=rank, world=world)
        eager_step(pool, [("t", "r16", 9)])
        x = torch.randn(9, HIDDEN, generator=torch.Generator(device="cuda").manual_seed(21 + rank), **BF16)
        width = pool.targets["in_proj_fused"].out_features
        y = pool.layer_delta("in_proj_fused", 0)(x, torch.zeros(9, width, **BF16))
        full = expected_delta(x, refs["r16"], 0, "in_proj_fused")
        k, v, h = KEY_DIM // world, VALUE_DIM // world, LIN_HEADS // world
        pad = -h % 16
        starts = [rank * k, KEY_DIM + rank * k, 2 * KEY_DIM + rank * v, 2 * KEY_DIM + VALUE_DIM + rank * v,
                  2 * KEY_DIM + 2 * VALUE_DIM + rank * h, 2 * KEY_DIM + 2 * VALUE_DIM + LIN_HEADS + rank * h]
        parts = [full[:, s:s + n] for s, n in zip(starts, (k, k, v, v, h, h), strict=True)]
        parts.insert(5, torch.zeros(9, pad, device="cuda"))
        want = torch.cat(parts, dim=1)
        assert want.shape[1] == width
        a_end = 2 * k + 2 * v + h
        assert not y[:, a_end:a_end + pad].any(), "the `a` padding rows got a delta"
        check(y, want, f"TP{world} rank {rank}")


@pytest.mark.parametrize("world", [1, 4])
def test_lm_head_rows_per_request_against_fp32(head_adapters, world):
    """One head row per request (prefill and decode rows); under TP4 each rank's vocab columns."""
    paths, refs = head_adapters
    rows = [("p0", "h16", 300), ("p1", None, 1), ("p2", "h4", 7), ("p3", "h16", 1)]
    hidden = torch.randn(len(rows), HIDDEN, generator=torch.Generator(device="cuda").manual_seed(3), **BF16)
    parts = []
    for rank in range(world):
        pool = make_pool(paths, head=True, rank=rank, world=world)
        eager_step(pool, rows)
        parts.append(pool.layer_delta("lm_head", 0)(hidden, torch.zeros(len(rows), VOCAB // world, **BF16)))
    got = torch.cat(parts, dim=1).float()
    for i, (_, name, _) in enumerate(rows):
        if name is None:
            assert not got[i].any(), "a base row's logits changed"
        else:
            check(got[i], expected_delta(hidden[i:i + 1], refs[name], 0, "lm_head")[0], f"row {i} ({name})")


@pytest.mark.parametrize("case", ["layers", "layers-compiled", "lm_head"])
def test_one_capture_serves_every_adapter_mix(adapters, head_adapters, case):
    """Deltas captured once on the zero adapter (as the runner captures), replayed for mixes that
    move adapters between slots (loads that evict, a batch of base rows): bitwise equal to the same
    rows run eagerly, bucket padding untouched, fp32-accurate. ``layers-compiled`` captures the
    Inductor-compiled function as the runner does; ``lm_head`` has one row per request."""
    head = case == "lm_head"
    paths, refs = head_adapters if head else adapters
    pool = make_pool(paths, head=head)
    lease = SlotLease(slot=0, bucket=BucketKey("prefill_text", BUCKET_BS, BUCKET_TOKENS))
    for i in range(BUCKET_BS):
        pool.ingest_capture_row(-(i + 1))
    bucket_step(pool, [(-(i + 1), BUCKET_TOKENS // BUCKET_BS) for i in range(BUCKET_BS)], lease, capturing=True)
    rows = BUCKET_BS if head else BUCKET_TOKENS
    io = [(pool.layer_delta(t, layer), t, layer, torch.zeros(rows, pool.targets[t].in_features, **BF16),
           torch.zeros(rows, pool.targets[t].out_features, **BF16))
          for t, layer in ([("lm_head", 0)] if head else [("qkv_proj", 1), ("down_proj", 0)])]

    def forward():
        return [delta(x, y.clone()) for delta, _, _, x, y in io]

    fn = torch.compile(forward, mode="max-autotune-no-cudagraphs", fullgraph=True) if "compiled" in case else forward
    graph, outs = capture(fn)
    a, b, c = ("h16", "h4", "h16") if head else ("r16", "r8", "r4")
    mixes = [
        [("m0", a, 9), ("m1", None, 1), ("m2", b, 20)],
        [("m3", c, 5), ("m4", a, 30)],  # with three adapters, loads evict
        [("m5", None, 3), ("m6", None, 7)],
        [("m7", b, 33), ("m8", c, 1), ("m9", None, 2)],  # ... and b comes back after its eviction
    ]
    g = torch.Generator(device="cuda").manual_seed(13)
    for mix in mixes:
        for rid, name, _ in mix:
            pool.ingest_request(rid, None if name is None else LoraReqConfig(adapter=name))
        bucket_step(pool, [(rid, n) for rid, _, n in mix], lease)
        spans = [1 if head else n for *_, n in mix]
        real = sum(spans)
        for *_, x, y in io:
            x.copy_(torch.randn(x.shape, generator=g, device="cuda").to(torch.bfloat16))
            y.copy_(torch.randn(y.shape, generator=g, device="cuda").to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        step, ctx = eager_step(pool, mix, ingest=False)
        eager = [delta(x[:real], y[:real].clone()) for delta, _, _, x, y in io]
        pool.commit(step, ctx)
        torch.cuda.synchronize()
        for out, want_eager, (_, target, layer, x, y) in zip(outs, eager, io, strict=True):
            assert torch.equal(out[:real], want_eager), f"{target}: replay differs from eager"
            assert torch.equal(out[real:], y[real:]), f"{target}: bucket padding changed"
            at = 0
            for (_, name, _), n in zip(mix, spans, strict=True):
                if name is None:
                    assert torch.equal(out[at:at + n], y[at:at + n]), f"{target}: base rows changed"
                else:
                    want = expected_delta(x[at:at + n], refs[name], layer, target)
                    check(out[at:at + n].float() - y[at:at + n].float(), want, f"{target} {name}")
                at += n
        for rid, *_ in mix:
            pool.remove_request(rid)


@pytest.mark.parametrize("captured", [False, True])
def test_an_fp32_input_under_autocast_takes_the_weights_dtype(adapters, captured):
    """An fp32 ``LayerNorm`` output under autocast (Qwen3.5's vision blocks): the base linear
    runs in bf16, and so must the delta, eagerly and inside a capture."""
    paths, refs = adapters
    pool = make_pool(paths)
    eager_step(pool, [("a", "r16", 4), ("b", None, 4)])
    norm = torch.nn.LayerNorm(HIDDEN, device="cuda")
    linear = torch.nn.Linear(HIDDEN, 2 * INTER, bias=False, device="cuda")
    delta = pool.layer_delta("gate_up_proj", 0)
    hidden = torch.randn(8, HIDDEN, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9))

    def forward():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = norm(hidden)
            assert x.dtype == torch.float32
            return delta(x, linear(x)), x

    if captured:
        graph, (out, x) = capture(forward)
        graph.replay()
    else:
        out, x = forward()
    torch.cuda.synchronize()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        base = linear(x)
    check(out[:4].float() - base[:4].float(), expected_delta(x[:4], refs["r16"], 0, "gate_up_proj"), "fp32 input")
    assert torch.equal(out[4:], base[4:])


@pytest.mark.parametrize("op,max_rank,in_features,slices,tokens", [
    ("apply", 16, 1024, (4096, 512, 512), 1), ("apply", 32, 3584, (1024,), 5),
    ("apply", 64, 1000, (700,), 33), ("apply", 96, 1024, (4096, 512, 512), 200),
    ("apply", 128, 3584, (1024,), 1500), ("apply", 128, 1000, (700,), 5),
    ("apply", 256, 1000, (700,), 1), ("apply", 256, 1024, (4096, 512, 512), 1500),
    ("apply", 512, 3584, (1024,), 33), ("apply", 512, 1024, (4096, 512, 512), 200),
    ("gather", 128, 5000, (1024,), 1), ("gather", 128, 5000, (1024,), 7), ("gather", 256, 5000, (1024,), 300),
])
def test_kernels_match_the_reference(op, max_rank, in_features, slices, tokens):
    """``lora_apply_`` (split contraction, compact row tiles, an input width off the tile) and the
    embedding path ``lora_gather_shrink`` + ``lora_expand_`` (ids off the vocabulary get nothing)
    against the fp32 reference ops, over slots of mixed ranks with a rank-0 slot and base rows."""
    gather = op == "gather"
    g = torch.Generator(device="cuda").manual_seed(tokens + 100 * gather)
    slots, out_features = 6, sum(slices)
    ranks = torch.tensor([0, max_rank, max_rank // 4, 0, max_rank // 2 + 3, max_rank], dtype=torch.int32, device="cuda")
    a = torch.randn(slots, len(slices) * max_rank, in_features, generator=g, device="cuda") / (
        1 if gather else in_features ** 0.5)
    b = torch.randn(slots, out_features, max_rank, generator=g, device="cuda") / max_rank ** 0.5
    scalings = torch.rand(slots, generator=g, device="cuda") + 0.5
    token_slots = torch.randint(0, slots, (tokens,), generator=g, device="cuda")
    order, indptr = slot_layout(token_slots, slots)
    addressing = (order, indptr, token_slots.to(torch.int32))
    y0 = torch.randn(tokens, out_features, generator=g, device="cuda")
    want, got = y0.clone(), y0.to(torch.bfloat16)
    untouched = (token_slots == 0) | (token_slots == 3)  # the zero adapter and the rank-0 slot
    if gather:
        ids = torch.randint(0, in_features, (tokens,), generator=g, device="cuda")
        ids[::5], ids[3::7] = -1, in_features + 11
        untouched |= (ids < 0) | (ids >= in_features)
        h_ref = ref_ops.gather_shrink_reference(ids, a, addressing[2], ranks)
        h = ops.lora_gather_shrink(ids, a.to(torch.bfloat16), addressing[2], ranks)
        ops.lora_expand_(got, h, b.to(torch.bfloat16), *addressing, ranks, scalings, list(slices))
    else:
        x = torch.randn(tokens, in_features, generator=g, device="cuda")
        h_ref = ref_ops.shrink_reference(x, a, addressing[2], ranks, len(slices))
        ops.lora_apply_(got, x.to(torch.bfloat16), a.to(torch.bfloat16), b.to(torch.bfloat16), *addressing,
                        ranks, scalings, list(slices))
    ref_ops.expand_reference_(want, h_ref, b, addressing[2], ranks, scalings, list(slices))
    check(got.float() - y0.to(torch.bfloat16).float(), want - y0, op)
    assert torch.equal(got[untouched], y0.to(torch.bfloat16)[untouched])


def test_a_tiled_rank_inside_a_captured_graph():
    """``lora_apply_`` at max_rank 128 captured once (every row on the zero adapter), replayed for
    slot mixes and ranks that change between replays: bitwise equal to eager."""
    g = torch.Generator(device="cuda").manual_seed(5)
    slots, max_rank, in_features, slices, tokens = 4, 128, 1024, [1024, 256, 256], 24
    a = (torch.randn(slots, len(slices) * max_rank, in_features, generator=g, device="cuda") / 32).to(torch.bfloat16)
    b = (torch.randn(slots, sum(slices), max_rank, generator=g, device="cuda") / 4).to(torch.bfloat16)
    scalings = torch.rand(slots, generator=g, device="cuda") + 0.5
    ranks = torch.zeros(slots, dtype=torch.int32, device="cuda")
    order, indptr = slot_layout(torch.zeros(tokens, dtype=torch.long, device="cuda"), slots)
    token_slot = torch.zeros(tokens, dtype=torch.int32, device="cuda")  # the Triton kernels read order / indptr
    x, y = torch.zeros(tokens, in_features, **BF16), torch.zeros(tokens, sum(slices), **BF16)

    def run():
        out = y.clone()
        ops.lora_apply_(out, x, a, b, order, indptr, token_slot, ranks, scalings, slices)
        return out

    graph, out = capture(run)
    for mix_ranks, seed in (([0, 128, 33, 64], 1), ([0, 16, 128, 100], 2), ([0, 0, 0, 0], 3)):
        gm = torch.Generator(device="cuda").manual_seed(seed)
        ranks.copy_(torch.tensor(mix_ranks, dtype=torch.int32))
        order[:], indptr[:] = slot_layout(torch.randint(0, slots, (tokens,), generator=gm, device="cuda"), slots)
        x.copy_(torch.randn(x.shape, generator=gm, device="cuda").to(torch.bfloat16))
        y.copy_(torch.randn(y.shape, generator=gm, device="cuda").to(torch.bfloat16))
        graph.replay()
        eager = run()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), f"replay differs from eager for ranks {mix_ranks}"
        if not any(mix_ranks):
            assert torch.equal(out, y)


@pytest.mark.parametrize("tokens,top_k,stage", [(1, 8, "gate_up"), (37, 8, "gate_up"), (300, 8, "gate_up"),
                                                 (5, 8, "down"), (300, 8, "down")])
def test_experts_kernels_match_the_reference(tokens, top_k, stage):
    """``apply_experts_`` on the fused-MoE grouped GEMM (one align over combined ids, a shrink, an
    accumulating expand per slice) at the Qwen3-Omni Thinker's TP1 shapes against the fp32 reference,
    over 7 slots of 128 experts with rows on no id (the zero adapter, a skipped expert)."""
    g = torch.Generator(device="cuda").manual_seed(tokens + top_k)
    hidden, inter, rank, num_ids = 2048, 768, 16, 7 * 128
    gate_up = stage == "gate_up"
    in_features, slices = (hidden, (inter, inter)) if gate_up else (inter, (hidden,))
    a = torch.randn(num_ids, len(slices) * rank, in_features, generator=g, device="cuda") / in_features ** 0.5
    b = torch.randn(num_ids, sum(slices), rank, generator=g, device="cuda") / rank ** 0.5
    ids = torch.randint(0, num_ids + 1, (tokens, top_k), generator=g, device="cuda", dtype=torch.int32)
    ids[::3, 0] = num_ids  # rows that get nothing
    weight = torch.rand(tokens, top_k, generator=g, device="cuda") + 0.5
    x = torch.randn(tokens if gate_up else tokens * top_k, in_features, generator=g, device="cuda")
    x_top_k = top_k if gate_up else 1
    y0 = torch.randn(tokens * top_k, sum(slices), generator=g, device="cuda")
    want = y0.clone()
    ref_ops.TorchLoraBackend().apply_experts_(x, want, a, b, ids, weight, x_top_k, slices)
    backend = TritonLoraBackend()
    got = y0.to(torch.bfloat16)
    alignment = backend.align_experts(ids, num_ids)
    backend.apply_experts_(x.to(torch.bfloat16), got, a.to(torch.bfloat16), b.to(torch.bfloat16), ids, weight,
                           x_top_k, slices, alignment)
    check(got.float() - y0.to(torch.bfloat16).float(), want - y0, f"experts {stage}")
    untouched = (ids.reshape(-1) == num_ids)
    assert torch.equal(got[untouched], y0.to(torch.bfloat16)[untouched])


# a small MoE layer: experts, hidden, intermediate, top-k, adapter rank
MOE_E, MOE_HIDDEN, MOE_INTER, MOE_TOP_K, MOE_RANK = 8, 256, 128, 2, 8


def write_experts_adapter(path, seed):
    """Per-expert PEFT factors on every expert's gate / up / down (alpha = rank, so scaling 1);
    returns each ``(expert, proj)``'s ``B @ A``."""
    g = torch.Generator().manual_seed(seed)
    sd, ref = {}, {}
    for e in range(MOE_E):
        for proj, (out, inp) in (("gate", (MOE_INTER, MOE_HIDDEN)), ("up", (MOE_INTER, MOE_HIDDEN)),
                                 ("down", (MOE_HIDDEN, MOE_INTER))):
            a, b = torch.randn(MOE_RANK, inp, generator=g) / inp ** 0.5, torch.randn(out, MOE_RANK, generator=g)
            module = f"base_model.model.layers.0.mlp.experts.{e}.{proj}_proj"
            sd[f"{module}.lora_A.weight"], sd[f"{module}.lora_B.weight"] = a, b
            sd[f"{module}.alpha"] = torch.tensor(float(MOE_RANK))
            ref[(e, proj)] = b @ a
    save_file(sd, str(path))
    return ref


def experts_layers(paths):
    """A bound ``FusedExperts`` on a two-slot Triton pool, and the same experts unbound."""
    import re

    from mstar.engine.resources.lora.config import LoraModuleRef, LoraSpec
    from mstar.model.components.lora.capable import make_lora_capable
    from mstar.model.components.moe import FusedExperts

    plain = FusedExperts(num_experts=MOE_E, hidden_size=MOE_HIDDEN, intermediate_size=MOE_INTER).to(**BF16)
    g = torch.Generator(device="cuda").manual_seed(5)
    for p in plain.parameters():
        p.data.copy_(torch.randn(p.shape, generator=g, device="cuda") / p.shape[-1] ** 0.5)

    def place(module):
        m = re.fullmatch(r"layers\.0\.mlp\.experts\.(\d+)\.(gate|up|down)_proj", module)
        if m is None:
            return None
        param = "down_proj" if m[2] == "down" else "gate_up_proj"
        return LoraModuleRef(f"experts_{param}", 0, f"{m[2]}:{m[1]}", param)

    spec = LoraSpec(
        resource_key="lora", nodes={"N"}, num_layers=1, module_map=place, max_slots=2, max_rank=MOE_RANK,
        targets={f"experts_{p}": t for p, t in plain.lora_targets().items()}, adapters=dict(paths), backend="triton",
    )
    spec.adapter_digests = {name: adapter_digest(read_lora_file(path)) for name, path in paths.items()}
    pool = LoraAdapterPool(spec, "cuda", dtype=torch.bfloat16)
    bound = FusedExperts(num_experts=MOE_E, hidden_size=MOE_HIDDEN, intermediate_size=MOE_INTER).to(**BF16)
    bound.load_state_dict(plain.state_dict())
    make_lora_capable(bound).lora_params = {
        p: pool.layer_delta(f"experts_{p}", 0) for p in ("gate_up_proj", "down_proj")
    }
    return pool, bound, plain


def merged_delta(plain, ref, x, weights, ids):
    """fp32: the experts' output with each expert's adapter merged into it, less the base's."""
    from mstar.model.components.moe import dispatch_experts_fused

    gate_up, down = plain.gate_up_proj.float(), plain.down_proj.float()
    merged_gate_up, merged_down = gate_up.clone(), down.clone()
    for e in range(MOE_E):
        merged_gate_up[e, :MOE_INTER] += ref[(e, "gate")].cuda()
        merged_gate_up[e, MOE_INTER:] += ref[(e, "up")].cuda()
        merged_down[e] += ref[(e, "down")].cuda()

    def run(w1, w2):
        return dispatch_experts_fused(x.float(), w1, w2, MOE_E, ids, weights.float())

    return run(merged_gate_up, merged_down) - run(gate_up, down)


def routed(tokens, g):
    x = torch.randn(tokens, MOE_HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    probs = torch.softmax(torch.randn(tokens, MOE_E, generator=g, device="cuda"), dim=-1)
    weights, ids = torch.topk(probs, MOE_TOP_K, dim=-1)
    return x, weights, ids


@pytest.mark.parametrize("captured", [False, True])
def test_bound_experts_match_each_rows_merged_adapter(tmp_path, captured):
    """A bound ``FusedExperts`` through ``fused_experts(lora=)`` (gate/up delta before the SwiGLU,
    routing-weighted down delta before the top-k sum), two adapters and base rows in one batch,
    against each row's adapter merged into its experts in fp32: eagerly, and inside one captured
    graph replayed for other adapter mixes. Base rows (and bucket padding) are bitwise the unbound
    layer's."""
    paths, refs = {}, {}
    for name, seed in (("e1", 1), ("e2", 2)):
        paths[name] = str(tmp_path / f"{name}.safetensors")
        refs[name] = write_experts_adapter(paths[name], seed)
    pool, bound, plain = experts_layers(paths)
    g = torch.Generator(device="cuda").manual_seed(17)

    def compare(mix, out, base, x, weights, ids):
        at = 0
        for _, name, n in mix:
            rows = slice(at, at + n)
            if name is None:
                assert torch.equal(out[rows], base[rows]), "base rows changed"
            else:
                want = merged_delta(plain, refs[name], x[rows], weights[rows], ids[rows])
                check(out[rows].float() - base[rows].float(), want, f"experts {name}")
            at += n

    mixes = [[("m0", "e1", 9), ("m1", None, 1), ("m2", "e2", 20)], [("m3", "e2", 5), ("m4", None, 30)]]
    if not captured:
        for mix in mixes:
            step, ctx = eager_step(pool, mix)
            x, weights, ids = routed(sum(n for *_, n in mix), g)
            with torch.no_grad():
                out, base = bound(x, weights, ids), plain(x, weights, ids)
            pool.commit(step, ctx)
            compare(mix, out, base, x, weights, ids)
            for rid, *_ in mix:
                pool.remove_request(rid)
        return
    lease = SlotLease(slot=0, bucket=BucketKey("prefill_text", BUCKET_BS, BUCKET_TOKENS))
    for i in range(BUCKET_BS):
        pool.ingest_capture_row(-(i + 1))
    bucket_step(pool, [(-(i + 1), BUCKET_TOKENS // BUCKET_BS) for i in range(BUCKET_BS)], lease, capturing=True)
    x, weights, ids = routed(BUCKET_TOKENS, g)

    def forward():
        with torch.no_grad():
            return bound(x, weights, ids), plain(x, weights, ids)

    graph, (out, base) = capture(forward)
    for mix in mixes:
        for rid, name, _ in mix:
            pool.ingest_request(rid, None if name is None else LoraReqConfig(adapter=name))
        bucket_step(pool, [(rid, n) for rid, _, n in mix], lease)
        for static, fresh in zip((x, weights, ids), routed(BUCKET_TOKENS, g), strict=True):
            static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        real = sum(n for *_, n in mix)
        compare(mix, out, base, x, weights, ids)
        assert torch.equal(out[real:], base[real:]), "bucket padding took a delta"
        for rid, *_ in mix:
            pool.remove_request(rid)
