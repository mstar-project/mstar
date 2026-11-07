"""The 4-bit formats on CPU: AutoAWQ, GPTQ and compressed-tensors int4 repacked into one layout bit for bit, MXFP4 and
modelopt NVFP4 codes and scale layouts exact, every layout loaded through the engine's path into the tiny TP root
(each rank's packed weights, scales and zero points the slices of the unsharded ones at TP 1/2/4), mixed layers
dequantized, on-the-fly quantization equal to a stored checkpoint, and the refusals. The references here pack and
dequantize on their own (strided slices, not the format's helpers). CUDA cases compare the kernels with the
reference on the GPU they run on."""

import json
import re

import pytest
import torch
import torch.nn.functional as F
from quant_helpers import EXPERTS, HEAD_DIM, HEADS, HIDDEN, INTER, KV_HEADS, LAYERS, full_weights, load
from safetensors.torch import save_file

from mstar.model.components import int4_formats as I
from mstar.model.components.block_scaled import NVFP4
from mstar.quantization.checkpoint import StoredBase, scan_checkpoint
from mstar.quantization.resolve import Action
from mstar.utils.fp8_block import FP4_DTYPE, FP8_DTYPE, unswizzle_block_scales

WEIGHTS = full_weights()
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def _threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


# ---------------------------------------------------------------- references, packed by hand
def words(v: torch.Tensor) -> torch.Tensor:
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def pack_k(u):  # [R, C] nibbles -> [R, C/8], value c at word c // 8, nibble c % 8 (compressed-tensors, GPTQ zeros)
    v = u.to(torch.int64)
    return words(sum(v[:, j::8] << (4 * j) for j in range(8)))


def pack_rows(u):  # [R, C] -> [R/8, C] along rows (GPTQ qweight, compressed-tensors zero points)
    return pack_k(u.t()).t().contiguous()


def pack_awq(u):  # [R, C] -> [R, C/8] in AutoAWQ's order: nibble i of word j holds column 8j + (0,2,4,6,1,3,5,7)[i]
    v = u.to(torch.int64)
    return words(sum(v[:, c::8] << (4 * i) for i, c in enumerate((0, 2, 4, 6, 1, 3, 5, 7))))


def rtn(w: torch.Tensor, group: int, symmetric: bool, scale_dtype=torch.float16):
    """Round-to-nearest int4 of ``w [N, K]``: unsigned values u, zero points z ([N, K/g]), scales s."""
    n, k = w.shape
    g = w.float().view(n, k // group, group)
    if symmetric:
        s = (g.abs().amax(-1) / 7.5).clamp_min(1e-8).to(scale_dtype)
        z = torch.full_like(s, 8, dtype=torch.int64)
    else:
        lo, hi = g.amin(-1), g.amax(-1)
        s = ((hi - lo) / 15).clamp_min(1e-8).to(scale_dtype)
        z = torch.round(-lo / s.float()).clamp(0, 15).to(torch.int64)
    u = (torch.round(g / s.float()[..., None]) + z[..., None]).clamp(0, 15).to(torch.int64).view(n, k)
    return u, z, s


def deq(u, z, s, group):
    n, k = u.shape
    return ((u.view(n, -1, group) - z[..., None]).float() * s.float()[..., None]).view(n, k)


def layout_tensors(w, layout, group, tensor_scale=None):
    """One linear's checkpoint tensors in ``layout`` (by suffix) and the fp32 weight they stand for (NVFP4: in units
    of ``tensor_scale``, shared by the shards of a fused layer as modelopt calibrates them)."""
    if layout == "mxfp4":
        q = I.MXFP4.quantize_storage(w)
        codes = torch.stack([q["weight_packed"] & 15, q["weight_packed"] >> 4], -1).view(w.shape)
        table = torch.tensor(I.E2M1_VALUES)
        values = torch.where(codes & 8 != 0, -table[(codes & 7).long()], table[(codes & 7).long()])
        ref = (values.view(w.shape[0], -1, 32) * torch.exp2(q["weight_scale"].float() - 127)[..., None]).view(w.shape)
        return {"weight_packed": q["weight_packed"], "weight_scale": q["weight_scale"]}, ref
    if layout == "nvfp4":
        q = NVFP4.quantize(w, tensor_scale)
        n, k = w.shape
        linear_scale = unswizzle_block_scales(q.scale.view(torch.uint8), n, k // 16).contiguous()
        out = {"weight": q.data.view(torch.uint8), "weight_scale": linear_scale.view(FP8_DTYPE),
               "weight_scale_2": q.tensor_scale.reshape(()), "input_scale": torch.tensor(0.5)}
        return out, NVFP4.dequantize(q)
    kind, sym = layout.split("-")[0], layout.endswith("sym") and not layout.endswith("asym")
    scale_dtype = torch.bfloat16 if kind == "ct" else torch.float16
    u, z, s = rtn(w, group, sym, scale_dtype)
    ref = deq(u, z, s, group)
    if kind == "ct":
        out = {"weight_packed": pack_k(u), "weight_scale": s, "weight_shape": torch.tensor(list(w.shape))}
        if not sym:
            out["weight_zero_point"] = pack_rows(z)
        return out, ref
    if kind == "awq":
        return {"qweight": pack_awq(u.t()), "qzeros": pack_awq(z.t()), "scales": s.t().contiguous()}, ref
    # gptq: v1 stores each zero point minus one, gptq2 as is
    offset = 1 if kind == "gptq" else 0
    return {"qweight": pack_rows(u.t()), "qzeros": pack_k((z.t() - offset) & 15), "scales": s.t().contiguous(),
            "g_idx": torch.arange(w.shape[1], dtype=torch.int32) // group}, ref


QUANT_CONFIG = {
    "ct": {"quant_method": "compressed-tensors", "format": "pack-quantized"},
    "awq": {"quant_method": "awq", "bits": 4, "version": "gemm", "zero_point": True},
    "gptq": {"quant_method": "gptq", "bits": 4, "desc_act": False},
    "gptq2": {"quant_method": "gptq", "bits": 4, "desc_act": False, "checkpoint_format": "gptq_v2"},
    "mxfp4": {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized"},
    "nvfp4": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
}


def source_of(layout, group):
    kind = layout.split("-")[0]
    return StoredBase("w4a16", "compressed-tensors" if kind == "ct" else "gptq" if kind.startswith("gptq") else kind,
                      (1, group)).with_params(
        group=group, symmetric=layout.endswith("-sym"), scale_dtype="BF16" if kind == "ct" else "F16",
        **({"zero_offset": int(kind == "gptq")} if kind.startswith("gptq") else {}))


def _fused_group(base):
    """The bases a fused layer loads together (q/k/v, gate/up): modelopt gives them one tensor scale."""
    for members in (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")):
        for m in members:
            if base.endswith("." + m):
                return [base[: -len(m)] + other for other in members]
    return [base]


def write_layout(root, layout, group, pick=lambda base: base != "lm_head" and not base.endswith("moe.gate"),
                 shared_tensor_scale=True):
    """The tiny root's checkpoint with the linears ``pick`` selects stored in ``layout``; returns ``(path, refs)``,
    refs = the fp32 weight each stored base stands for."""
    root.mkdir(parents=True, exist_ok=True)
    tensors, refs = {}, {}
    for name, w in WEIGHTS.items():
        base = name.removesuffix(".weight")
        if not pick(base):
            tensors[name] = w
            continue
        scale = None
        if layout == "nvfp4" and shared_tensor_scale:
            amax = max(WEIGHTS[b + ".weight"].float().abs().max() for b in _fused_group(base))
            scale = amax / (6.0 * 448.0)
        parts, refs[base] = layout_tensors(w, layout, group, scale)
        tensors.update({f"{base}.{s}": t.contiguous() for s, t in parts.items()})
    save_file(tensors, str(root / "model.safetensors"))
    config = dict(QUANT_CONFIG[layout.split("-")[0]])
    if layout.startswith("gptq"):
        config["sym"] = layout.endswith("-sym")
    (root / "config.json").write_text(json.dumps({"quantization_config": config}))
    return root, refs


LAYOUTS = [("ct-sym", 32), ("ct-asym", 32), ("awq-asym", 32), ("gptq-sym", 128), ("gptq-asym", 32),
           ("gptq2-asym", 32), ("mxfp4", 32), ("nvfp4", 16)]


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    root = tmp_path_factory.mktemp("int4")
    return {(layout, g): write_layout(root / f"{layout}{g}", layout, g) for layout, g in LAYOUTS}


# ---------------------------------------------------------------- formats: packing and repacking, exact
def test_int4_words_unpack_and_pack_along_either_dim():
    u = torch.randint(0, 16, (24, 64))
    assert torch.equal(I.unpack_int4(pack_k(u)), u.to(torch.uint8))
    assert torch.equal(I.unpack_int4(pack_rows(u), dim=0), u.to(torch.uint8))
    assert torch.equal(I.pack_int4(u), pack_k(u)) and torch.equal(I.pack_int4(u, dim=0), pack_rows(u))
    assert torch.equal(I.unpack_awq(pack_awq(u)), u.to(torch.uint8)) and torch.equal(I.pack_awq(u), pack_awq(u))
    # an int32 word is four bytes, low nibble first: the uint8 view the kernels read
    assert torch.equal(I.unpack_fp4(pack_k(u).view(torch.uint8)), u.to(torch.uint8))


@pytest.mark.parametrize("layout, group", [l for l in LAYOUTS if l[0] not in ("mxfp4", "nvfp4")])
def test_awq_gptq_and_ct_repack_to_one_layout_bit_for_bit(layout, group):
    torch.manual_seed(1)
    w = torch.randn(96, 256) * 0.05
    tensors, ref = layout_tensors(w, layout, group)
    fmt = I.W4A16.for_stored(source_of(layout, group))
    buffers = {}
    for suffix, tensor in tensors.items():
        adapted = fmt.adapt_checkpoint(suffix, tensor)
        if adapted is not None:
            buffers[adapted[0]] = fmt.from_checkpoint(*adapted)
    assert set(buffers) == {"weight_packed", "weight_scale"} | ({"weight_zero_point"} if "asym" in layout else set())
    assert torch.equal(fmt.dequantize(buffers), ref)
    # the canonical bytes: compressed-tensors' own packing of the same values
    u, z, _ = rtn(w, group, layout.endswith("-sym"), fmt.scale_dtype)
    assert torch.equal(buffers["weight_packed"], pack_k(u))
    if "asym" in layout:
        assert torch.equal(buffers["weight_zero_point"], pack_rows(z))


def test_gptq_act_order_and_inconsistent_zeros_are_refused(tmp_path):
    fmt = I.W4A16.for_stored(source_of("gptq-sym", 32))
    with pytest.raises(ValueError, match="act-order"):
        fmt.adapt_checkpoint("g_idx", torch.randperm(256).to(torch.int32) // 32)
    with pytest.raises(ValueError, match="zero points other than 8"):
        fmt.adapt_checkpoint("qzeros", pack_k(torch.full((8, 96), 5)))
    # at boot: a desc_act checkpoint's g_idx is read (a few KB) and a permuted one refuses its layer
    for permuted in (False, True):
        path, _ = write_layout(tmp_path / f"act{permuted}", "gptq-sym", 32,
                               pick=lambda base: base == "layers.0.mlp.gate_proj")
        from safetensors.torch import load_file
        tensors = load_file(str(path / "model.safetensors"))
        if permuted:
            tensors["layers.0.mlp.gate_proj.g_idx"] = (torch.randperm(HIDDEN) // 32).to(torch.int32)
        save_file(tensors, str(path / "model.safetensors"))
        config = json.loads((path / "config.json").read_text())
        config["quantization_config"]["desc_act"] = True
        (path / "config.json").write_text(json.dumps(config))
        stored = scan_checkpoint(path).bases["layers.0.mlp.gate_proj"]
        assert bool(stored.param("act_order")) is permuted
        if permuted:
            with pytest.raises(ValueError, match="act-order"):
                load(path)
        else:
            assert load(path)[1].layers["layers.0.mlp.gate_up_proj"].action is Action.DEQUANT  # up stays bf16


def test_mxfp4_codes_and_e8m0_scales_are_exact():
    torch.manual_seed(2)
    codes = torch.randint(0, 16, (64, 128), dtype=torch.uint8)
    scales = torch.randint(110, 140, (64, 4), dtype=torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    table = torch.tensor(I.E2M1_VALUES)
    values = torch.where(codes & 8 != 0, -table[(codes & 7).long()], table[(codes & 7).long()])
    ref = (values.view(64, 4, 32) * torch.exp2(scales.float() - 127)[..., None]).view(64, 128)
    assert torch.equal(I.MXFP4.dequantize({"weight_packed": packed, "weight_scale": scales}), ref)
    # quantizing what an MXFP4 weight stands for gives back the same values (no clipping, no drift)
    again = I.MXFP4.quantize_weight(ref, None)
    assert torch.equal(I.MXFP4.dequantize(again), ref)
    # a scale byte is 2 ** (s - 127): Kimi K3's e8m0 storage
    assert I.MXFP4.dequantize({"weight_packed": torch.full((1, 16), 0x22, dtype=torch.uint8),
                               "weight_scale": torch.tensor([[127]], dtype=torch.uint8)}).unique().tolist() == [1.0]


def test_nvfp4_modelopt_layout_meets_the_on_the_fly_buffers():
    torch.manual_seed(3)
    w = (torch.randn(160, 256) * 0.05).to(torch.bfloat16)
    tensors, ref = layout_tensors(w, "nvfp4", 16)
    storage = {st.name: NVFP4.from_checkpoint(st.name, tensors[st.name]) for st in NVFP4.storage(160, 256)}
    views = {"weight": storage["weight"].view(FP4_DTYPE), "weight_scale": storage["weight_scale"].view(FP8_DTYPE),
             "weight_scale_2": storage["weight_scale_2"].view(torch.float32),
             "input_scale": storage["input_scale"].view(torch.float32)}
    done = NVFP4.after_load(views)
    fly = NVFP4.quantize_weight(w, None)
    assert torch.equal(done["weight"].view(torch.uint8), fly["weight"].view(torch.uint8))
    assert torch.equal(done["weight_scale"].view(torch.uint8), fly["weight_scale"].view(torch.uint8))
    assert torch.equal(done["weight_scale_2"], fly["weight_scale_2"])
    assert torch.equal(NVFP4.dequantize(tensors), ref) and torch.equal(NVFP4.dequantize(done), ref)
    # swizzled rows: 128-row tiles; a [160, 16] scale pads to [256, 16]
    assert done["weight_scale"].numel() == 256 * 16
    # one GEMM takes one tensor scale: shards that disagree are refused
    with pytest.raises(ValueError, match="different weight_scale_2"):
        NVFP4.after_load({**views, "weight_scale_2": torch.tensor([1.0, 2.0])})


# ---------------------------------------------------------------- loading through the engine's path
def _linear_weight(path, root):
    """``path``'s weight as the module now holds it, dequantized to fp32."""
    module = root.get_submodule(path)
    fmt = module.quant_format
    buffers = {}
    for name, dtype in module._quant_views.items():
        t = getattr(module, name).data
        buffers[name] = t if dtype is None or t.dtype == dtype else t.view(dtype)
    return fmt.dequantize(buffers)


def _expert_weights(block):
    return [torch.stack([block.experts_packed.dequantize(block._expert_buffers(p, i)) if block.experts_packed.name !=
                         "nvfp4" else NVFP4.dequantize(_nvfp4_expert(block, p, i)) for i in range(EXPERTS)])
            for p in ("gate_up_proj", "down_proj")]


def _nvfp4_expert(block, proj, i):
    b = block._expert_buffers(proj, i)
    return {**b, "weight_scale_2": b["weight_scale_2"][:1]}


def _qkv_rows(ref_q, ref_k, ref_v, rank, tp):
    kv_chunks = min(tp, KV_HEADS)
    kv = rank * kv_chunks // tp
    return torch.cat([ref_q.chunk(tp)[rank], ref_k.chunk(kv_chunks)[kv], ref_v.chunk(kv_chunks)[kv]])


def _expected(refs, layer, kind, rank, tp):
    p = f"layers.{layer}"
    if kind == "qkv":
        return _qkv_rows(*(refs[f"{p}.self_attn.{x}_proj"] for x in "qkv"), rank, tp)
    if kind == "o":
        return refs[f"{p}.self_attn.o_proj"].chunk(tp, dim=1)[rank]
    if kind == "gate_up":
        return torch.cat([refs[f"{p}.mlp.gate_proj"].chunk(tp)[rank], refs[f"{p}.mlp.up_proj"].chunk(tp)[rank]])
    return refs[f"{p}.mlp.down_proj"].chunk(tp, dim=1)[rank]


PATHS = {"qkv": "self_attn.qkv_proj", "o": "self_attn.o_proj", "gate_up": "mlp.gate_up_proj", "down": "mlp.down_proj"}


@pytest.mark.parametrize("layout, group", LAYOUTS)
def test_every_layout_loads_as_stored_and_runs_the_weights_it_stands_for(stored, layout, group):
    path, refs = stored[(layout, group)]
    root, plan = load(path)
    assert {l.action for l in plan.layers.values()} == {Action.DIRECT, Action.KEEP}, plan.summary()
    fmt_name = "w4a16" if layout not in ("mxfp4", "nvfp4") else layout
    torch.manual_seed(4)
    x = torch.randn(5, 512, dtype=torch.bfloat16)
    for layer in range(LAYERS):
        for kind, rel in PATHS.items():
            module = root.get_submodule(f"layers.{layer}.{rel}")
            assert module.quant_format.name == fmt_name
            want = _expected(refs, layer, kind, 0, 1)
            assert torch.equal(_linear_weight(f"layers.{layer}.{rel}", root), want), (layer, kind)
            xi = x[:, :want.shape[1]].contiguous()
            got = module._quant_local(xi, None)
            if layout == "nvfp4":
                ref = NVFP4.reference_linear(NVFP4.quantize(xi, torch.tensor([0.5])),
                                             NVFP4._as_blockquant(_buffers(module)))
            else:
                ref = F.linear(xi, want.to(torch.bfloat16))
            assert torch.equal(got, ref), (layer, kind)
        block = root.layers[layer].moe
        assert block.experts_packed.name == fmt_name and block.experts_quantized
        up, down = _expert_weights(block)
        for e in range(EXPERTS):
            p = f"layers.{layer}.moe.experts.{e}"
            assert torch.equal(up[e], torch.cat([refs[f"{p}.gate_proj"], refs[f"{p}.up_proj"]]))
            assert torch.equal(down[e], refs[f"{p}.down_proj"])
    # the experts' forward (the CPU reference) against the bf16 dispatch over the dequantized experts
    from mstar.model.components.moe import dispatch_experts_fused

    block = root.layers[0].moe
    up, down = _expert_weights(block)
    h = torch.randn(7, HIDDEN, dtype=torch.bfloat16)
    weights, ids, _ = block.gate(h)
    got = block._experts_quant(h, weights, ids)
    assert torch.isfinite(got.float()).all()
    if layout != "nvfp4":  # NVFP4 also quantizes the activations; its reference is the kernels' yardstick
        want = dispatch_experts_fused(h, up.to(torch.bfloat16), down.to(torch.bfloat16), EXPERTS, ids, weights)
        torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=1e-2)


def _buffers(module):
    out = {}
    for name, dtype in module._quant_views.items():
        t = getattr(module, name).data
        out[name] = t if dtype is None or t.dtype == dtype else t.view(dtype)
    return out


TP_LAYOUTS = [("ct-sym", 32, 4), ("ct-asym", 32, 4), ("awq-asym", 32, 4), ("gptq-sym", 128, 2),
              ("gptq-asym", 32, 4), ("mxfp4", 32, 4), ("nvfp4", 16, 4), ("ct-asym", 32, 2)]


@pytest.mark.parametrize("layout, group, tp", TP_LAYOUTS)
def test_each_rank_holds_the_slices_of_the_unsharded_tensors(stored, layout, group, tp):
    path, refs = stored[(layout, group)]
    full, _ = load(path)
    # the dense 4-bit kernels take K in 128-column blocks: down's 64 a rank at TP 4 serves bf16, said at boot
    narrow = layout != "nvfp4" and (INTER // tp) % 128 != 0
    for rank in range(tp):
        root, plan = load(path, None, rank, tp)
        bf16 = {name for name, l in plan.layers.items() if l.action is Action.DEQUANT}
        assert bf16 == ({f"layers.{i}.mlp.down_proj" for i in range(LAYERS)} if narrow else set()), plan.notes
        assert not narrow or all("K in blocks of 128" in note for note in plan.notes), plan.notes
        assert {l.action for l in plan.layers.values()} - {Action.DEQUANT} == {Action.DIRECT, Action.KEEP}
        for layer in range(LAYERS):
            for kind, rel in PATHS.items():
                if narrow and kind == "down":
                    continue
                got = _linear_weight(f"layers.{layer}.{rel}", root)
                assert torch.equal(got, _expected(refs, layer, kind, rank, tp)), (rank, layer, kind)
            # raw storage too: column-parallel rows / row-parallel columns of rank 0's full tensors
            if layout not in ("nvfp4",):
                gu, fgu = root.layers[layer].mlp.gate_up_proj, full.layers[layer].mlp.gate_up_proj
                rows = INTER // tp
                # (the planes layout the kernels read shuffles within a row)
                for name in ("weight_planes", "weight_scale"):
                    a, b = getattr(gu, name), getattr(fgu, name)
                    assert torch.equal(a, torch.cat([b[:INTER][rank * rows:(rank + 1) * rows],
                                                     b[INTER:][rank * rows:(rank + 1) * rows]])), name
                if layout.endswith("-asym"):
                    zp, fzp = gu.weight_zero_point, fgu.weight_zero_point
                    r8 = rows // 8
                    assert torch.equal(zp, torch.cat([fzp[:INTER // 8][rank * r8:(rank + 1) * r8],
                                                      fzp[INTER // 8:][rank * r8:(rank + 1) * r8]]))
            block, fblock = root.layers[layer].moe, full.layers[layer].moe
            up, down = _expert_weights(block)
            fup, fdown = _expert_weights(fblock)
            inter = INTER // tp
            for e in range(EXPERTS):
                want_up = torch.cat([fup[e][:INTER][rank * inter:(rank + 1) * inter],
                                     fup[e][INTER:][rank * inter:(rank + 1) * inter]])
                assert torch.equal(up[e], want_up), (rank, layer, e)
                assert torch.equal(down[e], fdown[e][:, rank * inter:(rank + 1) * inter]), (rank, layer, e)


def test_a_fused_layer_with_int4_and_bf16_shards_is_dequantized(tmp_path):
    # Qwen3.5 AWQ's delta-net input: qkv and z int4, a and b bf16; here q bf16, k and v int4
    path, refs = write_layout(tmp_path / "mixed", "awq-asym", 32,
                              pick=lambda b: b.endswith(("k_proj", "v_proj")) and b.startswith("layers.0"))
    root, plan = load(path)
    layer = plan.layers["layers.0.self_attn.qkv_proj"]
    assert (layer.stored, layer.action) == ("mixed", Action.DEQUANT)
    qkv = root.layers[0].self_attn.qkv_proj.weight
    want = torch.cat([WEIGHTS["layers.0.self_attn.q_proj.weight"].float(),
                      refs["layers.0.self_attn.k_proj"], refs["layers.0.self_attn.v_proj"]]).to(torch.bfloat16)
    assert qkv.dtype == torch.bfloat16 and torch.equal(qkv, want)


@pytest.mark.parametrize("fmt, tp", [("w4a16", 1), ("w4a16", 2), ("mxfp4", 1), ("mxfp4", 2)])
def test_quantized_at_load_equals_the_same_weights_stored(tmp_path, fmt, tp):
    """A bf16 checkpoint quantized at load and the same quantization written as a checkpoint load into the same
    tensors on every rank (the stored one in the canonical layout, ``quantize_weight``'s own)."""
    from quant_helpers import write_checkpoint

    bf16 = write_checkpoint(tmp_path / "bf16", WEIGHTS)
    stored_dir = tmp_path / fmt
    stored_dir.mkdir()
    tensors = {}
    fmt_ = I.W4A16 if fmt == "w4a16" else I.MXFP4
    for name, w in WEIGHTS.items():
        base = name.removesuffix(".weight")
        if base == "lm_head" or base.endswith("moe.gate"):
            tensors[name] = w
            continue
        stored = {st.name for st in fmt_.storage(*w.shape)}
        tensors.update({f"{base}.{s}": t for s, t in fmt_.quantize_storage(w).items() if s in stored})
    save_file(tensors, str(stored_dir / "model.safetensors"))
    method = {"w4a16": QUANT_CONFIG["ct"], "mxfp4": QUANT_CONFIG["mxfp4"]}[fmt]
    (stored_dir / "config.json").write_text(json.dumps({"quantization_config": method}))
    for rank in range(tp):
        fly, plan = load(bf16, fmt, rank, tp)
        assert {l.action for l in plan.layers.values()} == {Action.QUANTIZE, Action.KEEP}
        direct, plan = load(stored_dir, None, rank, tp)
        assert {l.action for l in plan.layers.values()} == {Action.DIRECT, Action.KEEP}
        a = {n: t for n, t in fly.state_dict().items() if t.dtype != torch.bfloat16}
        b = {n: t for n, t in direct.state_dict().items() if t.dtype != torch.bfloat16}
        assert a.keys() == b.keys() and len(a) > 0
        for name, tensor in a.items():
            assert torch.equal(tensor, b[name]), (rank, name)
        # the node's dtype cast leaves the integer containers alone
        before = {n: t.clone() for n, t in a.items()}
        fly.to(torch.bfloat16)
        assert all(torch.equal(before[n], t) for n, t in fly.state_dict().items() if n in before)


def test_refusals(stored, tmp_path):
    from quant_helpers import declaration

    from mstar.quantization.resolve import resolve_plan
    from mstar.quantization.spec import parse_quantization

    def plan(path, quantization=None, capability=(9, 0), tp=1):
        from quant_helpers import build
        root = build(0, tp)
        return resolve_plan(root, declaration(path), parse_quantization(quantization).for_node("LLM"),
                            scan_checkpoint(path), capability)

    # NVFP4 has no GEMM before sm_100: a stored checkpoint is refused on an H200, not served as bytes
    path, _ = stored[("nvfp4", 16)]
    assert any("needs compute capability 10.0" in p for p in plan(path).problems)
    assert plan(path, capability=(10, 0)).problems == []
    # NVFP4 experts load as stored only
    from quant_helpers import write_checkpoint
    bf16 = write_checkpoint(tmp_path / "bf16", WEIGHTS)
    rule = {"layers": [{"match": "layers.*.moe.experts", "format": "nvfp4"}]}
    assert any("nvfp4 experts as a checkpoint stores them" in p for p in plan(bf16, rule, (10, 0)).problems)
    # a group that straddles a rank's K is refused when asked for explicitly
    path, _ = stored[("gptq-sym", 128)]
    asked = {"layers": [{"match": "layers.*.mlp.down_proj", "format": "w4a16"}]}
    assert any("whole groups of 128" in p for p in plan(path, asked, tp=4).problems)
    # ... and left bf16 by default
    by_default = plan(path, tp=4)
    assert by_default.problems == [] and by_default.layers["layers.0.mlp.down_proj"].action is Action.DEQUANT


def test_nvfp4_shards_with_different_tensor_scales_are_refused_at_load(tmp_path):
    path, _ = write_layout(tmp_path / "unshared", "nvfp4", 16, shared_tensor_scale=False)
    with pytest.raises(ValueError, match="different weight_scale_2"):
        load(path)


def test_an_int4_scale_that_never_arrives_is_refused(stored):
    from types import SimpleNamespace

    from quant_helpers import STACKED, build, declaration, remap

    from mstar.model.loader import iter_safetensors_shards, load_weights_into
    from mstar.quantization.scope import node_scope

    path, _ = stored[("ct-sym", 32)]
    model = SimpleNamespace(get_quantization_roots=lambda: [declaration(path)])
    lost = "layers.1.mlp.up_proj.weight_scale"
    with pytest.raises(ValueError, match="some of its scales never arrived"):
        with node_scope(model, "LLM", {}) as scope:
            root = build()
            load_weights_into(root, ((n, t) for n, t in iter_safetensors_shards(path) if n != lost),
                              stacked_params=STACKED, name_remapper=remap)
            scope.finish(root)


# ---------------------------------------------------------------- the kernels compile for Hopper (no GPU needed)
def _compile_cases():
    from mstar.utils import w4a16
    from mstar.utils.fused_moe import w4a16 as moe

    ptrs = {"x_ptr": "*bf16", "w_ptr": "*i32", "s_ptr": "*bf16", "z_ptr": "*i32", "out_ptr": "*bf16"}
    dims = {name: "i32" for name in ("M", "N", "K", "stride_xm", "stride_wn", "stride_sn", "stride_zn", "stride_os",
                                       "stride_om")}
    dense = {**ptrs, **dims}
    yield "dot", w4a16._w4_dot_kernel, dense, dict(GROUP=128, BN=64, TB=16, NB=2, SPLIT_K=1, STAGES=2)
    yield "dot_g32", w4a16._w4_dot_kernel, dense, dict(GROUP=32, BN=128, TB=32, NB=1, SPLIT_K=2, STAGES=3)
    yield "dot8", w4a16._w4_dot8_kernel, dense, dict(GROUP=32, BN=64, TB=16, NB=1, SPLIT_K=2, STAGES=2)
    yield "dot8_g128", w4a16._w4_dot8_kernel, dense, dict(GROUP=128, BN=64, TB=64, NB=2, SPLIT_K=1, STAGES=3)
    yield "gemv", w4a16._w4_gemv_kernel, dense, dict(GROUP=128, BN=32, NB=4, SPLIT_K=1, MB=1)
    yield "gemv_g32", w4a16._w4_gemv_kernel, dense, dict(GROUP=32, BN=16, NB=2, SPLIT_K=2, MB=4)
    yield "marlin_dequant", w4a16._marlin_dequant_kernel, {
        "w_ptr": "*i32", "s_ptr": "*bf16", "z_ptr": "*i32", "wt_ptr": "*i32", "st_ptr": "*i32", "zt_ptr": "*i32",
        "out_ptr": "*bf16", **{name: "i32" for name in ("stride_w", "stride_s", "stride_z", "stride_on")}}, dict(
        GROUP=128, BN=64, BK=128)
    yield "dequant", w4a16._w4_dequant_kernel, {
        "w_ptr": "*i32", "s_ptr": "*bf16", "z_ptr": "*i32", "out_ptr": "*bf16",
        **{name: "i32" for name in ("N", "K", "stride_wn", "stride_sn", "stride_zn", "stride_on")}}, dict(
        GROUP=32, BN=32)
    yield "experts", moe.fused_moe_w4a16_kernel, {
        "a_ptr": "*bf16", "b_ptr": "*i32", "s_ptr": "*bf16", "z_ptr": "*i32", "c_ptr": "*bf16",
        "topk_weights_ptr": "*fp32", "sorted_token_ids_ptr": "*i32", "expert_ids_ptr": "*i32",
        "num_tokens_post_padded_ptr": "*i32",
        **{name: "i32" for name in ("N", "K", "EM", "num_valid_tokens", "stride_am", "stride_be", "stride_bn",
                                    "stride_se", "stride_sn", "stride_ze", "stride_zn", "stride_cm")}}, dict(
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, SB=128, GROUP_SIZE_M=1, MUL_ROUTED_WEIGHT=True, top_k=1,
        compute_type=None, GROUP=128, STAGES=2)


@pytest.mark.parametrize("kind", [I.KIND_INT4_SYM, I.KIND_INT4_ZP, I.KIND_E2M1])
@pytest.mark.parametrize("case", ["dot", "dot_g32", "dot8", "dot8_g128", "gemv", "gemv_g32", "dequant",
                                  "marlin_dequant", "experts"])
def test_every_kernel_compiles_for_hopper(case, kind):
    """Triton's compiler, not its CPU interpreter, type-checks a kernel: a helper returning a tensor zero point in
    one constexpr branch and a scalar in another compiled for no GPU (job 1479's CompilationError) while the
    interpreter, plain Python, ran it. This compiles every kernel for every weight kind for sm_90, on any host."""
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    name, fn, signature, constexprs = next(c for c in _compile_cases() if c[0] == case)
    constexprs = {**constexprs, "KIND": kind}
    if "compute_type" in constexprs:
        constexprs["compute_type"] = tl.bfloat16
    src = ASTSource(fn=fn, signature={**signature, **{c: "constexpr" for c in constexprs}}, constexprs=constexprs)
    try:
        triton.compile(src, target=GPUTarget("cuda", 90, 32))
    except triton.CompilationError:
        raise
    except Exception as e:  # noqa: BLE001  (a host without the CUDA backend's tools: the frontend still ran)
        if "CompilationError" in type(e).__name__:
            raise
        pytest.skip(f"no CUDA backend toolchain here: {type(e).__name__}: {e}"[:200])


# ---------------------------------------------------------------- the vendored Marlin's layout (CPU)
def test_marlin_layout_is_vllms_bit_for_bit():
    """``mstar.utils.marlin.prepare`` builds the B operand, scales and zero points Marlin reads in torch at load; on
    fixed inputs its bytes hash as vLLM 0.33.0's own reference functions' do (``marlin_weights``,
    ``marlin_permute_scales``, ``marlin_zero_points``, ``mxfp4_marlin_process_scales``, digests computed in a vLLM
    environment), so the vendored kernel reads the layout it was written for."""
    import hashlib

    from mstar.utils import marlin

    def digest(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

    k, n, g = 256, 128, 64
    kk, nn, gg = torch.arange(k)[:, None], torch.arange(n)[None, :], torch.arange(k // g)[:, None]
    codes = ((kk * 31 + nn * 7 + 3) % 16).to(torch.uint8)
    s = ((gg * 37 + nn) % 200 + 1).to(torch.bfloat16)
    z = ((gg * 5 + nn * 3) % 16).to(torch.uint8)
    e = ((torch.arange(n)[:, None] * 3 + torch.arange(k // 32)[None, :] * 11) % 40 + 100).to(torch.uint8)
    assert digest(marlin.repack_codes(codes)) == "4b0d84388ea59149"
    assert digest(marlin.permute_scales(s)) == "6341d8d9255a6f11"
    assert digest(marlin.zero_points(z)) == "a4d1b990568ff00b"
    buffers = marlin.prepare(marlin.KIND_ZP, codes.t(), s.t(), z.t())
    assert digest(buffers["marlin_weight"]) == "4b0d84388ea59149"
    assert digest(buffers["marlin_zero"]) == "a4d1b990568ff00b"
    mx = marlin.prepare(marlin.KIND_E2M1, codes.t(), e)
    assert digest(mx["marlin_scale"]) == "8460cded5b686f89" and digest(mx["marlin_weight"]) == "4b0d84388ea59149"


@pytest.mark.parametrize("kind, group", [(I.KIND_INT4_SYM, 128), (I.KIND_INT4_ZP, 128), (I.KIND_INT4_ZP, 32),
                                         (I.KIND_INT4_SYM, 64), (I.KIND_E2M1, 32)])
def test_one_marlin_copy_holds_the_whole_weight(kind, group):
    """A layer that takes Marlin keeps only Marlin's copy: ``unprepare`` reads back exactly the codes, scales and
    zero points ``prepare`` was given, and the format dequantizes that copy to the planes layout's weight (what the
    Marlin-layout dequant kernel, the large-M path, matches on a GPU)."""
    from mstar.utils import marlin

    torch.manual_seed(10)
    n, k = 192, 512
    w = torch.randn(n, k) * 0.05
    if kind == I.KIND_E2M1:
        fmt, stored = I.MXFP4, I.MXFP4.quantize_storage(w)
        codes, scale, zero = I.unpack_fp4(stored["weight_packed"]), stored["weight_scale"], None
    else:
        fmt = I.W4A16Format(group=group, symmetric=kind == I.KIND_INT4_SYM, scale_dtype=torch.bfloat16)
        stored = fmt.quantize_storage(w.to(torch.bfloat16))
        zero = torch.randint(0, 16, (n, k // group), dtype=torch.uint8) if kind == I.KIND_INT4_ZP else None
        if zero is not None:
            stored["weight_zero_point"] = I.pack_int4(zero, dim=0)
        codes, scale = I.unpack_int4(stored["weight_packed"]), stored["weight_scale"]
    held = marlin.prepare(kind, codes, scale, zero)
    back = marlin.unprepare(kind, held)
    assert torch.equal(back[0], codes) and torch.equal(back[1], scale)
    assert zero is None or torch.equal(back[2], zero)
    one_copy = {name: torch.empty(0, dtype=torch.int32) for name in fmt.buffers} | held
    assert torch.equal(fmt.dequantize(one_copy), fmt.dequantize(fmt.after_load(dict(stored))))


ONE_COPY_TOKENS = (1, 64, 256, 1024)


def _stub_marlin(monkeypatch):
    """Stand a reference in for the CUDA Marlin op (built nowhere on a CPU box): ``available`` says built, so the
    formats' after_load takes the branch a GPU takes (Marlin's copy only, the others empty), and ``gemm`` computes
    from that copy. Returns the list each Marlin call appends its token count to."""
    from mstar.utils import marlin, w4a16

    calls = []

    def gemm(lib, x, weight, scale, zero, kind, n, fp32_reduce=True, atomic_add=False):
        calls.append(x.shape[0])
        group = 32 if kind == I.KIND_E2M1 else weight.shape[0] * 16 // scale.shape[0]
        return F.linear(x, I.dequantize_marlin(weight, scale, zero, group, kind).to(x.dtype))

    monkeypatch.setattr(marlin, "available", lambda kind, group, device: "stub")
    monkeypatch.setattr(marlin, "gemm", gemm)
    monkeypatch.setattr(w4a16, "_marlin_lib", lambda kind, group, device: "stub")
    return calls


@pytest.mark.parametrize("layout, group", [("ct-asym", 32), ("awq-asym", 32), ("gptq-sym", 128), ("mxfp4", 32)])
def test_one_copy_layers_serve_every_token_count_after_load(stored, layout, group, monkeypatch):
    """Where Marlin is taken a layer keeps only Marlin's copy. Job 1551 served that into a crash: the op's fake
    (what torch.compile traces) read N from the empty planes layout, ``(M, 0)``. Loaded with Marlin standing in, every
    4-bit linear of the tiny root holds one copy, reads back the checkpoint's weight (Marlin's scales in bf16: fp16
    scales round once), and at 1 / 64 / 256 / 1024 tokens its fake has the real output's shape and its output,
    eager and compiled, matches that weight."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    _stub_marlin(monkeypatch)
    path, refs = stored[(layout, group)]
    root, _ = load(path)
    torch.manual_seed(11)
    for layer in range(LAYERS):
        for kind, rel in PATHS.items():
            module = root.get_submodule(f"layers.{layer}.{rel}")
            assert module.weight_planes.numel() == 0 and module.marlin_weight.numel() > 0, (layer, kind)
            want = _expected(refs, layer, kind, 0, 1)
            weight = _linear_weight(f"layers.{layer}.{rel}", root)
            assert ((weight - want).norm() / want.norm()).item() < 4e-3, (layer, kind)
            compiled = torch.compile(module._quant_local, backend="eager", fullgraph=True, dynamic=False)
            for m in ONE_COPY_TOKENS:
                x = torch.randn(m, want.shape[1], dtype=torch.bfloat16)
                with FakeTensorMode(allow_non_fake_inputs=True) as mode:
                    fake = module._quant_local(mode.from_tensor(x), None)
                assert fake.shape == (m, want.shape[0]), (layer, kind, m, fake.shape)
                ref = F.linear(x.float(), weight)
                for out in (module._quant_local(x, None), compiled(x, None)):
                    assert out.shape == (m, want.shape[0])
                    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-2, (layer, kind, m)


_ONE_COPY_DISPATCH = """
import sys, torch
import torch.nn.functional as F
from mstar.model.components import int4_formats as I
from mstar.utils import marlin, w4a16
marlin_calls, dequant_calls = [], []
def gemm(lib, x, weight, scale, zero, kind, n, fp32_reduce=True, atomic_add=False):
    marlin_calls.append(x.shape[0])
    group = 32 if kind == I.KIND_E2M1 else weight.shape[0] * 16 // scale.shape[0]
    return F.linear(x, I.dequantize_marlin(weight, scale, zero, group, kind).to(x.dtype))
marlin.gemm, marlin.available = gemm, (lambda kind, group, device: "stub")
w4a16._marlin_lib = lambda kind, group, device: "stub"
real_dequant = w4a16.marlin_dequantize
def spy(*a, **kw):
    dequant_calls.append(1)
    return real_dequant(*a, **kw)
w4a16.marlin_dequantize = spy
torch.manual_seed(0)
n, k = 192, 512
for kind, group in ((I.KIND_INT4_SYM, 128), (I.KIND_INT4_ZP, 32), (I.KIND_E2M1, 32)):
    if kind == I.KIND_E2M1:
        fmt = I.MXFP4
        stored = fmt.quantize_storage(torch.randn(n, k) * 0.05)
    else:
        fmt = I.W4A16Format(group=group, symmetric=kind == I.KIND_INT4_SYM, scale_dtype=torch.float16)
        stored = fmt.quantize_storage((torch.randn(n, k) * 0.05).to(torch.bfloat16))
        if kind == I.KIND_INT4_ZP:
            stored["weight_zero_point"] = I.pack_int4(torch.randint(0, 16, (n, k // group), dtype=torch.uint8), dim=0)
    held = fmt.after_load(stored)
    assert held["weight_planes"].numel() == 0 and held["marlin_weight"].numel() > 0
    ref_w = fmt.dequantize(held)
    for m in (1, 64, 256, 1024):
        marlin_calls.clear(); dequant_calls.clear()
        x = torch.randn(m, k).to(torch.bfloat16)
        got = w4a16.gemm(x, held["weight_planes"].view(torch.uint8), held["weight_scale"],
                         held.get("weight_zero_point"), fmt.group, kind, held.get("tinygemm_weight"),
                         held.get("tinygemm_scale_zero"), held["marlin_weight"], held["marlin_scale"],
                         held.get("marlin_zero"))
        ref = F.linear(x.float(), ref_w)
        err = ((got.float() - ref).norm() / ref.norm()).item()
        assert got.shape == (m, n) and err < 1e-2, (kind, m, got.shape, err)
        want_marlin = m < w4a16.MARLIN_M
        assert (marlin_calls == [m]) == want_marlin and bool(dequant_calls) == (not want_marlin), (kind, m)
    # the large-M dequant's lookup tables are copied to the device once, not per call (job 1551: a host copy
    # inside CUDA-graph capture)
    assert marlin.tables(torch.device("cpu")) is marlin.tables(torch.device("cpu"))
print("ok")
"""


def test_one_copy_dispatch_runs_marlin_then_dequantizes_its_copy():
    """``w4a16.gemm`` on one-copy buffers, Marlin standing in and the Marlin-layout dequant kernel in Triton's CPU
    interpreter (a subprocess: the interpreter is chosen at import): Marlin below ``MARLIN_M`` tokens, the
    dequant + GEMM from it at and above, both matching the reference at 1 / 64 / 256 / 1024 tokens for int4
    symmetric g128, zero points g32 and MXFP4."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("triton")
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "TRITON_INTERPRET": "1", "OMP_NUM_THREADS": "2",
           "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])}
    run = subprocess.run([sys.executable, "-c", _ONE_COPY_DISPATCH], env=env, capture_output=True, text=True,
                         check=False, timeout=600)
    assert run.returncode == 0 and run.stdout.strip().endswith("ok"), (run.stdout[-2000:], run.stderr[-4000:])


def test_marlin_takes_the_shapes_it_was_built_for():
    from mstar.utils import marlin

    assert marlin.shape_reason(marlin.KIND_ZP, 128, 5120, 17408) is None
    assert "groups" in marlin.shape_reason(marlin.KIND_SYM, 256, 5120, 5120)
    assert "multiple of 64" in marlin.shape_reason(marlin.KIND_SYM, 128, 5136, 5120)
    assert "two groups" in marlin.shape_reason(marlin.KIND_SYM, 128, 128, 128)
    # off CUDA nothing is built and the format keeps empty placeholders
    fmt = I.W4A16Format(group=128, symmetric=False)
    stored = fmt.quantize_storage(torch.randn(128, 256))
    stored["weight_zero_point"] = I.pack_int4(torch.full((128, 2), 8, dtype=torch.uint8), dim=0)
    done = fmt.after_load(stored)
    assert set(done) == set(fmt.buffers) and done["marlin_weight"].numel() == 0


# ---------------------------------------------------------------- CUDA: the kernels against the reference
def _kernel_case(kind, n, k, group, device):
    w = torch.randn(n, k) * 0.05
    if kind == I.KIND_E2M1:
        q = I.MXFP4.quantize_weight(w, None)
        return q["weight_planes"].view(torch.uint8).to(device), q["weight_scale"].to(device), None
    fmt = I.W4A16Format(group=group, symmetric=kind == I.KIND_INT4_SYM)
    q = fmt.quantize_weight(w, None)
    zp = None if kind == I.KIND_INT4_SYM else I.pack_int4(torch.randint(4, 12, (n, k // group)), dim=0).to(device)
    return q["weight_planes"].view(torch.uint8).to(device), q["weight_scale"].to(device), zp


@cuda
@pytest.mark.parametrize("kind, group", [(I.KIND_INT4_SYM, 128), (I.KIND_INT4_ZP, 128), (I.KIND_INT4_SYM, 32),
                                         (I.KIND_INT4_ZP, 32), (I.KIND_E2M1, 32)])
@pytest.mark.parametrize("m", [1, 3, 7, 48, 300, 600])
def test_cuda_dequant_gemm_matches_the_reference(kind, group, m):
    from mstar.utils import w4a16

    torch.manual_seed(5)
    dev = torch.device("cuda")
    n, k = 2048, 1024
    packed, scale, zp = _kernel_case(kind, n, k, group, dev)
    ref_w = I.dequantize_kind(packed.cpu(), scale.cpu(), None if zp is None else zp.cpu(), group, kind)
    assert torch.equal(w4a16.dequantize(packed, scale, zp, group, kind, torch.float32).cpu(), ref_w)
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    got = torch.ops.mstar.w4a16_linear(x, packed, scale, zp, group, kind)
    ref = F.linear(x.float(), ref_w.to(dev).to(torch.bfloat16).float())
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 5e-3, rel
    # captured and replayed, the same output
    graph = torch.cuda.CUDAGraph()
    torch.ops.mstar.w4a16_linear(x, packed, scale, zp, group, kind)
    with torch.cuda.graph(graph):
        out = torch.ops.mstar.w4a16_linear(x, packed, scale, zp, group, kind)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, got)


@cuda
@pytest.mark.parametrize("kind, group", [(I.KIND_INT4_SYM, 128), (I.KIND_INT4_ZP, 32), (I.KIND_E2M1, 32)])
@pytest.mark.parametrize("path, m, cfg", [
    ("gemv", 1, {"BN": 32, "NB": 4, "SPLIT_K": 1, "num_warps": 4}),
    ("gemv", 4, {"BN": 16, "NB": 2, "SPLIT_K": 3, "num_warps": 4}),
    ("dot", 5, {"BN": 64, "TB": 16, "NB": 2, "SPLIT_K": 2, "STAGES": 3, "num_warps": 4}),
    ("dot", 200, {"BN": 64, "TB": 64, "NB": 1, "SPLIT_K": 1, "STAGES": 3, "num_warps": 8}),
    ("dot8", 9, {"BN": 128, "TB": 16, "NB": 1, "SPLIT_K": 2, "STAGES": 2, "num_warps": 8}),
    ("dot8", 40, {"BN": 64, "TB": 64, "NB": 2, "SPLIT_K": 1, "STAGES": 3, "num_warps": 4}),
])
def test_cuda_every_path_and_split_matches_the_reference(kind, group, path, m, cfg, monkeypatch):
    if path == "dot8" and cfg["TB"] == 64 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10:
        pytest.skip("dot8 with 64 tokens a tile needs 520 tensor-memory columns, Blackwell has 512 (config() takes 32)")
    """Each path at a tile the built-in rule may not pick (the benchmark sweeps them)."""
    from mstar.utils import w4a16

    torch.manual_seed(6)
    dev = torch.device("cuda")
    n, k = 1536, 1024
    packed, scale, zp = _kernel_case(kind, n, k, group, dev)
    ref_w = I.dequantize_kind(packed.cpu(), scale.cpu(), None if zp is None else zp.cpu(), group, kind)
    monkeypatch.setattr(w4a16, "CONFIG_OVERRIDE", lambda *a: (path, cfg))
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    got = w4a16.gemm(x, packed, scale, zp, group, kind)
    ref = F.linear(x.float(), ref_w.to(dev).to(torch.bfloat16).float())
    assert ((got.float() - ref).norm() / ref.norm()).item() < 5e-3


@cuda
@pytest.mark.parametrize("symmetric, group", [(True, 128), (False, 128), (True, 32)])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_cuda_tinygemm_copy_matches_the_reference(symmetric, group, m, monkeypatch):
    """Without Marlin, the format's tinygemm copy (built at load on CUDA) runs 1-3 tokens; its bf16 scales and zeros
    round within a GEMM's tolerance of the reference."""
    monkeypatch.setenv("MSTAR_W4A16_MARLIN", "0")
    torch.manual_seed(8)
    dev = torch.device("cuda")
    n, k = 1024, 2048
    fmt = I.W4A16Format(group=group, symmetric=symmetric, scale_dtype=torch.float16)
    stored = fmt.quantize_storage((torch.randn(n, k, device=dev) * 0.05).to(torch.bfloat16))
    if not symmetric:
        stored["weight_zero_point"] = I.pack_int4(torch.randint(2, 14, (n, k // group), device=dev), dim=0)
    buffers = fmt.after_load(stored)
    assert buffers["tinygemm_weight"].numel() > 0
    ref_w = fmt.dequantize(buffers).to(torch.bfloat16)
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    got = fmt.linear(x, *(buffers[b] for b in fmt.buffers))
    ref = F.linear(x.float(), ref_w.float())
    assert ((got.float() - ref).norm() / ref.norm()).item() < 5e-3


@cuda
@pytest.mark.parametrize("kind, group", [(I.KIND_INT4_SYM, 128), (I.KIND_INT4_ZP, 128), (I.KIND_INT4_SYM, 32),
                                         (I.KIND_INT4_ZP, 32), (I.KIND_E2M1, 32)])
def test_cuda_marlin_matches_the_reference(kind, group):
    """The vendored Marlin (built on this GPU's toolchain: a GPU job has nvcc, so a failed build fails here) runs
    what the format hands ``linear()`` below ``MARLIN_M`` tokens and its copy dequantized above, within a GEMM's
    tolerance of the reference read from that copy (its scales in bf16: AWQ's fp16 scales round once), the same
    output captured and replayed in a CUDA graph."""
    from mstar.utils import marlin, w4a16

    torch.manual_seed(9)
    dev = torch.device("cuda")
    n, k = 1536, 1024
    assert marlin.available(kind, group, dev), "the vendored Marlin did not build here (see the log)"
    if kind == I.KIND_E2M1:
        fmt = I.MXFP4
        buffers = fmt.after_load(fmt.quantize_storage(torch.randn(n, k, device=dev) * 0.05))
    else:
        fmt = I.W4A16Format(group=group, symmetric=kind == I.KIND_INT4_SYM, scale_dtype=torch.float16)
        stored = fmt.quantize_storage((torch.randn(n, k, device=dev) * 0.05).to(torch.bfloat16))
        if kind == I.KIND_INT4_ZP:
            stored["weight_zero_point"] = I.pack_int4(torch.randint(2, 14, (n, k // group), device=dev), dim=0)
        buffers = fmt.after_load(stored)
    # one copy: Marlin's, the planes layout empty
    assert buffers["marlin_weight"].numel() > 0 and buffers["weight_planes"].numel() == 0
    ref_w32 = fmt.dequantize(buffers)
    ref_w = ref_w32.to(torch.bfloat16)
    # the large-M path's dequant from Marlin's copy, exact
    m_zero = buffers.get("marlin_zero")
    assert torch.equal(w4a16.marlin_dequantize(buffers["marlin_weight"], buffers["marlin_scale"], m_zero, group, kind,
                                               torch.float32), ref_w32)
    args = [buffers[b] for b in fmt.buffers]
    compiled = torch.compile(fmt.linear, fullgraph=True, dynamic=False)
    for m in (1, 5, 16, 33, 64, 100, w4a16.MARLIN_M - 1, w4a16.MARLIN_M, 1024):
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        got = fmt.linear(x, *args)
        ref = F.linear(x.float(), ref_w.float())
        assert ((got.float() - ref).norm() / ref.norm()).item() < 5e-3, m
        if m in (1, 64, w4a16.MARLIN_M, 1024):
            # captured and replayed (job 1551: a host copy inside capture at 256+), and compiled (its fake)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fmt.linear(x, *args)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, got), m
            assert torch.equal(compiled(x, *args), got), m


@cuda
@pytest.mark.parametrize("layout, group", [("ct-sym", 32), ("awq-asym", 32), ("gptq-sym", 128), ("mxfp4", 32)])
@pytest.mark.parametrize("skip", [False, True])
def test_cuda_experts_match_the_reference(stored, layout, group, skip):
    path, _ = stored[(layout, group)]
    root, _ = load(path)
    block = root.layers[0].moe
    torch.manual_seed(6)
    h = torch.randn(9, HIDDEN, dtype=torch.bfloat16)
    weights, ids, _ = block.gate(h)
    if skip:
        ids[2, 0] = EXPERTS
    ref = block._experts_reference(h, weights, ids, skip)
    root.to("cuda")
    block.experts_after_load()
    got = block._experts_quant(h.cuda(), weights.cuda(), ids.cuda(), skip).cpu()
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 1e-2, rel


blackwell = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                               reason="needs sm_100 / sm_103 (B200, B300)")


@blackwell
@pytest.mark.parametrize("layout, group, tol", [("mxfp4", 32, 2e-2), ("nvfp4", 16, 5e-2)])
def test_blackwell_fp4_experts_match_the_reference(stored, layout, group, tol):
    path, _ = stored[(layout, group)]
    root, _ = load(path)
    block = root.layers[0].moe
    torch.manual_seed(7)
    h = torch.randn(33, HIDDEN, dtype=torch.bfloat16)
    weights, ids, _ = block.gate(h)
    ref = block._experts_reference(h, weights, ids, False)
    # the dequantized experts on bf16 activations, read before the kernel's layout replaces the checkpoint's
    gate_up, down = _expert_weights(block)
    exact = torch.zeros(h.shape)
    for t in range(h.shape[0]):
        for slot in range(ids.shape[1]):
            e = int(ids[t, slot])
            g, u = (gate_up[e].float() @ h[t].float()).chunk(2)
            exact[t] += float(weights[t, slot]) * (down[e].float() @ (F.silu(g) * u).bfloat16().float())
    root.to("cuda")
    block.experts_after_load()
    assert block._experts_kernel is not None
    got = block._experts_quant(h.cuda(), weights.cuda(), ids.cuda(), False).cpu()
    if layout == "nvfp4":
        # the kernel rounds an activation block's e4m3 scale to nearest (TRT-LLM / modelopt), the reference rounds
        # it up, so their activation roundings are independent (~0.18 apart on random data): bound the kernel by
        # the reference's own distance from the bf16-activation output
        err = lambda y: ((y.float() - exact).norm() / exact.norm()).item()  # noqa: E731
        assert err(got) <= 1.05 * err(ref), (err(got), err(ref))
        return
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < tol, rel


@blackwell
def test_blackwell_nvfp4_stored_linear_matches_the_reference(stored):
    path, _ = stored[("nvfp4", 16)]
    root, _ = load(path)
    module = root.layers[0].mlp.gate_up_proj
    x = torch.randn(17, HIDDEN, dtype=torch.bfloat16)
    ref = module._quant_local(x, None)
    module.to("cuda")
    got = module._quant_local(x.cuda(), None).cpu()
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 2e-2, rel


def test_dims():  # the tiny root's sizes the TP cases above rely on
    assert HIDDEN % (4 * 32) == 0 and INTER % (4 * 32) == 0 and HEADS * HEAD_DIM % 4 == 0


# ---------------------------------------------------------------- Qwen3.5 through its own loader
@pytest.fixture(scope="module")
def qwen_awq(tmp_path_factory):
    """A tiny Qwen3.5 in the layout QuantTrio/Qwen3.5-27B-AWQ ships: AutoAWQ g128 for the MLPs, o_proj and the delta
    net's in_proj_qkv / in_proj_z / out_proj; q/k/v, in_proj_a/b, layer 0 and the head bf16."""
    import shutil
    import sys
    from pathlib import Path

    from safetensors.torch import load_file

    sys.path.insert(0, str(Path(__file__).parent))
    from test_quantization_models import TEXT  # noqa: F401  (the dims there take 128-wide groups)
    from test_qwen3_5_video import write_synthetic_checkpoint
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    bf16 = write_synthetic_checkpoint(tmp_path_factory.mktemp("qwen_bf16"))
    config = json.loads((bf16 / "config.json").read_text())
    config["text_config"].update({**TEXT, "num_hidden_layers": 4,
                                  "layer_types": ["linear_attention"] * 3 + ["full_attention"]})
    config["tie_word_embeddings"] = False
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(Qwen3_5Config(**config)).to(torch.bfloat16).save_pretrained(bf16)
    awq = tmp_path_factory.mktemp("qwen_awq")
    for f in bf16.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            shutil.copy(f, awq / f.name)
    keep = re.compile(r"layers\.0\.|self_attn\.(q|k|v)_proj|in_proj_(a|b)|lm_head|visual|embed|norm|conv1d")
    tensors, refs = {}, {}
    for f in sorted(bf16.glob("*.safetensors")):
        for name, w in load_file(str(f)).items():
            base = name.removesuffix(".weight")
            if not name.endswith(".weight") or w.dim() != 2 or keep.search(name):
                tensors[name] = w
                continue
            parts, refs[base] = layout_tensors(w, "awq-asym", 128)
            tensors.update({f"{base}.{s}": t.contiguous() for s, t in parts.items()})
    save_file(tensors, str(awq / "model.safetensors"))
    config = json.loads((awq / "config.json").read_text())
    config["quantization_config"] = {**QUANT_CONFIG["awq"], "group_size": 128}
    (awq / "config.json").write_text(json.dumps(config))
    return awq, refs


def test_qwen3_5_loads_an_awq_checkpoint_through_its_own_loader(qwen_awq):
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    from mstar.quantization.scope import node_scope

    path, refs = qwen_awq
    model = Qwen3_5DenseModel(str(path))
    with node_scope(model, "LLM", {}) as scope:
        llm = model.get_submodule("LLM", "cpu")
        scope.finish(llm)
    plan = next(iter(scope.plans.values()))[1]
    actions = {p: l.action.value for p, l in plan.layers.items()}
    assert actions["model.layers.0.mlp.gate_up_proj"] == "keep"
    assert actions["model.layers.1.mlp.gate_up_proj"] == actions["model.layers.1.mlp.down_proj"] == "direct"
    assert actions["model.layers.1.self_attn.out_proj"] == actions["model.layers.3.self_attn.o_proj"] == "direct"
    assert actions["model.layers.1.self_attn.in_proj_fused"] == "dequant"  # qkv/z int4, a/b bf16
    assert actions["model.layers.3.self_attn.qkv_proj"] == "keep"
    lm = llm.model
    t = "model.language_model.layers.1"
    gate_up = lm.model.layers[1].mlp.gate_up_proj
    want = torch.cat([refs[f"{t}.mlp.gate_proj"], refs[f"{t}.mlp.up_proj"]])
    assert torch.equal(gate_up.quant_format.dequantize(_buffers(gate_up)), want)
    o = lm.model.layers[3].self_attn.o_proj
    assert torch.equal(o.quant_format.dequantize(_buffers(o)), refs["model.language_model.layers.3.self_attn.o_proj"])
    fused = lm.model.layers[1].self_attn.in_proj_fused.weight
    assert fused.dtype == torch.bfloat16 and torch.isfinite(fused.float()).all()
    x = torch.randn(3, gate_up.input_size_per_partition, dtype=torch.bfloat16)
    assert torch.equal(gate_up(x), F.linear(x, want.to(torch.bfloat16)))
