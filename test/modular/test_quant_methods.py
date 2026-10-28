"""Quantization methods: each stored format against its reference, and the routed experts loaded
and run through the method the way a model does (GLM-5.2's fp8, Kimi K2.7's int4, Kimi K3's and
DeepSeek-V4.1's fp4).

The reference transcriptions below are the checkpoints' own: compressed-tensors' ``pack_to_int32``
(``compressors/pack_quantized/helpers.py``) and ``pack_fp4_to_uint8`` / ``compress_mx_scale``, and
DeepSeek-V4.1's ``inference/convert.py`` (``FP4_TABLE``, ``cast_e2m1fn_to_e4m3fn``). Where the
compressed-tensors package is installed it is checked directly too.
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.components.quantization import (  # noqa: E402
    ExpertMath,
    Fp4E2M1,
    Fp8Block,
    Int4Group,
    QuantizedExperts,
    QuantMethod,
    quant_method_for,
)
from mstar.model.components.quantization.formats import (  # noqa: E402
    E2M1_VALUES,
    e8m0_to_float,
    unpack_e2m1,
    unpack_int4,
)
from mstar.model.loader import load_hf_weights  # noqa: E402
from mstar.utils.fp8_block import FP8_DTYPE  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---- the checkpoints' own packers, transcribed --------------------------------------------------

def _ct_pack_int32(q: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """compressed-tensors ``pack_to_int32`` (packed_dim=1)."""
    value = (q + (1 << (num_bits - 1))).to(torch.uint8)
    pack_factor = 32 // num_bits
    rows, cols = value.shape
    padded = -(-cols // pack_factor) * pack_factor
    if padded > cols:
        value = F.pad(value, (0, padded - cols))
    reshaped = value.view(rows, padded // pack_factor, pack_factor).to(torch.int32)
    shifts = torch.arange(pack_factor, dtype=torch.int32) * num_bits
    return (reshaped << shifts).sum(dim=2, dtype=torch.int32)


_DS_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _ds_cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor):
    """DeepSeek-V4.1 ``convert.py``'s lossless fp4 -> fp8 (32 x 32 blocks) cast."""
    out_dim, in_dim = x.size()
    in_dim *= 2
    b = 32
    x = x.view(torch.uint8)
    low, high = x & 0x0F, (x >> 4) & 0x0F
    x = torch.stack([_DS_FP4_TABLE[low.long()], _DS_FP4_TABLE[high.long()]], dim=-1).flatten(2)
    bo, bi = out_dim // b, in_dim // b
    x = x.view(bo, b, bi, b).transpose(1, 2)
    scale = scale.float().view(bo, b, bi, -1).transpose(1, 2).flatten(2)
    top = scale.amax(dim=-1, keepdim=True) / (2 ** 6)
    offset = (scale / top).unflatten(-1, (b, -1)).repeat_interleave(32, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(FP8_DTYPE), top.squeeze(-1).to(torch.float8_e8m0fnu)


def _e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(E2M1_VALUES)
    return (values.unsqueeze(-1) == table).float().argmax(-1)  # -0.0 == 0.0 picks code 0


def _pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """compressed-tensors ``pack_fp4_to_uint8``: element 2j low nibble, 2j + 1 high."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


# ---- formats ------------------------------------------------------------------------------------

def test_int4_unpack_inverts_compressed_tensors_packing():
    q = torch.randint(-8, 8, (6, 60), dtype=torch.int8)  # 60: the last word is padded
    packed = _ct_pack_int32(q)
    assert packed.shape == (6, 8) and packed.dtype == torch.int32
    assert torch.equal(unpack_int4(packed, 60), q)


def test_int4_against_the_compressed_tensors_package():
    helpers = pytest.importorskip("compressed_tensors.compressors.pack_quantized.helpers")
    q = torch.randint(-8, 8, (5, 96), dtype=torch.int8)
    assert torch.equal(helpers.pack_to_int32(q, 4), _ct_pack_int32(q))
    assert torch.equal(unpack_int4(helpers.pack_to_int32(q, 4), 96), q)


def test_int4_group_dequantizes_per_group():
    q = torch.randint(-8, 8, (4, 64), dtype=torch.int8)
    scale = torch.rand(4, 2).bfloat16() + 0.01
    w = Int4Group(32).dequantize({"packed": _ct_pack_int32(q), "scale": scale}, 4, 64,
                                 torch.float32)
    expect = q.float() * scale.float().repeat_interleave(32, dim=1)
    assert torch.equal(w, expect)


def test_e2m1_bytes_decode_as_deepseek_v41_converts_them():
    every_byte = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(1, 256)
    ds = torch.stack([_DS_FP4_TABLE[(every_byte & 0xF).long()],
                      _DS_FP4_TABLE[(every_byte >> 4).long()]], dim=-1).flatten(1)
    assert torch.equal(unpack_e2m1(every_byte, 512), ds)
    # DeepSeek's int8 container reads the same bits
    assert torch.equal(unpack_e2m1(every_byte.view(torch.int8), 512), ds)


def test_e8m0_is_a_power_of_two_biased_127():
    e = torch.arange(1, 255, dtype=torch.int32).to(torch.uint8)
    assert torch.equal(e8m0_to_float(e), torch.exp2(e.float() - 127))
    assert e8m0_to_float(torch.tensor([0], dtype=torch.uint8)).item() == 2.0 ** -127
    # torch's own dtype reads the bytes the same way
    every = torch.arange(255, dtype=torch.int32).to(torch.uint8).view(torch.float8_e8m0fnu)
    assert torch.equal(e8m0_to_float(every), every.float())
    assert torch.isnan(e8m0_to_float(torch.tensor([255], dtype=torch.uint8))).all()


def test_fp4_against_the_compressed_tensors_package():
    helpers = pytest.importorskip("compressed_tensors.compressors.nvfp4.helpers")
    mx = pytest.importorskip("compressed_tensors.compressors.mx_utils")
    codes = torch.randint(0, 16, (3, 64))
    packed = _pack_e2m1(codes)
    values = torch.tensor(E2M1_VALUES)[codes]
    assert torch.equal(helpers.pack_fp4_to_uint8(values), packed)
    exp = torch.randint(120, 135, (3, 2), dtype=torch.int32).to(torch.uint8)
    ours = Fp4E2M1().dequantize({"packed": packed, "scale": exp}, 3, 64, torch.bfloat16)
    theirs = (helpers.unpack_fp4_from_uint8(packed, 3, 64, torch.bfloat16)
              * mx.decompress_mx_scale(exp).repeat_interleave(32, dim=1))
    assert torch.equal(ours, theirs)


def test_fp4_dequant_equals_deepseek_v41s_lossless_fp8_cast():
    """V4.1's converter recasts fp4 experts to fp8 with one e8m0 scale per 32 x 32 block; both
    readings of the same bytes give the same weight."""
    rows, cols = 64, 128
    packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.int32).to(torch.uint8)
    exp = torch.randint(118, 128, (rows, cols // 32), dtype=torch.int32).to(torch.uint8)
    scale = exp.view(torch.float8_e8m0fnu)
    fp8, block_scale = _ds_cast_e2m1fn_to_e4m3fn(packed.view(torch.int8), scale)
    via_fp8 = Fp8Block((32, 32), scale="scale", e8m0=True).dequantize(
        {"fp8": fp8, "scale_inv": block_scale}, rows, cols, torch.float32)
    ours = Fp4E2M1(("weight", "scale")).dequantize(
        {"packed": packed.view(torch.int8), "scale": scale}, rows, cols, torch.float32)
    assert torch.equal(ours, via_fp8)


def test_fp8_block_with_e8m0_scales():
    w = (torch.randn(64, 96) * 4).to(FP8_DTYPE)
    exp = torch.randint(120, 130, (2, 3), dtype=torch.int32).to(torch.uint8)
    got = Fp8Block((32, 32), scale="scale", e8m0=True).dequantize(
        {"fp8": w, "scale_inv": exp.view(torch.float8_e8m0fnu)}, 64, 96, torch.float32)
    expect = w.float() * torch.exp2(exp.float() - 127).repeat_interleave(32, 0).repeat_interleave(
        32, 1)
    assert torch.equal(got, expect)


# ---- quant_method_for on the family's configs ---------------------------------------------------

GLM52 = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
         "weight_block_size": [128, 128]}
V41 = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
       "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
_CT_IGNORE = ["re:.*self_attn.*", "re:.*shared_experts.*",
              "re:.*mlp\\.(gate|up|gate_up|down)_proj.*", "re:.*lm_head.*"]


def _ct(fmt: str, typ: str, extra=None) -> dict:
    weights = {"num_bits": 4, "type": typ, "symmetric": True, "strategy": "group",
               "group_size": 32, "dynamic": False, **(extra or {})}
    return {"quant_method": "compressed-tensors", "format": fmt, "ignore": _CT_IGNORE,
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": weights,
                                          "input_activations": None, "format": fmt}}}


K27 = _ct("pack-quantized", "int")
K3 = _ct("mxfp4-pack-quantized", "float", {"scale_dtype": "torch.uint8"})


def test_quant_method_for_the_family():
    glm = quant_method_for(GLM52)
    assert isinstance(glm.format_for("routed_experts"), Fp8Block)
    assert glm.format_for("linear").block == (128, 128) and not glm.format_for("linear").e8m0
    assert glm.keeps_quantized("routed_experts") and not glm.keeps_quantized("linear")

    v41 = quant_method_for(V41)
    assert isinstance(v41.format_for("routed_experts"), Fp4E2M1)
    assert v41.format_for("routed_experts").parts[0].checkpoint == "weight"
    dense = v41.format_for("linear")
    assert dense.block == (32, 32) and dense.e8m0 and dense.scale_name == "scale"

    k27 = quant_method_for(K27)
    assert isinstance(k27.format_for("routed_experts"), Int4Group)
    assert k27.format_for("routed_experts").group == 32
    k3 = quant_method_for(K3)
    assert isinstance(k3.format_for("routed_experts"), Fp4E2M1)
    assert k3.format_for("routed_experts").parts[0].checkpoint == "weight_packed"

    assert quant_method_for(None) is None
    for bad in ({"quant_method": "awq"}, _ct("pack-quantized", "int", {"symmetric": False}),
                _ct("nvfp4-pack-quantized", "float"), {**V41, "expert_dtype": "int2"}):
        with pytest.raises(ValueError):
            quant_method_for(bad)
    with pytest.raises(ValueError):
        quant_method_for(GLM52, kernels="triton")


# ---- quantizers that write a checkpoint in each format ------------------------------------------

def _quantize(fmt, w: torch.Tensor) -> dict[str, torch.Tensor]:
    """``w`` (rows, cols) as the checkpoint's parts, keyed by checkpoint suffix."""
    rows, cols = w.shape
    if isinstance(fmt, Fp8Block):
        bo, bi = fmt.block
        blocks = w.float().unflatten(0, (-1, bo)).unflatten(-1, (-1, bi))
        amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4)
        if fmt.e8m0:
            exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127).to(torch.int32) + 127
            scale = torch.exp2(exp.float() - 127)
            scale_part = exp.to(torch.uint8).view(torch.float8_e8m0fnu)
        else:
            scale = amax / 448.0
            scale_part = scale
        q = (blocks / scale[:, None, :, None]).flatten(2, 3).flatten(0, 1).to(FP8_DTYPE)
        return {"weight": q, fmt.scale_name: scale_part}
    if isinstance(fmt, Int4Group):
        g = fmt.group
        groups = w.float().view(rows, cols // g, g)
        scale = (groups.abs().amax(-1) / 7).clamp_min(1e-4).bfloat16()
        q = torch.round(groups / scale.float()[..., None]).clamp(-8, 7).to(torch.int8)
        return {"weight_packed": _ct_pack_int32(q.view(rows, cols)), "weight_scale": scale,
                "weight_shape": torch.tensor([rows, cols], dtype=torch.int32)}
    groups = w.float().view(rows, cols // 32, 32)
    exp = (torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 6.0)) + 127).clamp(1, 254)
    scaled = groups / torch.exp2(exp - 127)[..., None]
    table = torch.tensor(E2M1_VALUES)
    codes = (scaled.unsqueeze(-1) - table).abs().argmin(-1).view(rows, cols)
    values_name, scale_name = (p.checkpoint for p in fmt.parts)
    packed, exp = _pack_e2m1(codes), exp.to(torch.int32).to(torch.uint8)
    if values_name == "weight":  # DeepSeek-V4.1's containers
        return {"weight": packed.view(torch.int8), "scale": exp.view(torch.float8_e8m0fnu)}
    return {values_name: packed, scale_name: exp}


FORMATS = {
    "fp8-128": (Fp8Block((16, 16)), ("gate_proj", "up_proj", "down_proj")),
    "fp8-e8m0": (Fp8Block((32, 32), scale="scale", e8m0=True), ("w1", "w3", "w2")),
    "int4": (Int4Group(32), ("gate_proj", "up_proj", "down_proj")),
    "fp4-ct": (Fp4E2M1(), ("w1", "w3", "w2")),
    "fp4-ds": (Fp4E2M1(("weight", "scale")), ("w1", "w3", "w2")),
}
E, HIDDEN, INTER, TOPK = 4, 64, 64, 2
_EXPERT_KEY = re.compile(r"(.*)\.experts\.(\d+)\.(\w+)\.(\w+)$")


class _Layer(nn.Module):
    """``mlp.experts``, as a model holds them."""

    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = experts


def _checkpoint(fmt, names, dense):
    """One layer's experts as checkpoint tensors ``mlp.experts.<i>.<proj>.<part>``."""
    tensors = []
    for i, (gate_up, down) in enumerate(dense):
        for proj, w in ((names[0], gate_up[:INTER]), (names[1], gate_up[INTER:]),
                        (names[2], down)):
            for suffix, t in _quantize(fmt, w).items():
                tensors.append((f"mlp.experts.{i}.{proj}.{suffix}", t))
    return tensors


def _load(fmt, names, tensors, rank=0, world=1, kernels="auto", math=ExpertMath()):
    method = QuantMethod({"routed_experts": fmt, "linear": fmt}, kernels=kernels)
    group = SimpleNamespace(rank=rank, world_size=world)
    experts = method.routed_experts(E, INTER, HIDDEN, group, math=math)
    layer = _Layer(experts)
    proj_of = dict(zip(names, ("gate_proj", "up_proj", "down_proj"), strict=True))

    def remap(name):
        m = _EXPERT_KEY.match(name)
        prefix, i, proj, suffix = m.groups()
        return f"{prefix}.experts.{proj_of[proj]}.__expert{i}__.{suffix}"

    stream = method.checkpoint_stream(iter(tensors), lambda base: "routed_experts")
    loaded = load_hf_weights(
        layer, ((remap(n), t) for n, t in stream),
        stacked_params=method.expert_rules(E, lambda proj, i: f".experts.{proj}.__expert{i}__"))
    assert loaded == {n for n, _ in layer.named_parameters()}
    return experts


def _dense_layer():
    return [((torch.randn(2 * INTER, HIDDEN) * 0.05), (torch.randn(HIDDEN, INTER) * 0.05))
            for _ in range(E)]


@pytest.mark.parametrize("key", list(FORMATS))
def test_experts_load_sharded_like_the_whole(key):
    """TP2's two ranks hold exactly the halves of the TP1 experts' dequantized weights."""
    fmt, names = FORMATS[key]
    tensors = _checkpoint(fmt, names, _dense_layer())
    whole = _load(fmt, names, tensors)
    ranks = [_load(fmt, names, tensors, rank=r, world=2) for r in (0, 1)]
    half = INTER // 2
    for e in range(E):
        gu = whole.expert_weights("gate_up_proj", e, torch.float32)
        down = whole.expert_weights("down_proj", e, torch.float32)
        for r, shard in enumerate(ranks):
            sl = slice(r * half, (r + 1) * half)
            assert torch.equal(shard.expert_weights("gate_up_proj", e, torch.float32),
                               torch.cat([gu[:INTER][sl], gu[INTER:][sl]]))
            assert torch.equal(shard.expert_weights("down_proj", e, torch.float32), down[:, sl])


def _routed(x, topk_weights, topk_ids, weights, math=ExpertMath()):
    """The HF experts' combine over dense per-expert weights, token by token."""
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            gate_up, down = weights[int(topk_ids[t, k])]
            gate, up = (x[t:t + 1] @ gate_up.T).chunk(2, dim=-1)
            h = F.silu(gate) * up if math.act is None else math.act(gate, up)
            w = topk_weights[t, k]
            o = (h * w).to(x.dtype) @ down.T if math.weights_before_down else (
                (h.to(x.dtype) @ down.T) * w)
            out[t] += o[0].to(out.dtype)
    return out


def _clamped_swiglu(gate, up, limit=10.0):
    """DeepSeek-V4.1's expert activation (``inference/model.py`` ``Expert``)."""
    gate, up = gate.float(), up.float()
    return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("key", list(FORMATS))
@pytest.mark.parametrize("math", [ExpertMath(), ExpertMath(_clamped_swiglu, True)],
                         ids=["silu-after-down", "v41-before-down"])
def test_reference_forward_is_the_dense_experts(key, device, math):
    fmt, names = FORMATS[key]
    experts = _load(fmt, names, _checkpoint(fmt, names, _dense_layer()), math=math).to(device)
    experts.resolve_kernels(device)
    assert not experts.fused or device == "cuda"
    experts.fused = False
    x = (torch.randn(6, HIDDEN, device=device) * 0.5).float()
    logits = torch.randn(6, E, device=device)
    topk_weights, topk_ids = torch.topk(logits.softmax(-1), TOPK, dim=-1)
    dense = [(experts.expert_weights("gate_up_proj", e, x.dtype),
              experts.expert_weights("down_proj", e, x.dtype)) for e in range(E)]
    got = experts(x, topk_weights, topk_ids)
    # one GEMM over the hit tokens vs one per token: CUDA's fp32 GEMMs (TF32 where enabled)
    # round differently by shape
    tol = dict(rtol=1e-5, atol=1e-6) if device == "cpu" else dict(rtol=2e-3, atol=3e-5)
    torch.testing.assert_close(got, _routed(x, topk_weights, topk_ids, dense, math), **tol)


def test_kernel_choice():
    fmt, names = FORMATS["int4"]
    tensors = _checkpoint(fmt, names, _dense_layer())
    auto = _load(fmt, names, tensors)
    auto.resolve_kernels("cpu")
    assert auto.fused is False  # no fused int4 kernel yet: the reference loop serves
    with pytest.raises(RuntimeError, match="fused"):
        _load(fmt, names, tensors, kernels="fused").resolve_kernels("cpu")
    fp8, fp8_names = FORMATS["fp8-128"]
    swiglu = _load(fp8, fp8_names, _checkpoint(fp8, fp8_names, _dense_layer()),
                   math=ExpertMath(_clamped_swiglu, True))
    swiglu.resolve_kernels("cuda")  # the fused kernel serves only the default math
    assert swiglu.fused is False


@pytest.mark.parametrize("key", list(FORMATS))
def test_a_module_dtype_cast_leaves_the_stored_parts_alone(key):
    """The engine casts each submodule to its autocast dtype after loading: the experts' stored
    parts keep their container dtypes and values (fp8 block scales stay fp32, as the fused kernel
    takes them), while the module's other weights are cast."""
    fmt, names = FORMATS[key]
    experts = _load(fmt, names, _checkpoint(fmt, names, _dense_layer()))
    before = {n: p.detach().clone() for n, p in experts.named_parameters()}
    layer = _Layer(experts)
    layer.router = nn.Linear(HIDDEN, E, bias=False)
    layer.to(torch.bfloat16)
    assert layer.router.weight.dtype == torch.bfloat16
    for n, p in experts.named_parameters():
        assert p.dtype == before[n].dtype and torch.equal(p, before[n]), n
        assert hasattr(p, "weight_loader"), n


def test_shards_that_cut_a_stored_element_are_refused():
    with pytest.raises(ValueError, match="int4"):
        QuantizedExperts(Int4Group(32), E, 40, HIDDEN, SimpleNamespace(rank=0, world_size=2))
    with pytest.raises(ValueError, match="fp4"):
        QuantizedExperts(Fp4E2M1(), E, 96, HIDDEN, SimpleNamespace(rank=0, world_size=2))


# ---- the checkpoint stream ----------------------------------------------------------------------

def test_stream_dequantizes_what_stays_bf16_and_passes_the_rest():
    int4 = Int4Group(32)
    method = QuantMethod({"routed_experts": int4, "linear": int4})
    dense = torch.randn(8, 64) * 0.05
    parts = _quantize(int4, dense)
    norm = torch.randn(64).bfloat16()
    stream = [("lin.weight_packed", parts["weight_packed"]),
              ("lin.weight_scale", parts["weight_scale"]),
              ("lin.weight_shape", parts["weight_shape"]),
              ("experts.0.up_proj.weight_shape", parts["weight_shape"]),
              ("experts.0.up_proj.weight_packed", parts["weight_packed"]),
              ("norm.weight", norm)]
    role = {"lin": "linear", "experts.0.up_proj": "routed_experts"}
    out = list(method.checkpoint_stream(iter(stream), lambda b: role.get(b, "linear")))
    names = [n for n, _ in out]
    # the resident expert passes through as stored (its aux part dropped), the linear dequantizes
    assert names == ["lin.weight", "experts.0.up_proj.weight_packed", "norm.weight"]
    expect = int4.dequantize({"packed": parts["weight_packed"], "scale": parts["weight_scale"]},
                             8, 64)
    assert torch.equal(dict(out)["lin.weight"], expect)


def test_stream_refuses_what_no_format_claims():
    method = quant_method_for(K27)
    with pytest.raises(ValueError, match="claims"):
        list(method.checkpoint_stream(iter([("x.weight", torch.zeros(2, 2, dtype=torch.int8))]),
                                      lambda b: "linear"))
    with pytest.raises(ValueError, match="incomplete"):
        list(method.checkpoint_stream(
            iter([("x.weight_packed", torch.zeros(2, 4, dtype=torch.int32))]), lambda b: "linear"))
