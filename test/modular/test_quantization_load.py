"""Loading under a quantization plan, on CPU: a bf16 checkpoint quantized at load and the same weights stored as
block fp8 end up as the same tensors on every rank, a stored layer kept bf16 is dequantized exactly, and the
fp8 layers' outputs track bf16's within e4m3's rounding."""

import pytest
import torch
import torch.nn.functional as F
from quant_helpers import EXPERTS, HEADS, HIDDEN, INTER, KV_HEADS, build, full_weights, load, write_checkpoint

from mstar.model.components.quant_linear import linear_format
from mstar.model.loader import iter_safetensors_shards, load_weights_into
from mstar.quantization.resolve import Action
from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight

WEIGHTS = full_weights()
# routers stay bf16, as in the released fp8 checkpoints
NOT_HEAD = lambda base: base != "lm_head" and not base.endswith("moe.gate")  # noqa: E731


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt")
    return {
        "bf16": write_checkpoint(root / "bf16", WEIGHTS),
        "fp8": write_checkpoint(root / "fp8", WEIGHTS, NOT_HEAD),
        "q_only": write_checkpoint(root / "q_only", WEIGHTS, lambda base: base.endswith("q_proj")),
        "gates": write_checkpoint(root / "gates", WEIGHTS, lambda base: base.endswith("moe.gate")),
    }


@pytest.fixture(autouse=True)
def _threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def state(root):
    """Every quantized tensor of the root, by path."""
    return {name: t for name, t in root.state_dict().items() if t.dtype in (torch.uint8, torch.int32)}


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_on_the_fly_fp8_block_swaps_every_candidate(checkpoints):
    root, plan = load(checkpoints["bf16"], "fp8_block")
    assert {layer.action for layer in plan.layers.values()} == {Action.QUANTIZE, Action.KEEP}
    qkv = root.layers[0].self_attn.qkv_proj
    assert qkv.quant_format.name == "fp8_block"
    assert qkv.weight.dtype == torch.uint8 and qkv.weight_scale_inv.dtype == torch.int32
    assert qkv.weight_scale_inv.shape == ((HEADS + 2 * KV_HEADS) * 128 // 128, HIDDEN // 128)
    assert root.layers[0].moe.experts.gate_up_proj.dtype == torch.uint8
    assert root.lm_head.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("tp", [1, 2])
def test_a_stored_fp8_checkpoint_loads_into_the_tensors_on_the_fly_quantization_makes(checkpoints, tp):
    for rank in range(tp):
        direct, plan = load(checkpoints["fp8"], None, rank, tp)
        assert {layer.action for layer in plan.layers.values()} == {Action.DIRECT, Action.KEEP}
        quantized, _ = load(checkpoints["bf16"], "fp8_block", rank, tp)
        a, b = state(direct), state(quantized)
        # per layer: qkv, o, gate_up, down (weight + scale each) and the experts' four
        assert a.keys() == b.keys() and len(a) == 2 * (4 * 2 + 4)
        for name in a:
            assert torch.equal(a[name], b[name]), name


def test_each_rank_loads_its_rows_and_scale_rows(checkpoints):
    full = state(load(checkpoints["fp8"])[0])
    ranks = [state(load(checkpoints["fp8"], None, rank, 2)[0]) for rank in range(2)]
    q, kv = HEADS * 128, KV_HEADS * 128

    def qkv_rows(t, r, unit):
        # rank r's q heads, then its k and v heads: the full tensor's [q | k | v] blocks, each split in two
        blocks = t.split([q // unit, kv // unit, kv // unit])
        return torch.cat([b.chunk(2)[r] for b in blocks])

    for r in range(2):
        prefix = "layers.1.self_attn"
        assert torch.equal(ranks[r][f"{prefix}.qkv_proj.weight"], qkv_rows(full[f"{prefix}.qkv_proj.weight"], r, 1))
        assert torch.equal(ranks[r][f"{prefix}.qkv_proj.weight_scale_inv"],
                           qkv_rows(full[f"{prefix}.qkv_proj.weight_scale_inv"], r, 128))
        # row-parallel: columns, and the scales' columns
        assert torch.equal(ranks[r][f"{prefix}.o_proj.weight"], full[f"{prefix}.o_proj.weight"].chunk(2, 1)[r])
        assert torch.equal(ranks[r][f"{prefix}.o_proj.weight_scale_inv"],
                           full[f"{prefix}.o_proj.weight_scale_inv"].chunk(2, 1)[r])
        # experts: gate and up halves split per rank, down's K split
        gate_up = full["layers.1.moe.experts.gate_up_proj"]
        assert torch.equal(ranks[r]["layers.1.moe.experts.gate_up_proj"],
                           torch.cat([h.chunk(2, 1)[r] for h in gate_up.chunk(2, 1)], 1))
        assert torch.equal(ranks[r]["layers.1.moe.experts.down_proj_scale_inv"],
                           full["layers.1.moe.experts.down_proj_scale_inv"].chunk(2, 2)[r])


def test_a_stored_layer_kept_bf16_is_dequantized_exactly(checkpoints):
    root, plan = load(checkpoints["fp8"], {"keep_bf16": ["layers.0.**"]})
    assert plan.layers["layers.0.mlp.gate_up_proj"].action is Action.DEQUANT
    loaded = root.layers[0].mlp.gate_up_proj.weight
    assert loaded.dtype == torch.bfloat16
    expected = torch.cat([
        dequantize_fp8_block_weight(*_stored(checkpoints["fp8"], f"layers.0.mlp.{p}")) for p in ("gate_proj", "up_proj")
    ])
    assert torch.equal(loaded, expected)
    assert root.layers[1].mlp.gate_up_proj.quant_format.name == "fp8_block"


def _stored(path, base):
    tensors = dict(iter_safetensors_shards(path))
    return tensors[f"{base}.weight"], tensors[f"{base}.weight_scale_inv"]


def test_a_half_quantized_fused_layer_is_dequantized_then_requantized(checkpoints):
    root, plan = load(checkpoints["q_only"], None)
    assert plan.layers["layers.0.self_attn.qkv_proj"].action is Action.DEQUANT
    assert root.layers[0].self_attn.qkv_proj.weight.dtype == torch.bfloat16
    root, plan = load(checkpoints["q_only"], "fp8_block")
    assert plan.layers["layers.0.self_attn.qkv_proj"].action is Action.REQUANT
    assert root.layers[0].self_attn.qkv_proj.quant_format.name == "fp8_block"


def test_a_scale_that_never_arrives_is_refused(checkpoints, monkeypatch):
    from types import SimpleNamespace

    from quant_helpers import STACKED, declaration, remap

    from mstar.quantization.scope import node_scope

    model = SimpleNamespace(get_quantization_roots=lambda: [declaration(checkpoints["fp8"])])
    lost = "layers.1.self_attn.k_proj.weight_scale_inv"
    stream = ((n, t) for n, t in iter_safetensors_shards(checkpoints["fp8"]) if n != lost)
    with pytest.raises(ValueError, match="some of its scales never arrived"):
        with node_scope(model, "LLM", {}) as scope:
            root = build()
            load_weights_into(root, stream, stacked_params=STACKED, name_remapper=remap)
            scope.finish(root)


def test_fp8_layers_track_bf16_and_stored_equals_on_the_fly(checkpoints):
    torch.manual_seed(0)
    x = torch.randn(7, HIDDEN, dtype=torch.bfloat16)
    ref, _ = load(checkpoints["bf16"])
    direct, _ = load(checkpoints["fp8"])
    fly, _ = load(checkpoints["bf16"], "fp8_block")
    for leaf in ("self_attn.qkv_proj", "mlp.gate_up_proj"):
        r = ref.layers[0].get_submodule(leaf)(x)
        d = direct.layers[0].get_submodule(leaf)(x)
        assert torch.equal(d, fly.layers[0].get_submodule(leaf)(x))
        # e4m3 keeps 3 mantissa bits: about 2-3% of a GEMM over random rows; a wrong scale is order 1
        assert rel(d, r) < 0.05, leaf
    h = torch.randn(5, INTER // 1 * 2, dtype=torch.bfloat16)[:, :HEADS * 128]
    assert rel(direct.layers[0].self_attn.o_proj(h), ref.layers[0].self_attn.o_proj(h)) < 0.05
    m_ref, m_fp8 = ref.layers[0].moe(x), direct.layers[0].moe(x)
    assert torch.equal(m_fp8, fly.layers[0].moe(x))
    assert rel(m_fp8, m_ref) < 0.05


def test_a_stored_linear_outside_every_declared_layer_is_dequantized(checkpoints):
    # the routers are no declared layer; a checkpoint storing them fp8 still loads them as bf16
    root, plan = load(checkpoints["gates"], None)
    assert plan.layers["layers.0.moe.gate"].kind == "other"
    assert plan.layers["layers.0.moe.gate"].action is Action.DEQUANT
    weight, scale = _stored(checkpoints["gates"], "layers.0.moe.gate")
    assert torch.equal(root.layers[0].moe.gate.weight, dequantize_fp8_block_weight(weight, scale))


def test_the_engines_dtype_cast_leaves_the_quantized_bytes(checkpoints):
    root, _ = load(checkpoints["fp8"])
    before = {k: v.clone() for k, v in state(root).items()}
    root.to(dtype=torch.bfloat16)
    assert all(torch.equal(before[k], v) for k, v in state(root).items())


@pytest.mark.parametrize("name", ["fp8_tensor", "fp8_channel", "fp8_block"])
def test_each_fp8_format_rounds_within_e4m3(name):
    torch.manual_seed(0)
    fmt = linear_format(name)
    # rows over six decades: per-tensor flushes the small ones, per-channel and per-block keep them
    w = (torch.randn(256, 512) * torch.logspace(-3, 1, 256)[:, None]).to(torch.bfloat16)
    buffers = fmt.quantize_weight(w, None)
    assert all(t.dtype in (torch.uint8, torch.int32) for t in buffers.values())
    back = fmt.dequantize(buffers)
    # within an eighth of its scale's amax (the tensor's for fp8_tensor, its row's or less otherwise), a value is
    # an e4m3 normal
    amax = w.float().abs().amax() if name == "fp8_tensor" else w.float().abs().amax(dim=1, keepdim=True)
    big = w.float().abs() > amax / 8
    # three mantissa bits: a value near its scale's top is within 1/16 of itself
    assert ((back - w.float()).abs()[big] <= w.float().abs()[big] / 15).all()
    x = torch.randn(9, 512, dtype=torch.bfloat16)
    out = fmt.linear(x, *(buffers[b] for b in fmt.buffers))
    assert out.dtype == torch.bfloat16 and out.shape == (9, 256)
    # the GEMM of what the stored operands stand for
    assert rel(out, F.linear(x.float(), back)) < (0.05 if name != "fp8_block" else 0.01)


def test_a_checkpoint_tensor_lands_in_the_container_it_stands_for():
    fmt = linear_format("fp8_block")
    w = torch.randn(256, 256).to(FP8_DTYPE)
    assert fmt.from_checkpoint("weight", w).dtype == torch.uint8
    scale = fmt.from_checkpoint("weight_scale_inv", torch.ones(2, 2, dtype=torch.bfloat16))
    assert scale.dtype == torch.int32 and torch.equal(scale.view(torch.float32), torch.ones(2, 2))
    channel = linear_format("fp8_channel").from_checkpoint("weight_scale", torch.ones(256))
    assert channel.shape == (256, 1) and channel.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
                    reason="needs e4m3 tensor cores")
@pytest.mark.parametrize("name", ["fp8_tensor", "fp8_channel", "fp8_block"])
def test_the_cuda_kernels_match_the_reference(name):
    torch.manual_seed(0)
    fmt = linear_format(name)
    fmt.prepare_device(torch.device("cuda"))
    w = (torch.randn(512, 1024, device="cuda") * 0.05).to(torch.bfloat16)
    # viewed as a served layer hands them to linear() (what the bytes stand for)
    stands_for = {st.name: st.stands_for for st in fmt.storage(512, 1024)}
    buffers = {name: t.view(stands_for[name]) for name, t in fmt.quantize_weight(w, None).items()}
    for m in (1, 7, 64, 300):
        x = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16)
        gpu = fmt.linear(x, *(buffers[b] for b in fmt.buffers))
        cpu = fmt.linear(x.cpu(), *(buffers[b].cpu() for b in fmt.buffers))
        assert rel(gpu.cpu(), cpu) < 0.02, (name, m)


def test_experts_count_matches_the_helper():
    assert EXPERTS == 4


def test_linear_dtype_quantizes_a_tensor_parallel_linear_in_place(checkpoints, monkeypatch):
    # the per-model `linear_dtype` path (FLUX.2 klein, Z-Image) must keep a TP linear's own forward
    from quant_helpers import LINEARS

    from mstar.model.components.distributed import QKVParallelLinear
    from mstar.model.components.quant_linear import apply_linear_dtype

    fmt = linear_format("fp8_block")
    monkeypatch.setattr(type(fmt), "unavailable_reason", lambda self, device: None)
    root, _ = load(checkpoints["bf16"], None, rank=1, tp=2)
    names = apply_linear_dtype(root, "fp8_block", LINEARS)
    qkv = root.layers[0].self_attn.qkv_proj
    assert "layers.0.self_attn.qkv_proj" in names
    assert type(qkv) is QKVParallelLinear and qkv.quant_format is fmt


@pytest.mark.parametrize("source", ["fp8", "bf16"])
def test_the_format_sees_what_the_stored_bytes_stand_for(checkpoints, monkeypatch, source):
    # stored (DIRECT) and quantized at load hand linear() the same dtypes: e4m3 and fp32, never the containers
    root, _ = load(checkpoints[source], None if source == "fp8" else "fp8_block")
    qkv = root.layers[0].self_attn.qkv_proj
    seen = []
    real = type(qkv.quant_format).linear

    def spy(self, x, *buffers):
        seen.append([b.dtype for b in buffers])
        return real(self, x, *buffers)

    monkeypatch.setattr(type(qkv.quant_format), "linear", spy)
    qkv(torch.randn(3, HIDDEN, dtype=torch.bfloat16))
    assert seen == [[FP8_DTYPE, torch.float32]]


def test_the_triton_launch_takes_the_weight_bytes(monkeypatch):
    # dense_fp8's w8a16 kernel bitcasts loaded bytes; an e4m3 pointer fails its masked load at compile
    from mstar.utils import dense_fp8

    launched = {}

    class Kernel:
        def __getitem__(self, grid):
            return lambda x, w, *args, **kwargs: launched.setdefault("w", w.dtype)

    monkeypatch.setattr(dense_fp8, "_w8a16_kernel", Kernel())
    w = torch.zeros(256, 256, dtype=FP8_DTYPE)
    s = torch.ones(2, 2)
    try:
        dense_fp8._w8a16(torch.zeros(4, 256, dtype=torch.bfloat16), w, s, (128, 128), False)
    except Exception:  # noqa: BLE001  (the stub writes no output; only the launch's dtype matters)
        pass
    assert launched["w"] == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
                    reason="needs e4m3 tensor cores")
def test_a_stored_layer_runs_the_cuda_kernels(checkpoints):
    # the boot crash of Qwen3.5-27B-FP8: a DIRECT layer's buffers through the CUDA path
    torch.manual_seed(0)
    direct, _ = load(checkpoints["fp8"])
    fly, _ = load(checkpoints["bf16"], "fp8_block")
    for root in (direct, fly):
        layer = root.layers[0].mlp.gate_up_proj
        layer.quant_format.prepare_device(torch.device("cuda"))
        x = torch.randn(5, HIDDEN, dtype=torch.bfloat16)
        cpu = layer(x)
        gpu = layer.to("cuda")(x.cuda())
        assert rel(gpu.cpu(), cpu) < 0.02
