"""Kimi K3's MXFP4 experts as the release stores them, through the quantization API, on CPU: a tiny
checkpoint in the release's format (packed e2m1 experts, e8m0 scales, its ``quantization_config``)
loads under the node's quantization scope as the engine loads it, the experts held packed, and
the real Engine's logits match the reference over the weights the packing stands for. The
experts' SiTU-GLU reaches the 4-bit path (CPU: its per-expert reference; CUDA: the Triton
kernel's activation spec, ``test_components``). At TP2 over gloo each rank holds its
intermediate slice of the packed experts."""
import datetime

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mstar.model.kimi_k3.config import SAMPLER
from mstar.model.kimi_k3.tests import engine_helpers as H
from mstar.model.kimi_k3.tests.reference import reference_logits
from mstar.model.kimi_k3.tests.test_truncate import make_tiny_release
from mstar.utils.situ import SiTU

PROMPT = torch.tensor([11, 40, 7, 99, 3, 56, 21, 8])
STEPS = 3


def _engine(path, rank: int = 0, group=None, groups=None):
    """The checkpoint at ``path`` loaded as the engine loads a node (``load_submodule``: under its
    quantization scope, no ``quantization:`` block, so the experts load as stored)."""
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo
    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model
    from mstar.quantization.scope import load_submodule

    model = KimiK31Model(str(path), strict_config=False, kda_max_requests=4, tokenizer_mode="byte",
                         kda_conv_dtype=torch.float32)
    sub = load_submodule(model, "LLM", {}, "cpu", tp_group=group, autocast_dtype=torch.float32)
    engine = Engine(None)
    engine.load_model({"LLM": sub}, model.get_node_resources(),
                      parallel_groups=groups or WorkerParallelGroups(num_workers=1, global_rank=0),
                      device=torch.device("cpu"),
                      transfer_engine_info=TransferEngineInfo(
                          my_entity_id=f"e{rank}", my_session_id="s", transfer_engine=LocalTransferEngine("localhost")),
                      kv_cache_type=torch.float32)
    greedy = H.GreedySampler()
    engine._resources[SAMPLER] = engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.node_resources[SAMPLER] = greedy
    engine.warmup()
    return engine, sub, model


def _greedy(engine):
    H.add_request(engine, "a")
    tokens = [H.run_step(engine, "prefill", {"a": PROMPT})["a"].view(1)]
    for _ in range(STEPS):
        tokens.append(H.run_step(engine, "decode", {"a": tokens[-1]})["a"].view(1))
    logits = torch.stack([x[-1] for x in engine._resources[SAMPLER].logits])
    return torch.cat(tokens), logits


def _check(ref_sd, cfg, tokens, logits):
    seq = torch.cat([PROMPT, tokens[:-1]])
    ref = reference_logits({k: v.float() for k, v in ref_sd.items()}, seq, cfg)[len(PROMPT) - 1:]
    torch.testing.assert_close(logits, ref, atol=5e-5, rtol=0)
    assert torch.equal(tokens, ref.argmax(-1))


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    H.install_cpu_stubs(monkeypatch)
    torch.set_num_threads(2)


def test_mxfp4_experts_load_packed_and_match_the_reference(tmp_path):
    ref_sd = make_tiny_release(tmp_path)
    engine, sub, model = _engine(tmp_path)
    block = sub.language_model.model.layers[1].block_sparse_moe
    assert block.experts_packed.name == "mxfp4" and not hasattr(block.experts, "gate_up_proj")
    assert block.experts.gate_up_proj_packed.dtype == torch.uint8
    assert block._experts_activation == SiTU(4.0, 25.0)
    tokens, logits = _greedy(engine)
    _check(ref_sd, model.config, tokens, logits)


def test_the_reference_tells_situ_from_silu(tmp_path):
    """The control: the same packed experts under SwiGLU are visibly off the reference."""
    ref_sd = make_tiny_release(tmp_path)
    engine, sub, model = _engine(tmp_path)
    for layer in sub.language_model.model.layers[1:]:
        layer.block_sparse_moe.activation = "silu"
    tokens, logits = _greedy(engine)
    seq = torch.cat([PROMPT, tokens[:-1]])
    ref = reference_logits({k: v.float() for k, v in ref_sd.items()}, seq, model.config)[len(PROMPT) - 1:]
    assert (logits - ref).abs().max() > 1e-2


def _rank(rank: int, path: str, store: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120))
    torch.set_num_threads(1)
    from mstar.distributed.all_reduce import AllReduce
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups

    class _Patch:
        def setitem(self, d, k, v):
            d[k] = v

        def setattr(self, o, n, v):
            setattr(o, n, v)

    H.install_cpu_stubs(_Patch())
    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    group.device_group = group.cpu_group = dist.new_group([0, 1])
    group.initialized = True
    group._all_reduce = AllReduce(group.device_group)
    try:
        groups = WorkerParallelGroups(num_workers=2, global_rank=rank, any_parallelism=True)
        groups.add("LLM", group)
        engine, sub, _ = _engine(path, rank, group, groups)
        block = sub.language_model.model.layers[1].block_sparse_moe
        assert block.experts.down_proj_packed.shape[-1] * 2 == 32  # this rank's half of 64
        tokens, logits = _greedy(engine)
        torch.save({"tokens": tokens, "logits": logits}, f"{out_dir}/{rank}.pt")
    finally:
        group.device_group = group.cpu_group = group._all_reduce = None
        dist.destroy_process_group()


def test_mxfp4_experts_at_tp2_match_the_reference(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    ref_sd = make_tiny_release(ckpt, moe_intermediate=64)
    mp.spawn(_rank, args=(str(ckpt), str(tmp_path / "store"), str(tmp_path)), nprocs=2)
    r0, r1 = (torch.load(tmp_path / f"{r}.pt") for r in (0, 1))
    assert torch.equal(r0["tokens"], r1["tokens"]) and torch.equal(r0["logits"], r1["logits"])
    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model

    cfg = KimiK31Model(str(ckpt), strict_config=False, tokenizer_mode="byte").config
    _check(ref_sd, cfg, r0["tokens"], r0["logits"])


class _Allocations(torch.utils._python_dispatch.TorchDispatchMode):
    """Every tensor an op returns while it is on: (shape, dtype, op)."""

    def __init__(self):
        super().__init__()
        self.made = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in torch.utils._pytree.tree_leaves(out):
            if isinstance(t, torch.Tensor) and not t.is_meta:
                self.made.append((tuple(t.shape), t.dtype, str(func)))
        return out


def _load(path):
    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model
    from mstar.quantization.scope import load_submodule

    model = KimiK31Model(str(path), strict_config=False, kda_max_requests=4, tokenizer_mode="byte",
                         kda_conv_dtype=torch.float32)
    with _Allocations() as allocs:
        sub = load_submodule(model, "LLM", {}, "cpu", autocast_dtype=torch.bfloat16)
    return sub, model, allocs.made


def test_mxfp4_experts_never_allocate_the_bf16_stacks(tmp_path, monkeypatch):
    """B300 04c: built on meta, the model's ``to_empty`` gave every MoE block its bf16 expert stacks
    (4.59 GiB of gate_up a layer at TP8) before the scope swapped in the packed storage, and TP8
    ran out of memory at load. The scope prepares the root on meta now: no float tensor of a bf16
    expert stack's shape is ever made, and the bytes ``to_empty`` allocates for the experts are
    exactly their packed e2m1 codes and e8m0 scales (0.5 + 1/32 bytes a weight, not 2)."""
    at_to_empty = {}
    to_empty = torch.nn.Module.to_empty

    def counted(self, *, device, recurse=True):
        out = to_empty(self, device=device, recurse=recurse)
        at_to_empty[type(self).__name__] = {
            n: t.numel() * t.element_size() for n, t in [*self.named_parameters(), *self.named_buffers()]}
        return out

    monkeypatch.setattr(torch.nn.Module, "to_empty", counted)
    make_tiny_release(tmp_path)
    sub, model, made = _load(tmp_path)
    cfg = model.config
    e, inter, hidden = cfg.num_experts, cfg.moe_intermediate_size, cfg.routed_expert_hidden_size
    stacks = {(e, 2 * inter, hidden), (e, hidden, inter)}
    bf16_stacks = [m for m in made if m[0] in stacks and m[1].is_floating_point]
    assert not bf16_stacks, bf16_stacks[:3]
    sizes = at_to_empty["KimiK3ForCausalLM"]
    # what to_empty allocated is what the module held right after it, nothing made and dropped
    assert sum(torch.Size(s).numel() * d.itemsize for s, d, op in made if op == "aten.empty_like.default") \
        == sum(sizes.values())
    blocks = [m for m in sub.language_model.modules() if getattr(m, "experts_packed", None) is not None]
    assert blocks and all(b.experts_packed.name == "mxfp4" for b in blocks)
    weights = len(blocks) * e * (2 * inter * hidden + hidden * inter)
    assert sum(b for n, b in sizes.items() if ".experts." in n) == weights // 2 + weights // 32


def test_the_scales_are_marked_unloaded_once_they_exist(tmp_path):
    """Prepared on meta, the scales have no bytes to mark; ``materialize`` marks them after
    ``to_empty``, so a scale the stream never delivers still fails the load."""
    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model
    from mstar.quantization.scope import materialize, node_scope

    make_tiny_release(tmp_path)
    model = KimiK31Model(str(tmp_path), strict_config=False, kda_max_requests=4, tokenizer_mode="byte")
    model._resolve_checkpoint()
    with node_scope(model, "LLM", {}, "cpu") as scope:
        with torch.device("meta"):
            lm = KimiK3ForCausalLM(model.config)
        materialize(lm.to(torch.bfloat16), "cpu")
        blocks = [m for m in lm.modules() if getattr(m, "experts_packed", None) is not None]
        assert blocks and all((b.experts.gate_up_proj_scale == 0xFF).all() for b in blocks)
        assert all((b.experts.down_proj_scale == 0xFF).all() for b in blocks)
        with pytest.raises(ValueError, match="never arrived"):
            scope.finish(lm)


def test_on_a_blackwell_gpu_the_situ_experts_take_the_triton_kernel(tmp_path, monkeypatch):
    """B300 04c: FlashInfer's trtllm-gen has no MXFP4 x bf16 SiTU GEMM ("No kernel found" at the first
    decode), so on a GPU where its FP4 MoE is otherwise available the W4A16 SiTU experts stay on the
    Triton kernel instead of being converted to its layout."""
    from mstar.utils.fused_moe import fp4_flashinfer

    make_tiny_release(tmp_path)
    sub, _, _ = _load(tmp_path)
    block = next(m for m in sub.language_model.modules() if getattr(m, "experts_packed", None) is not None)

    def converted(*args, **kwargs):
        raise AssertionError("converted for the TRT-LLM FP4 MoE")

    monkeypatch.setattr(fp4_flashinfer, "unavailable_reason", lambda device: None)
    monkeypatch.setattr(fp4_flashinfer.TrtllmFp4Experts, "convert", converted)
    block.experts_after_load()
    assert block._experts_kernel is None
