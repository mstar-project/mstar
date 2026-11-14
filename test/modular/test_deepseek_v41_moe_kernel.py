"""DeepSeek-V4.1's routed experts on the grouped GEMM kernel, without a GPU: the kernel's launches
compiled for Hopper at the checkpoint's shapes, and the path around them (order, routing
weights, rounding, the slots a rank skips) against the torch dispatch with a torch stand-in for
the kernel."""
from __future__ import annotations

import dataclasses
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest
import torch

pytest.importorskip("triton.backends.nvidia")

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.model.components.moe_parallel import MoeParallelConfig  # noqa: E402
from mstar.model.deepseek_v41 import _testing  # noqa: E402
from mstar.model.deepseek_v41.components import moe as moe_module  # noqa: E402
from mstar.model.deepseek_v41.components.moe import DeepseekV41MoE  # noqa: E402
from mstar.model.deepseek_v41.config import DeepseekV41Config  # noqa: E402
from mstar.model.deepseek_v41.deepseek_v41_model import DeepseekV41Model, _cast_floating  # noqa: E402
from mstar.model.deepseek_v41.weight_loader import load_deepseek_v41_weights  # noqa: E402
from mstar.model.utils import check_loaded, process_weights_after_loading  # noqa: E402
from mstar.testing_utils.triton_compile import compile_launches, record_launches  # noqa: E402
from mstar.utils.fused_moe import w4a16  # noqa: E402
from mstar.utils.mx_round import dequant_row_fp4  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DTYPES = pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
# (ranks, the node group's expert split): one rank, a slice of every expert a rank, whole experts a rank
SPLITS = pytest.mark.parametrize("tp, parallel", [(1, "tp"), (2, "tp"), (2, "ep"), (8, "ep")],
                                 ids=["one-rank", "tp-slice", "whole-experts-tp2", "whole-experts-tp8"])


def _exact_gemm(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``a @ weight.T`` summed exactly (fp64 holds every partial sum of these products) and
    rounded to fp32 once: the same bits in any summation order."""
    return (a.double() @ weight.double().T).float()


class TorchGroupedGemm:
    """``fused_moe_w4a16_kernel`` in torch, from a launch's own arguments: each tile of
    ``BLOCK_SIZE_M`` aligned slots below ``num_tokens_post_padded`` gives its valid slots their
    row of ``A`` times the tile's expert, summed exactly and stored in ``C``'s dtype. A tile
    the launch's grid does not reach is an error."""

    def __init__(self, fn) -> None:
        self.arg_names, self.launches = fn.arg_names, 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            v = dict(zip(self.arg_names, args, strict=False)) | kwargs
            assert v["KIND"] == w4a16.KIND_E2M1 and v["GROUP"] == 32
            block_m, slots = v["BLOCK_SIZE_M"], v["num_valid_tokens"]
            padded = int(v["num_tokens_post_padded_ptr"])
            tiles = -(-v["EM"] // block_m)
            assert grid == (tiles * -(-v["N"] // v["BLOCK_SIZE_N"]),) and padded <= tiles * block_m
            a, c, words, scales = v["a_ptr"], v["c_ptr"], v["b_ptr"], v["s_ptr"]
            for tile in range(padded // block_m):
                slot = v["sorted_token_ids_ptr"][tile * block_m:(tile + 1) * block_m].long()
                slot = slot[slot < slots]
                expert = int(v["expert_ids_ptr"][tile])
                weight = dequant_row_fp4(words[expert].contiguous().view(torch.uint8),
                                         scales[expert].view(torch.float8_e8m0fnu), torch.float32)
                acc = _exact_gemm(a[slot // v["top_k"]], weight)
                if v["MUL_ROUTED_WEIGHT"]:
                    acc = acc * v["topk_weights_ptr"].flatten()[slot, None]
                c[slot] = acc.to(c.dtype)
            self.launches += 1

        return launch


@pytest.fixture
def stand_in(monkeypatch):
    """The grouped path on this host: the kernel replaced by ``TorchGroupedGemm``, and every
    other GEMM of the block summed exactly too, so the two paths can differ by their glue
    only (an fp32 sum depends on the order BLAS takes for the batch it is handed)."""
    kernel = TorchGroupedGemm(w4a16.fused_moe_w4a16_kernel)
    monkeypatch.setattr(w4a16, "fused_moe_w4a16_kernel", kernel)
    monkeypatch.setattr(w4a16, "grouped_gemm_unavailable", lambda device: None)
    monkeypatch.setattr(moe_module, "fp32_linear", lambda x, weight: _exact_gemm(x, weight).to(x.dtype))
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield kernel
    torch.set_num_threads(threads)


class _OneMoE(torch.nn.Module):
    """One block's MoE under the name the loader finds it by, ``model.layers.0.ffn``."""

    def __init__(self, config: DeepseekV41Config, group: CommGroup | None = None, moe_parallel=None) -> None:
        super().__init__()
        self.config, self.comm_group = config, group or CommGroup.trivial()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].ffn = DeepseekV41MoE(config, group, moe_parallel)

    @property
    def ffn(self) -> DeepseekV41MoE:
        return self.model.layers[0].ffn


def _all_reduce_after(sums: list[torch.Tensor]):
    """A rank's all-reduce in a group whose ranks run one after another: its tensor plus what
    the ranks before it sent in the same call (``sums``, which it updates)."""
    calls = itertools.count()

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        call = next(calls)
        if call == len(sums):
            sums.append(torch.zeros_like(tensor))
        sums[call] = sums[call] + tensor
        return sums[call]

    return all_reduce


def _block(config: DeepseekV41Config, tensors, dtype, rank: int = 0, tp: int = 1, parallel: str = "tp") -> _OneMoE:
    """One loaded MoE block as rank ``rank`` of ``tp``, in ``dtype`` as a served model is."""
    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(tp)))
    root = _cast_floating(_OneMoE(config, group, MoeParallelConfig(parallel=parallel)), dtype)
    check_loaded(root, load_deepseek_v41_weights(root, tensors.items()), "one block")
    process_weights_after_loading(root, torch.device("cpu"))
    return root


def _run_ranks(config: DeepseekV41Config, tensors, x: torch.Tensor, tp: int, parallel: str) -> list[torch.Tensor]:
    """The block over a ``tp``-wide group, rank after rank: each rank's output."""
    sums: list[torch.Tensor] = []
    outs = []
    for rank in range(tp):
        root = _block(config, tensors, x.dtype, rank, tp, parallel)
        assert root.ffn.capturable == (config.moe_kernel == "auto")
        root.ffn.comm_group.all_reduce = _all_reduce_after(sums)
        outs.append(root.ffn(x))
    return outs


def _inputs(config: DeepseekV41Config, tokens: int, dtype, seed: int = 1) -> torch.Tensor:
    """Hidden states whose 32-wide blocks span a few binades, some past the SwiGLU clamp."""
    g = torch.Generator().manual_seed(seed)
    scales = torch.exp2(torch.randint(-3, 3, (tokens, config.hidden_size // 32), generator=g).float())
    return (torch.randn(tokens, config.hidden_size, generator=g) * scales.repeat_interleave(32, dim=1)).to(dtype)


def _ffn_tensors(config: DeepseekV41Config) -> dict[str, torch.Tensor]:
    return {name: t for name, t in _testing.random_checkpoint(config, seed=0).items()
            if name.startswith("layers.0.ffn.")}


class TestGroupedPath:
    @DTYPES
    @SPLITS
    def test_the_grouped_path_is_the_dispatch_bit_for_bit(self, stand_in, monkeypatch, dtype, tp, parallel):
        """On every rank, with the rows of a step taken in several passes too."""
        monkeypatch.setattr(moe_module, "GROUPED_ROWS", 16)
        config = DeepseekV41Config.reduced(moe_intermediate_size=64 * tp, n_routed_experts=16, num_experts_per_tok=3)
        tensors, x = _ffn_tensors(config), _inputs(config, 37, dtype)
        grouped = _run_ranks(config, tensors, x, tp, parallel)
        assert stand_in.launches == tp * 2 * 3  # a gate/up and a down launch a pass
        dispatch = _run_ranks(dataclasses.replace(config, moe_kernel="torch"), tensors, x, tp, parallel)
        assert stand_in.launches == tp * 2 * 3
        for got, want in zip(grouped, dispatch, strict=True):
            assert torch.isfinite(want).all() and want.dtype == dtype and torch.equal(got, want)

    @DTYPES
    def test_through_the_engine_the_logits_are_the_dispatchs_bit_for_bit(self, stand_in, monkeypatch, tmp_path,
                                                                        dtype):
        """Loaded as a server loads it: a prefill and decode steps."""
        _testing.patch_cpu_engine(monkeypatch)
        config, _ = _testing.write_snapshot(tmp_path, DeepseekV41Config.reduced(), seed=0)
        ids = _testing.random_ids(config, 24 + 5, seed=1)
        logits = {}
        for kernel in ("auto", "torch"):
            engine, sampler, model = _testing.cpu_engine(tmp_path, dtype, moe_kernel=kernel)
            layers = model.get_submodule("LLM").language_model.model.layers
            assert [layer.ffn.capturable for layer in layers] == [kernel == "auto"] * len(layers)
            launches = stand_in.launches
            logits[kernel] = _testing.run_request(engine, sampler, ids, 24)
            assert (stand_in.launches > launches) == (kernel == "auto")
        assert torch.isfinite(logits["torch"]).all() and torch.equal(logits["auto"], logits["torch"])

    @DTYPES
    def test_a_tokens_experts_are_summed_in_the_dispatchs_order(self, stand_in, dtype):
        """By ascending expert id, whatever order the router lists them in: routing weights
        many binades apart make an fp32 sum of a token's experts depend on its order."""
        config = DeepseekV41Config.reduced(n_routed_experts=16, num_experts_per_tok=4)
        ffn = _block(config, _ffn_tensors(config), dtype).ffn
        g = torch.Generator().manual_seed(3)
        tokens, top_k = 29, config.num_experts_per_tok
        x = moe_module.round_fp8(_inputs(config, tokens, dtype), config.fp8_block)
        ids = torch.stack([torch.randperm(16, generator=g)[:top_k] for _ in range(tokens)])
        ids[0] = torch.tensor([9, 7, 4, 2])  # the router's order may be any, descending too
        binades = torch.rand(tokens, top_k, generator=g).argsort(1)
        weights = torch.rand(tokens, top_k, generator=g) * torch.exp2(-6.0 * binades)
        want = ffn._dispatch(x, weights, ids)
        assert torch.equal(ffn._grouped(x, weights, ids), want)
        # the data does tell orders apart: the same slots summed as the router lists them
        listed = torch.zeros_like(want)
        for slot in range(top_k):
            one = torch.zeros_like(weights)
            one[:, slot] = weights[:, slot]
            listed = listed + ffn._dispatch(x, one, ids)
        assert not torch.equal(listed, want)

    def test_the_launches_are_the_same_whatever_the_routing(self, monkeypatch):
        """Grids, tiles and buffer shapes follow from the row count: a step that sends every row
        to one expert launches what a step that spreads them launches."""
        monkeypatch.setattr(w4a16, "grouped_gemm_unavailable", lambda device: None)
        kernel = record_launches(monkeypatch, w4a16, "fused_moe_w4a16_kernel")
        config = DeepseekV41Config.reduced(n_routed_experts=16, num_experts_per_tok=3)
        ffn = _block(config, _ffn_tensors(config), torch.bfloat16).ffn
        tokens, top_k = 21, config.num_experts_per_tok
        x = _inputs(config, tokens, torch.bfloat16)
        weights = torch.rand(tokens, top_k)
        spread = torch.stack([torch.randperm(16, generator=torch.Generator().manual_seed(i))[:top_k]
                              for i in range(tokens)])
        seen = []
        for ids in (spread, torch.tensor([[5, 0, 11]]).expand(tokens, top_k)):
            kernel.calls.clear()
            kernel.grids.clear()
            ffn._grouped(x, weights, ids)
            seen.append([(grid, _shapes(kernel.arguments(i))) for i, grid in enumerate(kernel.grids)])
        assert len(seen[0]) == 2 and seen[0] == seen[1]


def _shapes(arguments: dict) -> dict:
    """A launch's arguments with each tensor as its shape and dtype."""
    return {name: (tuple(v.shape), v.dtype) if isinstance(v, torch.Tensor) else v for name, v in arguments.items()}


def _checkpoint_config() -> DeepseekV41Config:
    return DeepseekV41Config.from_hf_config(json.loads((FIXTURES / "deepseek_v41_config.json").read_text()))


class TestKernelCompiles:
    @DTYPES
    @pytest.mark.parametrize("tp, parallel", [(1, "tp"), (8, "tp"), (8, "ep")],
                             ids=["one-rank", "tp8-slice", "tp8-whole-experts"])
    def test_at_the_checkpoints_shapes_every_launch_compiles_for_hopper(self, monkeypatch, dtype, tp, parallel):
        """Hidden 5120, 384 experts, top 6; 2304 intermediate dims a rank, a TP8 slice's 288
        (K steps with masked tails) and a TP8 rank's 48 whole experts: a decode step's 16-row
        tiles and a prefill's 64-row tiles, as the block itself launches them."""
        kernel = record_launches(monkeypatch, w4a16, "fused_moe_w4a16_kernel")
        config = _checkpoint_config()
        group = CommGroup(my_global_rank=tp - 1, my_group_rank=tp - 1, group_members=list(range(tp)))
        with torch.device("meta"):  # the launcher reads the experts' shapes and strides only
            ffn = DeepseekV41MoE(config, group, MoeParallelConfig(parallel=parallel))
        hidden, top_k, inter = config.hidden_size, config.num_experts_per_tok, ffn.expert_inter
        for tokens in (1, 385):
            ids = (torch.arange(tokens * top_k).reshape(tokens, top_k) * 7) % config.n_routed_experts
            out = ffn._grouped(torch.zeros(tokens, hidden, dtype=dtype), torch.ones(tokens, top_k), ids)
            assert out.shape == (tokens, hidden) and out.dtype == torch.float32
        gate_up, down = kernel.arguments(0), kernel.arguments(1)
        assert (gate_up["N"], gate_up["K"], down["N"], down["K"]) == (2 * inter, hidden, hidden, inter)
        assert [kernel.arguments(i)["BLOCK_SIZE_M"] for i in (0, 2)] == [16, 64]
        assert not any(kernel.arguments(i)["MUL_ROUTED_WEIGHT"] for i in range(4))
        # a tile size's two launches differ in top_k alone: slots share a token's row, then own theirs
        assert compile_launches(kernel) == 4


class TestKernelKnob:
    @pytest.mark.parametrize("value", ["triton", "AUTO", True, None])
    def test_an_unknown_kernel_is_refused(self, value):
        with pytest.raises(ValueError, match="moe_kernel must be one of"):
            DeepseekV41Config(moe_kernel=value)
        with pytest.raises(ValueError, match="moe_kernel must be one of"):
            DeepseekV41Model("deepseek-ai/DeepSeek-V4.1-Flash", moe_kernel=value)

    @pytest.mark.parametrize("layout, kernel, where, capturable", [
        ("fp4", "auto", "gpu", True), ("fp4", "torch", "gpu", False), ("fp8", "auto", "gpu", False),
        ("dense", "auto", "gpu", False), ("fp4", "auto", "cpu", False),
    ])
    def test_capturable_says_which_path_runs_once_the_weights_are_loaded(self, monkeypatch, layout, kernel, where,
                                                                         capturable):
        """The grouped GEMMs for fp4 experts where the kernel runs and the knob leaves it on;
        the dispatch anywhere else."""
        if where == "gpu":
            monkeypatch.setattr(w4a16, "grouped_gemm_unavailable", lambda device: None)
        config = DeepseekV41Config.reduced(moe_experts=layout, moe_kernel=kernel)
        ffn = _OneMoE(config).ffn
        assert not ffn.capturable  # not before the weights are on their device
        process_weights_after_loading(ffn, torch.device("cpu"))
        assert ffn.capturable is capturable
