"""DeepSeek-V4.1 without its reference: the checkpoint's config and tensor layout against the
model's parameters, the loader's guards, and the model in a CPU engine against itself.

``fixtures/deepseek_v41_config.json`` is the checkpoint's own;
``fixtures/deepseek_v41_checkpoint_layout.json`` lists its 96,085 tensors by name, dtype and
shape, read from the 48 shard headers.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.host_table.config import HostTableConfig, HostTableSpec, HostTableStep
from mstar.engine.resources.step import StepContext
from mstar.model.components.moe_parallel import MoeParallelConfig
from mstar.model.deepseek_v41 import _testing
from mstar.model.deepseek_v41._testing import add_request, cpu_engine, logit_gap, random_ids, run_request, run_step
from mstar.model.deepseek_v41.components.engram import Engram, bucket_primes, host_tables, table_name, table_rows
from mstar.model.deepseek_v41.components.layers import DeepseekV41ForCausalLM
from mstar.model.deepseek_v41.components.moe import DeepseekV41MoE
from mstar.model.deepseek_v41.config import CSA, ENGRAM, MOE_EXPERTS, STATE, DeepseekV41Config, kv_cache_key
from mstar.model.deepseek_v41.deepseek_v41_model import DeepseekV41Model
from mstar.model.deepseek_v41.weight_loader import load_deepseek_v41_weights, skipped, upcast_fp4_block
from mstar.model.registry import get_model_class, model_init_kwargs
from mstar.model.utils import check_loaded
from mstar.utils.mx_round import dequant_row_fp4, pack_e2m1, pow2_ceil, round_e2m1
from mstar.validation import validate_deployment

FIXTURES = Path(__file__).parent / "fixtures"
DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn,
          "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8}


def _checkpoint_config() -> DeepseekV41Config:
    return DeepseekV41Config.from_hf_config(json.loads((FIXTURES / "deepseek_v41_config.json").read_text()))


def _checkpoint_tensors():
    """``(name, dtype, shape)`` of every tensor in the checkpoint."""
    layout = json.loads((FIXTURES / "deepseek_v41_checkpoint_layout.json").read_text())
    for pattern, dtype, shape, index in layout["groups"]:
        if index is None:
            names = [pattern]
        elif isinstance(index[0], int):
            names = [pattern.format(i) for i in index]
        else:
            names = [pattern.format(i, j) for i in index[0] for j in range(index[1])]
        for name in names:
            yield name, DTYPES[dtype], shape


def _rank(rank: int, tp: int) -> CommGroup:
    """Rank ``rank`` of a ``tp``-wide group that never communicates."""
    return CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(tp)))


class TestCheckpointConfig:
    def test_the_checkpoints_config_parses_to_the_defaults(self):
        assert _checkpoint_config() == DeepseekV41Config()

    def test_round_trip_through_the_hf_spelling(self):
        config = DeepseekV41Config.reduced()
        read = DeepseekV41Config.from_hf_config(_testing.to_hf_config(config))
        for knob in ("max_seq_len", "max_position_embeddings"):
            setattr(read, knob, getattr(config, knob))
        assert read == config

    def test_layer_sources(self):
        config = DeepseekV41Config()
        assert [config.kv_source_of(i) for i in (0, 1, 2, 7, 8, 19, 20, 39)] == [None, None, 2, 2, 8, 14, 20, 20]
        assert config.pooled_sources == (2, 8, 14)
        # a cache per pooling ratio, a plane per source of that ratio
        assert config.source_ratios == (1, 2)
        assert config.sources_of(2) == (2, 8, 14) and config.sources_of(1) == (20,)
        assert [config.plane_of(i) for i in (2, 7, 8, 14, 19, 20, 39)] == [0, 0, 1, 2, 2, 0, 0]
        assert [config.uses_candidates(i) for i in (20, 24)] == [False, True]

    def test_the_caches_take_890_bytes_a_token(self):
        """A latent and an index key per source, a row per pooled group, in their FP4 bytes."""
        config = DeepseekV41Config()
        latent, key = config.cache_widths()
        assert (2 * latent, 2 * key) == (288, 68)
        per_token = sum(len(config.sources_of(r)) * 2 * (latent + key) / r for r in config.source_ratios)
        assert per_token == 890

    def test_engram_tables_hold_their_bucket_primes_end_to_end(self):
        """The checkpoint's table sizes are the sums of the primes this layout derives."""
        config = DeepseekV41Config()
        primes = bucket_primes(config)
        assert len({p for layer in primes for sizes in layer for p in sizes}) == 2 * 3 * 8
        assert table_rows(config) == config.engram_num_embeddings == (384006168, 384016682)

    @pytest.mark.parametrize("change", [
        {"model_type": "glm_moe_dsa"},
        {"quantization_config": {"quant_method": "fp8", "scale_fmt": "ue8m0", "expert_dtype": "fp8",
                                 "weight_block_size": [32, 32]}},
    ])
    def test_another_checkpoint_is_refused(self, change):
        hf = json.loads((FIXTURES / "deepseek_v41_config.json").read_text()) | change
        with pytest.raises(ValueError):
            DeepseekV41Config.from_hf_config(hf)

    def test_candidates_that_cannot_fill_the_topk_are_refused(self):
        with pytest.raises(ValueError, match="cover index_topk"):
            DeepseekV41Config.reduced(candidate_topk_blocks=2)


class TestCheckpointLayout:
    """Every checkpoint tensor has a parameter of its shape or is named as not served, and
    every parameter is filled: the full-size model on ``meta``, the tensors too. At TP8, the
    last rank's shard of each."""

    @pytest.mark.parametrize("moe_experts, tp", [*((layout, 1) for layout in MOE_EXPERTS), ("fp4", 8)])
    def test_full_size_model_loads_every_tensor_by_name_and_shape(self, moe_experts, tp):
        config = dataclasses.replace(_checkpoint_config(), moe_experts=moe_experts)
        token_map = torch.arange(config.vocab_size) % config.engram_compressed_vocab_size
        with torch.device("meta"):
            lm = DeepseekV41ForCausalLM(config, token_map, comm_group=_rank(tp - 1, tp))
        served = [(n, d, s) for n, d, s in _checkpoint_tensors() if not skipped(n)]
        weights = ((name, torch.empty(shape, dtype=dtype, device="meta")) for name, dtype, shape in served)
        check_loaded(lm, load_deepseek_v41_weights(lm, weights), "checkpoint layout")

    @pytest.mark.parametrize("tp, parallel", [(1, "tp"), (8, "tp"), (8, "ep")])
    def test_fp4_experts_are_held_in_the_checkpoints_own_bytes(self, tp, parallel):
        """A rank's routed experts take its share of the bytes the checkpoint stores them in:
        a slice of each, or under ``ep_config: {parallel: ep}`` its 48 whole ones, 2304 wide."""
        config = _checkpoint_config()
        with torch.device("meta"):
            moe = DeepseekV41MoE(config, _rank(tp - 1, tp), MoeParallelConfig(parallel=parallel))
        whole = parallel == "ep"
        assert moe.experts.down_proj_packed.shape == (384 // tp if whole else 384, 5120, 1152 if whole else 1152 // tp)
        assert moe.shared.down_proj.shape == (5120, 2304 // tp)
        held = sum(p.numel() * p.element_size() for p in moe.experts.parameters())
        shipped = sum(math.prod(shape) * dtype.itemsize for name, dtype, shape in _checkpoint_tensors()
                      if re.match(r"layers\.0\.ffn\.experts\.\d+\.", name))
        assert held * tp == shipped
        assert round(held * config.num_hidden_layers / 1e9, 1) == {1: 288.8, 8: 36.1}[tp]

    def test_on_the_host_the_engram_tables_are_the_only_tensors_without_a_parameter(self):
        """``engram_tables: host`` loads every other tensor as before, and what it declares to
        the host table resource is the checkpoint's Engram tables, row for row and byte for byte."""
        config = _checkpoint_config()
        token_map = torch.arange(config.vocab_size) % config.engram_compressed_vocab_size
        served = [(n, d, s) for n, d, s in _checkpoint_tensors() if not skipped(n)]
        params = {}
        for placement in ("device", "host"):
            with torch.device("meta"):
                lm = DeepseekV41ForCausalLM(dataclasses.replace(config, engram_tables=placement), token_map)
            weights = ((name, torch.empty(shape, dtype=dtype, device="meta")) for name, dtype, shape in served)
            check_loaded(lm, load_deepseek_v41_weights(lm, weights), "checkpoint layout")
            params[placement] = {name for name, _ in lm.named_parameters()}
        assert params["host"] < params["device"]
        assert params["device"] - params["host"] == {
            f"model.layers.{layer}.engram.{leaf}" for layer in (1, 14) for leaf in ("table", "table_scale")}
        declared = {part.tensor: [table.rows, part.width]
                    for table in host_tables(config).values() for part in table.parts}
        stored = {name: [shape[0], shape[1] * dtype.itemsize] for name, dtype, shape in served
                  if ".engram.embed." in name}
        assert declared == stored and len(stored) == 4

    def test_what_is_skipped_is_dspark_vision_and_the_image_router_bias(self):
        names = [n for n, _, _ in _checkpoint_tensors() if skipped(n)]
        kinds = {n.split(".")[0] for n in names if not n.endswith("gate.bias_vl")}
        assert kinds == {"mtp", "vision", "aligner", "image_start", "image_end", "image_newline"}
        assert sum(n.endswith("gate.bias_vl") and n.startswith("layers.") for n in names) == 40


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    path = tmp_path_factory.mktemp("dsv41")
    config, _ = _testing.write_snapshot(path, DeepseekV41Config.reduced(), seed=0)
    return path, config


def _variant(snapshot, tmp_path, edit) -> Path:
    """``snapshot`` with its tensors passed through ``edit``."""
    path, _ = snapshot
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (path / name).is_file():
            (tmp_path / name).write_bytes((path / name).read_bytes())
    tensors = load_file(str(path / "model.safetensors"))
    edit(tensors)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


class TestLoaderGuards:
    def test_a_missing_tensor_refuses_to_load(self, snapshot, tmp_path):
        path = _variant(snapshot, tmp_path, lambda t: t.pop("layers.3.attn.indexer.wk.weight"))
        with pytest.raises(KeyError, match="missing 1 parameters"):
            DeepseekV41Model(str(path), max_seq_len=256).get_submodule("LLM")

    @pytest.mark.parametrize("moe_experts", MOE_EXPERTS)
    def test_one_missing_expert_matrix_refuses_to_load(self, snapshot, tmp_path, moe_experts):
        def drop(tensors):
            del tensors["layers.2.ffn.experts.5.w3.weight"], tensors["layers.2.ffn.experts.5.w3.scale"]

        with pytest.raises(KeyError, match="gate_up_proj"):
            DeepseekV41Model(str(_variant(snapshot, tmp_path, drop)), max_seq_len=256,
                             moe_experts=moe_experts).get_submodule("LLM")

    def test_a_weight_without_its_scale_refuses_to_load(self, snapshot, tmp_path):
        path = _variant(snapshot, tmp_path, lambda t: t.pop("layers.0.attn.wq_b.scale"))
        with pytest.raises(KeyError, match="without their weight or scale"):
            DeepseekV41Model(str(path), max_seq_len=256).get_submodule("LLM")

    def test_an_unknown_tensor_refuses_to_load(self, snapshot, tmp_path):
        path = _variant(snapshot, tmp_path, lambda t: t.update({"layers.0.attn.extra.weight": torch.zeros(2, 2)}))
        with pytest.raises(KeyError, match="has no parameter"):
            DeepseekV41Model(str(path), max_seq_len=256).get_submodule("LLM")

    def test_another_tokenizer_than_the_tables_were_hashed_with_is_refused(self, snapshot, tmp_path):
        path = _variant(snapshot, tmp_path, lambda t: None)
        hf = json.loads((path / "config.json").read_text())
        hf["text_config"]["engram_compressed_vocab_size"] += 1
        (path / "config.json").write_text(json.dumps(hf))
        with pytest.raises(ValueError, match="tables were hashed over"):
            DeepseekV41Model(str(path), max_seq_len=256).get_submodule("LLM")


class TestExpertUpcast:
    def _fp4(self, spread_bits: int):
        g = torch.Generator().manual_seed(0)
        w = torch.randn(64, 64, generator=g)
        w[::2] *= 2.0 ** -spread_bits  # every other row far below its neighbours
        groups = w.unflatten(-1, (-1, 32))
        scale = pow2_ceil(groups.abs().amax(-1) / 6.0)
        q = round_e2m1((groups / scale.unsqueeze(-1)).clamp(-6.0, 6.0)).flatten(-2)
        return pack_e2m1(q), scale

    def test_the_fp8_upcast_holds_every_fp4_value(self):
        packed, scale = self._fp4(spread_bits=6)
        fp8, block_scale = upcast_fp4_block(packed, scale, 32)
        dense = fp8.float() * block_scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
        assert torch.equal(dense, dequant_row_fp4(packed, scale, torch.float32))

    def test_rows_too_far_apart_for_fp8_are_refused(self):
        packed, scale = self._fp4(spread_bits=20)
        with pytest.raises(ValueError, match="too far apart"):
            upcast_fp4_block(packed, scale, 32)


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


def _read_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 bytes as an expert kernel reads them: input ``k``'s E2M1 code in byte ``k // 2``, the
    even input in the low nibble, the sign in the code's bit 3; a row's scale byte ``s`` for its
    inputs ``32 g`` to ``32 g + 31`` is ``2 ** (s - 127)``."""
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    values = torch.where(codes >= 8, -1.0, 1.0) * torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[codes & 7]
    # an fp32 with exponent field s and no mantissa bits is 2 ** (s - 127)
    return values * (scale.int() << 23).view(torch.float32).repeat_interleave(32, dim=-1)


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


def _run_ranks(config: DeepseekV41Config, tensors: dict[str, torch.Tensor], x: torch.Tensor,
               tp: int, parallel: str = "tp") -> list[torch.Tensor]:
    """One block's MoE over a ``tp``-wide group, rank after rank: each rank's output, the last
    one's being the group's."""
    sums: list[torch.Tensor] = []
    outs = []
    for rank in range(tp):
        root = _OneMoE(config, _rank(rank, tp), MoeParallelConfig(parallel=parallel))
        check_loaded(root, load_deepseek_v41_weights(root, tensors.items()), "one block")
        root.ffn.comm_group.all_reduce = _all_reduce_after(sums)
        outs.append(root.ffn(x))
    return outs


class TestExpertLayouts:
    @pytest.mark.parametrize("value", ["fp16", "FP4", True, None])
    def test_an_unknown_layout_is_refused(self, value):
        with pytest.raises(ValueError, match="moe_experts must be one of"):
            DeepseekV41Config(moe_experts=value)
        with pytest.raises(ValueError, match="moe_experts must be one of"):
            DeepseekV41Model("deepseek-ai/DeepSeek-V4.1-Flash", moe_experts=value)

    def test_a_rank_slice_that_splits_a_scale_group_is_refused(self):
        with pytest.raises(ValueError, match="scale blocks"):
            DeepseekV41MoE(DeepseekV41Config.reduced(), _rank(0, 2))  # 32 / 2 = 16 dims a rank
        # 48 dims are whole 16-wide activation blocks, not whole 32-wide fp4 scale groups
        config = DeepseekV41Config.reduced(fp8_block=16, moe_intermediate_size=48)
        DeepseekV41MoE(dataclasses.replace(config, moe_experts="dense"))
        with pytest.raises(ValueError, match="fp4 experts scale 32 inputs"):
            DeepseekV41MoE(config)

    @pytest.mark.parametrize("tp", [2, 8])
    def test_a_ranks_packed_bytes_are_its_slice_of_the_dense_weights(self, tp):
        """At the checkpoint's widths (TP8: 288 intermediate dims a rank, 9 scale groups), for
        every rank: the bytes it holds, read as the MXFP4 expert kernels read them, are its rows
        of gate then up and its columns of down."""
        config = dataclasses.replace(_checkpoint_config(), n_routed_experts=2, num_experts_per_tok=1)
        hidden, inter, local = config.hidden_size, config.moe_intermediate_size, config.moe_intermediate_size // tp
        g = torch.Generator().manual_seed(tp)
        tensors, full = {}, {}
        for which, (rows, cols) in {"w1": (inter, hidden), "w3": (inter, hidden), "w2": (hidden, inter)}.items():
            # every code, scales over 2 ** -8 to 2 ** 8; the stack's second expert
            packed = torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)
            scale = torch.randint(119, 136, (rows, cols // 32), generator=g, dtype=torch.uint8)
            tensors[f"layers.0.ffn.experts.1.{which}.weight"] = packed.view(torch.int8)
            tensors[f"layers.0.ffn.experts.1.{which}.scale"] = scale.view(torch.float8_e8m0fnu)
            full[which] = _read_mxfp4(packed, scale)

        def experts(layout: str, rank: int):
            root = _OneMoE(dataclasses.replace(config, moe_experts=layout), _rank(rank, tp))
            load_deepseek_v41_weights(root, tensors.items())
            return root.ffn.experts

        for rank in range(tp):
            mine = slice(rank * local, (rank + 1) * local)
            gate_up, down = torch.cat([full["w1"][mine], full["w3"][mine]]), full["w2"][:, mine]
            fp4 = experts("fp4", rank)
            assert {name: (p.dtype, tuple(p.shape)) for name, p in fp4.named_parameters()} == {
                "gate_up_proj_packed": (torch.uint8, (2, 2 * local, hidden // 2)),
                "gate_up_proj_scale": (torch.uint8, (2, 2 * local, hidden // 32)),
                "down_proj_packed": (torch.uint8, (2, hidden, local // 2)),
                "down_proj_scale": (torch.uint8, (2, hidden, local // 32)),
            }
            assert torch.equal(_read_mxfp4(fp4.gate_up_proj_packed[1], fp4.gate_up_proj_scale[1]), gate_up)
            assert torch.equal(_read_mxfp4(fp4.down_proj_packed[1], fp4.down_proj_scale[1]), down)
        # the last rank's slices are what the dense layout holds for it
        dense = experts("dense", tp - 1)
        assert torch.equal(dense.gate_up_proj[1], gate_up) and torch.equal(dense.down_proj[1], down)

    @pytest.mark.usefixtures("cpu")
    @pytest.mark.parametrize("parallel", ["tp", "ep"])
    @pytest.mark.parametrize("tp", [2, 8])
    def test_over_a_tp_group_fp4_is_dense_bit_for_bit_and_the_group_matches_one_rank(self, tp, parallel):
        """A slice of every expert a rank, or whole experts a rank (``ep_config: {parallel: ep}``)."""
        config = DeepseekV41Config.reduced(moe_intermediate_size=32 * tp)
        tensors = {name: t for name, t in _testing.random_checkpoint(config, seed=0).items()
                   if name.startswith("layers.0.ffn.")}
        x = torch.randn(24, config.hidden_size, generator=torch.Generator().manual_seed(1))
        fp4 = _run_ranks(config, tensors, x, tp, parallel)
        dense = _run_ranks(dataclasses.replace(config, moe_experts="dense"), tensors, x, tp, parallel)
        assert all(torch.equal(a, b) for a, b in zip(fp4, dense, strict=True))
        assert logit_gap(fp4[-1], _run_ranks(config, tensors, x, 1)[0]) < 1e-5

    def test_a_node_group_may_ask_for_whole_experts_a_rank_and_nothing_else_of_ep(self, snapshot, tmp_path):
        model = DeepseekV41Model(str(snapshot[0]), max_seq_len=256)
        group = {"node_names": ["LLM"], "ranks": [0, 1], "tp_size": 2, "ep_config": {"parallel": "ep"}}
        (tmp_path / "ep.yaml").write_text(json.dumps({"model": "deepseek_v41", "node_groups": [group]}))
        model.get_sharding_config(str(tmp_path / "ep.yaml"))  # refused for a node that is not EP-enabled
        with pytest.raises(ValueError, match="ep_config.parallel only"):
            model.get_submodule("LLM", moe_parallel=MoeParallelConfig(parallel="ep", log_expert_load_every=8))


@pytest.fixture
def cpu(monkeypatch):
    _testing.patch_cpu_engine(monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


@pytest.mark.usefixtures("cpu")
class TestEngineCpu:
    PROMPT, SHORT, STEPS = 45, 22, 6

    def _solo(self, snapshot, ids, prompt, **kwargs):
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32, **kwargs)
        return run_request(engine, sampler, ids, prompt)

    def _pair(self, snapshot, dtype=torch.float32, **kwargs):
        """Two requests prefilled in one step and decoded together: each one's ids, its logits
        step by step, and the model."""
        ids = {"a": random_ids(snapshot[1], self.PROMPT + self.STEPS, seed=1),
               "b": random_ids(snapshot[1], self.SHORT + self.STEPS, seed=2)}
        prompt = {"a": self.PROMPT, "b": self.SHORT}
        engine, sampler, model = cpu_engine(snapshot[0], dtype, **kwargs)
        for rid in ids:
            add_request(engine, rid)
        rows = [run_step(engine, sampler, "prefill", {rid: ids[rid][:prompt[rid]] for rid in ids})]
        for step in range(self.STEPS):
            rows.append(run_step(engine, sampler, "decode", {
                rid: ids[rid][prompt[rid] + step].view(1) for rid in ids}))
        return ids, {rid: torch.stack([row[rid] for row in rows]) for rid in ids}, model

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    @pytest.mark.parametrize("moe_experts", ["fp4", "fp8"])
    def test_packed_experts_give_the_dense_experts_logits_bit_for_bit(self, snapshot, moe_experts, dtype):
        def experts(model):
            return model.get_submodule("LLM").language_model.model.layers[0].ffn.experts

        _, dense, dense_model = self._pair(snapshot, dtype, moe_experts="dense")
        _, packed, packed_model = self._pair(snapshot, dtype, moe_experts=moe_experts)
        assert {p.dtype for p in experts(dense_model).parameters()} == {dtype}
        assert next(experts(packed_model).parameters()).dtype == torch.uint8
        for rid in dense:
            assert torch.isfinite(dense[rid]).all() and torch.equal(packed[rid], dense[rid])

    def test_two_requests_in_one_step_match_each_alone(self, snapshot):
        ids, together, _ = self._pair(snapshot)
        assert logit_gap(together["a"], self._solo(snapshot, ids["a"], self.PROMPT)) < 1e-5
        assert logit_gap(together["b"], self._solo(snapshot, ids["b"], self.SHORT)) < 1e-5

    @pytest.mark.parametrize("cut", [1, 8, 19, 32])
    def test_a_prompt_fed_in_two_steps_matches_one(self, snapshot, cut):
        """A second prompt step reads the windows, open groups and n-gram tail the first left."""
        ids = random_ids(snapshot[1], self.PROMPT, seed=3)
        whole = self._solo(snapshot, ids, self.PROMPT)[0]
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        add_request(engine, "r")
        run_step(engine, sampler, "prefill", {"r": ids[:cut]})
        got = run_step(engine, sampler, "prefill", {"r": ids[cut:]})["r"]
        assert logit_gap(got[None], whole[None]) < 1e-5

    def test_a_decode_step_matches_a_prefill_of_the_same_tokens(self, snapshot):
        ids = random_ids(snapshot[1], self.PROMPT + self.STEPS, seed=4)
        stepped = self._solo(snapshot, ids, self.PROMPT)
        for step in (1, self.STEPS):
            whole = self._solo(snapshot, ids[:self.PROMPT + step], self.PROMPT + step)[0]
            assert logit_gap(stepped[step][None], whole[None]) < 1e-5

    def test_the_launch_sizes_a_deployment_sets_change_no_logit(self, snapshot):
        """``score_bytes`` and ``attend_bytes`` cut a step's scoring and attention into
        launches: a few rows a launch give what one launch gives, on prefill and decode rows."""
        ids_a = random_ids(snapshot[1], self.PROMPT + self.STEPS, seed=8)
        ids_b = random_ids(snapshot[1], 30 + self.STEPS, seed=9)

        def run(**sizes):
            engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
            for name, value in sizes.items():
                setattr(engine._resources[CSA].config, name, value)
            for rid in "ab":
                add_request(engine, rid)
            rows = [run_step(engine, sampler, "prefill", {"a": ids_a[:self.PROMPT], "b": ids_b[:30]})]
            for step in range(self.STEPS):
                rows.append(run_step(engine, sampler, "decode", {
                    "a": ids_a[self.PROMPT + step].view(1), "b": ids_b[30 + step].view(1)}))
            return torch.stack([torch.stack([r[rid] for r in rows]) for rid in "ab"])

        whole = run()
        # about three prefill rows and one decode row a launch at the reduced sizes
        pieces = run(score_bytes=4 * 16 * 3 * 22, attend_bytes=8 * 16 * 32 * 3)
        assert logit_gap(pieces, whole) < 1e-5

    def test_a_cache_holds_a_row_per_pooled_group_however_the_tokens_arrive(self, snapshot):
        """19 prompt tokens in steps of 1, 7 and 11, then 5 decode steps: 24 tokens, so 24 rows
        in the ratio-1 cache and 12 in the ratio-2 one, whose odd steps write nothing."""
        config = snapshot[1]
        assert config.source_ratios == (1, 2)
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        ids = random_ids(config, 24, seed=5)
        add_request(engine, "r")
        tokens = 0
        for walk, n in (("prefill", 1), ("prefill", 7), ("prefill", 11), *[("decode", 1)] * 5):
            run_step(engine, sampler, walk, {"r": ids[tokens:tokens + n]})
            tokens += n
            for ratio in config.source_ratios:
                stream = engine._resources[kv_cache_key(ratio)]._streams["r"]["main"]
                assert stream.stored_len == tokens // ratio, (walk, tokens, ratio)

    def test_a_finished_request_frees_its_state_and_pages(self, snapshot):
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        pool = engine._resources[STATE]
        caches = [engine._resources[kv_cache_key(ratio)] for ratio in snapshot[1].source_ratios]
        free = pool.num_free_slots
        add_request(engine, "r")
        run_step(engine, sampler, "prefill", {"r": random_ids(snapshot[1], 20, seed=5)})
        assert pool.num_free_slots == free - 1 and all(kv._streams["r"] for kv in caches)
        engine.remove_request("r")
        assert pool.num_free_slots == free and not any(kv._streams for kv in caches)
        # and its token count: a new request under the same id starts at position 0
        assert not engine._resources[CSA]._tokens

    def test_a_prompt_past_the_window_is_refused_before_and_at_the_engine(self, snapshot):
        engine, sampler, model = cpu_engine(snapshot[0], torch.float32, max_seq_len=64, tokenizer_mode="byte")
        # at the API: the client's 400
        with pytest.raises(ValueError, match="at most 62"):
            model.process_prompt("x" * 100, ["text"], ["text"])
        assert model.process_prompt("x" * 62, ["text"], ["text"]).new_input_tensors["text_inputs"][0].numel() == 62
        # at the engine, for a prompt that got past it: only that request fails
        add_request(engine, "long")
        add_request(engine, "short")
        with pytest.raises(AssertionError, match="'long'") as failed:
            run_step(engine, sampler, "prefill", {
                "long": random_ids(snapshot[1], 65, seed=6), "short": random_ids(snapshot[1], 8, seed=7)})
        assert "'short'" not in str(failed.value)


@pytest.fixture(scope="module")
def two_table_snapshot(tmp_path_factory):
    """The checkpoint's Engram shape: two layers, 2- to 4-grams."""
    path = tmp_path_factory.mktemp("dsv41-engram")
    config, _ = _testing.write_snapshot(path, DeepseekV41Config.reduced(
        engram_layer_ids=(0, 5), engram_num_embeddings=(0, 0), engram_max_ngram_size=4, engram_n_heads=3), seed=3)
    return path, config


@pytest.mark.usefixtures("cpu")
class TestEngramOnHost:
    """``engram_tables: host`` reads the same rows from the checkpoint's mapped bytes, so the
    logits are the device tables', bit for bit."""

    PROMPT, STEPS = 45, 6

    @pytest.mark.parametrize("tables", ["one-table", "two-tables"])
    def test_prefill_and_decode_give_the_device_tables_logits(self, snapshot, two_table_snapshot, tables):
        path, config = snapshot if tables == "one-table" else two_table_snapshot
        ids = random_ids(config, self.PROMPT + self.STEPS, seed=1)
        device = run_request(*cpu_engine(path, torch.float32)[:2], ids, self.PROMPT)
        host = run_request(*cpu_engine(path, torch.float32, engram_tables="host")[:2], ids, self.PROMPT)
        assert torch.equal(host, device)

    def test_other_rows_from_the_host_move_the_logits(self, snapshot, monkeypatch):
        """The check above can fail: the logits follow the rows the host hands back."""
        from mstar.engine.resources.host_table.manager import HostTableManager

        fetch = HostTableManager.fetch
        monkeypatch.setattr(HostTableManager, "fetch", lambda self, ids: fetch(self, {
            name: (rows + 1) % self.config.tables[name].rows for name, rows in ids.items()}))
        ids = random_ids(snapshot[1], self.PROMPT + self.STEPS, seed=1)
        device = run_request(*cpu_engine(snapshot[0], torch.float32)[:2], ids, self.PROMPT)
        host = run_request(*cpu_engine(snapshot[0], torch.float32, engram_tables="host")[:2], ids, self.PROMPT)
        assert logit_gap(host, device) > 1e-3

    def test_two_requests_in_one_step_give_the_device_tables_logits(self, snapshot):
        ids_a = random_ids(snapshot[1], self.PROMPT + self.STEPS, seed=1)
        ids_b = random_ids(snapshot[1], 22 + self.STEPS, seed=2)

        def both(**kwargs) -> torch.Tensor:
            engine, sampler, _ = cpu_engine(snapshot[0], torch.float32, **kwargs)
            for rid in "ab":
                add_request(engine, rid)
            rows = [run_step(engine, sampler, "prefill", {"a": ids_a[:self.PROMPT], "b": ids_b[:22]})]
            for step in range(self.STEPS):
                rows.append(run_step(engine, sampler, "decode", {
                    "a": ids_a[self.PROMPT + step].view(1), "b": ids_b[22 + step].view(1)}))
            return torch.stack([torch.stack([row["a"], row["b"]]) for row in rows])

        assert torch.equal(both(engram_tables="host"), both())

    @pytest.mark.parametrize("cut", [1, 19])
    def test_a_prompt_fed_in_two_steps_gives_the_device_tables_logits(self, snapshot, cut):
        ids = random_ids(snapshot[1], self.PROMPT, seed=3)

        def second_step(**kwargs) -> torch.Tensor:
            engine, sampler, _ = cpu_engine(snapshot[0], torch.float32, **kwargs)
            add_request(engine, "r")
            run_step(engine, sampler, "prefill", {"r": ids[:cut]})
            return run_step(engine, sampler, "prefill", {"r": ids[cut:]})["r"]

        assert torch.equal(second_step(engram_tables="host"), second_step())

    def test_the_tables_are_neither_allocated_nor_read_for_the_device(self, snapshot, monkeypatch):
        from mstar.model.loader import iterators

        read, shards = {}, iterators.iter_safetensors_shards

        def recording(*args, **kwargs):
            for name, tensor in shards(*args, **kwargs):
                read[name] = tensor.numel()
                yield name, tensor

        monkeypatch.setattr(iterators, "iter_safetensors_shards", recording)
        model = DeepseekV41Model(str(snapshot[0]), max_seq_len=256, engram_tables="host")
        lm = model.get_submodule("LLM").language_model
        assert not [name for name, _ in lm.named_parameters() if ".engram.table" in name]
        tables = {name: n for name, n in read.items() if ".engram.embed." in name}
        assert len(tables) == 2 and not any(tables.values())
        assert read["layers.1.engram.wkv.weight"]  # the rest of the layer still loads
        assert [spec.resource_key for spec in model.get_node_resources()][-1] == ENGRAM

    def test_a_serving_config_sizes_the_resource_only_under_the_host_placement(self, snapshot):
        from mstar.model.base import serving_resources

        block = {"resources": {ENGRAM: {"staging_bytes": 1 << 20, "prefault": False}}}
        spec = serving_resources(DeepseekV41Model(str(snapshot[0]), engram_tables="host"), block)[-1]
        assert (spec.resource_key, spec.config.staging_bytes, spec.config.prefault) == (ENGRAM, 1 << 20, False)
        # the device placement declares no such resource, so its block is a config error
        model = DeepseekV41Model(str(snapshot[0]))
        assert ENGRAM not in [spec.resource_key for spec in model.get_node_resources()]
        with pytest.raises(ValueError, match="unknown resource"):
            serving_resources(model, block)

    def test_another_placement_is_refused(self, snapshot):
        with pytest.raises(ValueError, match="engram_tables must be 'device' or 'host'"):
            DeepseekV41Model(str(snapshot[0]), engram_tables="pinned")

    def test_under_tp_each_rank_reads_whole_rows_and_sums_nothing(self, tmp_path):
        """On the device a rank holds a shard of the table and the ranks sum their lookups;
        from the host every rank gets whole rows, the same values, with no collective."""
        config = DeepseekV41Config.reduced()
        layer, rows = config.engram_layer_ids[0], config.engram_num_embeddings[0]
        g = torch.Generator().manual_seed(0)
        pair = _testing._fp8_pair(torch.randn(rows, config.engram_head_dim, generator=g), 1, config.fp8_block)
        ids = torch.randint(0, rows, (9, config.engram_hash_cols), generator=g)

        summed = 0
        for rank in range(2):
            group = CommGroup(rank, rank, [0, 1])
            group.all_reduce = lambda local: local  # this rank's term of the sum
            engram = Engram(config, layer, group)
            mine = slice(engram.row_start, engram.row_start + engram.rows)
            engram.table.data.copy_(pair["weight"].view(torch.uint8)[mine])
            engram.table_scale.data.copy_(pair["scale"].view(torch.uint8)[mine])
            summed = summed + engram.lookup(ids, torch.float32)

        save_file({f"{table_name(layer)}.{leaf}": tensor for leaf, tensor in pair.items()},
                  str(tmp_path / "model.safetensors"))
        on_host = dataclasses.replace(config, engram_tables="host")
        tables = build_resource(
            HostTableSpec(resource_key=ENGRAM, nodes={"LLM"}, config=HostTableConfig(
                checkpoint=str(tmp_path), tables=host_tables(on_host))),
            EngineResourceInfo(device=torch.device("cpu")))
        group = CommGroup(1, 1, [0, 1])

        def no_collective(local):
            raise AssertionError("a lookup from the host sums nothing across ranks")

        group.all_reduce = no_collective
        engram = Engram(on_host, layer, group)
        engram.bind_resources({ENGRAM: tables})
        tables.plan(HostTableStep(), StepContext(request_ids=(), graph_walk="prefill", slot=0, capture=False))
        tables.fetch({engram.host_table: ids})
        assert torch.equal(engram.lookup(ids, torch.float32), summed)


def test_the_tp8_config_starts_as_shipped_and_with_its_host_tables_switched_on(tmp_path):
    """The serving config carries the host placement as comments: taken out of them, the
    model declares the resource the block sizes, and the deployment still validates."""
    shipped = (Path(__file__).resolve().parents[2] / "configs" / "deepseek_v41_flash_tp8.yaml").read_text()
    switched_on = re.sub(
        r"(?m)^( *)# ( *)(engram_tables: host|engram:|staging_bytes: \d+|prefault: \w+)$", r"\1\2\3", shipped)
    for text, placement in ((shipped, "device"), (switched_on, "host")):
        cfg = yaml.safe_load(text)
        model = get_model_class(cfg["model"])(**{**model_init_kwargs(cfg["model"]), **cfg["model_kwargs"]})
        assert model.config.engram_tables == placement
        assert (ENGRAM in cfg["resources"]) == (placement == "host")
        path = tmp_path / f"{placement}.yaml"
        path.write_text(text)
        validate_deployment(model, cfg, str(path))
