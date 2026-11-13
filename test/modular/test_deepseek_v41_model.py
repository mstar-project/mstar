"""DeepSeek-V4.1 without its reference: the checkpoint's config and tensor layout against the
model's parameters, the loader's guards, and the model in a CPU engine against itself.

``fixtures/deepseek_v41_config.json`` is the checkpoint's own;
``fixtures/deepseek_v41_checkpoint_layout.json`` lists its 96,085 tensors by name, dtype and
shape, read from the 48 shard headers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest
import torch
from safetensors.torch import load_file, save_file

from mstar.model.deepseek_v41 import _testing
from mstar.model.deepseek_v41._testing import add_request, cpu_engine, logit_gap, random_ids, run_request, run_step
from mstar.model.deepseek_v41.components.engram import bucket_primes, table_rows
from mstar.model.deepseek_v41.components.layers import DeepseekV41ForCausalLM
from mstar.model.deepseek_v41.config import KV_CACHE, STATE, DeepseekV41Config
from mstar.model.deepseek_v41.deepseek_v41_model import DeepseekV41Model
from mstar.model.deepseek_v41.weight_loader import load_deepseek_v41_weights, skipped, upcast_fp4_block
from mstar.model.utils import check_loaded
from mstar.utils.mx_round import dequant_row_fp4, pack_e2m1, pow2_ceil, round_e2m1

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


class TestCheckpointConfig:
    def test_the_checkpoints_config_parses_to_the_defaults(self):
        assert _checkpoint_config() == DeepseekV41Config()

    def test_round_trip_through_the_hf_spelling(self):
        config = DeepseekV41Config.reduced()
        read = DeepseekV41Config.from_hf_config(_testing.to_hf_config(config))
        for knob in ("max_seq_len", "moe_fp8_resident", "max_position_embeddings"):
            setattr(read, knob, getattr(config, knob))
        assert read == config

    def test_layer_sources(self):
        config = DeepseekV41Config()
        assert [config.kv_source_of(i) for i in (0, 1, 2, 7, 8, 19, 20, 39)] == [None, None, 2, 2, 8, 14, 20, 20]
        assert config.pooled_sources == (2, 8, 14) and config.num_planes == 4
        assert [config.uses_candidates(i) for i in (20, 24)] == [False, True]

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
    every parameter is filled: the full-size model on ``meta``, the tensors too."""

    @pytest.mark.parametrize("fp8_resident", [False, True])
    def test_full_size_model_loads_every_tensor_by_name_and_shape(self, fp8_resident):
        config = _checkpoint_config()
        config.moe_fp8_resident = fp8_resident
        token_map = torch.arange(config.vocab_size) % config.engram_compressed_vocab_size
        with torch.device("meta"):
            lm = DeepseekV41ForCausalLM(config, token_map)
        served = [(n, d, s) for n, d, s in _checkpoint_tensors() if not skipped(n)]
        weights = ((name, torch.empty(shape, dtype=dtype, device="meta")) for name, dtype, shape in served)
        check_loaded(lm, load_deepseek_v41_weights(lm, weights), "checkpoint layout")

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
            DeepseekV41Model(str(path), max_seq_len=256, moe_fp8_resident=False).get_submodule("LLM")

    def test_one_missing_expert_matrix_refuses_to_load(self, snapshot, tmp_path):
        def drop(tensors):
            del tensors["layers.2.ffn.experts.5.w3.weight"], tensors["layers.2.ffn.experts.5.w3.scale"]

        with pytest.raises(KeyError, match="gate_up_proj"):
            DeepseekV41Model(str(_variant(snapshot, tmp_path, drop)), max_seq_len=256,
                             moe_fp8_resident=False).get_submodule("LLM")

    def test_a_weight_without_its_scale_refuses_to_load(self, snapshot, tmp_path):
        path = _variant(snapshot, tmp_path, lambda t: t.pop("layers.0.attn.wq_b.scale"))
        with pytest.raises(KeyError, match="without their weight or scale"):
            DeepseekV41Model(str(path), max_seq_len=256, moe_fp8_resident=False).get_submodule("LLM")

    def test_an_unknown_tensor_refuses_to_load(self, snapshot, tmp_path):
        path = _variant(snapshot, tmp_path, lambda t: t.update({"layers.0.attn.extra.weight": torch.zeros(2, 2)}))
        with pytest.raises(KeyError, match="has no parameter"):
            DeepseekV41Model(str(path), max_seq_len=256, moe_fp8_resident=False).get_submodule("LLM")

    def test_another_tokenizer_than_the_tables_were_hashed_with_is_refused(self, snapshot, tmp_path):
        path = _variant(snapshot, tmp_path, lambda t: None)
        hf = json.loads((path / "config.json").read_text())
        hf["text_config"]["engram_compressed_vocab_size"] += 1
        (path / "config.json").write_text(json.dumps(hf))
        with pytest.raises(ValueError, match="tables were hashed over"):
            DeepseekV41Model(str(path), max_seq_len=256, moe_fp8_resident=False).get_submodule("LLM")


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


@pytest.fixture
def cpu(monkeypatch):
    _testing.patch_cpu_engine(monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


@pytest.mark.usefixtures("cpu")
class TestEngineCpu:
    PROMPT, STEPS = 45, 6

    def _solo(self, snapshot, ids, prompt, **kwargs):
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32, **kwargs)
        return run_request(engine, sampler, ids, prompt)

    def test_fp8_resident_experts_give_the_dequantized_experts_logits(self, snapshot):
        ids = random_ids(snapshot[1], self.PROMPT + self.STEPS, seed=1)
        dense = self._solo(snapshot, ids, self.PROMPT)
        resident = self._solo(snapshot, ids, self.PROMPT, moe_fp8_resident=True)
        assert torch.equal(resident, dense)

    def test_two_requests_in_one_step_match_each_alone(self, snapshot):
        ids_a = random_ids(snapshot[1], self.PROMPT + self.STEPS, seed=1)
        ids_b = random_ids(snapshot[1], 22 + self.STEPS, seed=2)
        alone_a, alone_b = self._solo(snapshot, ids_a, self.PROMPT), self._solo(snapshot, ids_b, 22)
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        for rid in "ab":
            add_request(engine, rid)
        rows = [run_step(engine, sampler, "prefill", {"a": ids_a[:self.PROMPT], "b": ids_b[:22]})]
        for step in range(self.STEPS):
            rows.append(run_step(engine, sampler, "decode", {
                "a": ids_a[self.PROMPT + step].view(1), "b": ids_b[22 + step].view(1)}))
        assert logit_gap(torch.stack([r["a"] for r in rows]), alone_a) < 1e-5
        assert logit_gap(torch.stack([r["b"] for r in rows]), alone_b) < 1e-5

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

    def test_a_finished_request_frees_its_state_and_pages(self, snapshot):
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        pool, kv = engine._resources[STATE], engine._resources[KV_CACHE]
        free = pool.num_free_slots
        add_request(engine, "r")
        run_step(engine, sampler, "prefill", {"r": random_ids(snapshot[1], 20, seed=5)})
        assert pool.num_free_slots == free - 1 and kv._streams["r"]["main"].stored_len == 20
        engine.remove_request("r")
        assert pool.num_free_slots == free and not kv._streams

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
