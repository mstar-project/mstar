"""DeepSeek-V4.1's draft head without its reference: what ``draft_head`` loads and declares,
and the head in a CPU engine against itself. ``test_deepseek_v41_draft_parity.py`` holds it to
the checkpoint's ``forward_spec``.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
from safetensors.torch import load_file, save_file

from mstar.distributed.communication import CommGroup
from mstar.model.components.moe_parallel import MoeParallelConfig
from mstar.model.deepseek_v41 import _testing
from mstar.model.deepseek_v41._testing import add_request, cpu_engine, logit_gap, random_ids, record_drafts, run_step
from mstar.model.deepseek_v41.components.layers import DeepseekV41ForCausalLM
from mstar.model.deepseek_v41.config import MOE_EXPERTS, STATE, WINDOW, DeepseekV41Config
from mstar.model.deepseek_v41.deepseek_v41_model import DeepseekV41Model
from mstar.model.deepseek_v41.weight_loader import load_deepseek_v41_weights
from mstar.model.utils import check_loaded

PROMPT = 21


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    _testing.patch_cpu_engine(monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    path = tmp_path_factory.mktemp("dsv41-draft")
    config, _ = _testing.write_snapshot(path, DeepseekV41Config.reduced(), seed=0, draft_head=True)
    return path, config


def _rank(rank: int, tp: int) -> CommGroup:
    """Rank ``rank`` of a ``tp``-wide group that never communicates."""
    return CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(tp)))


class TestDraftConfig:
    def test_the_head_is_off_unless_asked_for(self):
        assert DeepseekV41Config().draft_head is False and DeepseekV41Config().draft_layers == ()
        assert DeepseekV41Config(draft_head=True).draft_layers == (40, 41, 42)

    @pytest.mark.parametrize("change, match", [
        (dict(num_nextn_predict_layers=0), "names no draft layers"),
        (dict(compress_ratios=(0, 2, 2, 2, 1, 1, 1, 0)), "attend a window only"),
        (dict(compress_ratios=(0, 2, 2, 2, 1, 1, 1, 0, 1, 0)), "attend a window only"),
        (dict(dspark_target_layer_ids=(5, 7)), "not backbone layers"),
        (dict(dspark_target_layer_ids=()), "not backbone layers"),
        (dict(dspark_noise_token_id=512), "not a token"),
        (dict(dspark_num_experts_per_tok=5), "within dspark_n_routed_experts"),
        (dict(draft_head="yes"), "true or false"),
    ])
    def test_a_checkpoint_without_the_head_asked_for_is_refused(self, change, match):
        with pytest.raises(ValueError, match=match):
            DeepseekV41Config.reduced(**{"draft_head": True, **change})

    def test_the_same_geometry_is_fine_with_the_head_off(self):
        DeepseekV41Config.reduced(compress_ratios=(0, 2, 2, 2, 1, 1, 1, 0), dspark_target_layer_ids=())


class TestDraftLoading:
    def test_off_nothing_of_the_head_is_built_or_declared(self, snapshot):
        """The default: a checkpoint that ships its draft layers loads as one without them."""
        engine, _, model = cpu_engine(snapshot[0], torch.float32)
        lm = model.get_submodule("LLM").language_model
        assert not any("mtp" in name for name, _ in lm.named_parameters())
        assert engine._resources[STATE].config.num_layers == snapshot[1].num_hidden_layers

    def test_on_each_draft_layer_keeps_its_window_in_the_pool(self, snapshot):
        engine, _, model = cpu_engine(snapshot[0], torch.float32, draft_head=True)
        config = model.config
        assert config.draft_layers == (7, 8, 9)
        pool = engine._resources[STATE]
        assert pool.config.num_layers == config.num_hidden_layers + 3
        assert pool.block(WINDOW, config.draft_layers[-1]).shape[1:] == (config.sliding_window, config.head_dim)

    def test_a_missing_draft_tensor_refuses_to_load(self, snapshot, tmp_path):
        path = tmp_path
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            if (snapshot[0] / name).is_file():
                (path / name).write_bytes((snapshot[0] / name).read_bytes())
        tensors = load_file(str(snapshot[0] / "model.safetensors"))
        del tensors["mtp.2.markov_head.head.weight"]
        save_file(tensors, str(path / "model.safetensors"))
        with pytest.raises(KeyError, match="missing 1 parameters"):
            DeepseekV41Model(str(path), max_seq_len=256, draft_head=True).get_submodule("LLM")
        DeepseekV41Model(str(path), max_seq_len=256).get_submodule("LLM")  # which only the head reads

    @pytest.mark.parametrize("layout, parallel", [*((layout, "tp") for layout in MOE_EXPERTS), ("fp4", "ep")])
    def test_a_rank_reads_its_slices_of_the_draft_layers(self, tmp_path, layout, parallel):
        """Two ranks through the shard reader fill what whole tensors fill, the Markov head by
        vocabulary and the draft experts as the backbone's."""
        from mstar.model.deepseek_v41.components.engram import compressed_token_map

        _testing.write_snapshot(
            tmp_path, DeepseekV41Config.reduced(moe_intermediate_size=64, o_lora_rank=32), seed=4, draft_head=True)
        whole = load_file(str(tmp_path / "model.safetensors"))
        moe_parallel = MoeParallelConfig(parallel=parallel)
        for rank in (0, 1):
            model = DeepseekV41Model(str(tmp_path), max_seq_len=256, moe_experts=layout, draft_head=True)
            got = model.get_submodule("LLM", tp_group=_rank(rank, 2), moe_parallel=moe_parallel).language_model
            token_map = torch.tensor(compressed_token_map(model.tokenizer)[0], dtype=torch.long)
            want = DeepseekV41ForCausalLM(model.config, token_map, comm_group=_rank(rank, 2), moe_parallel=moe_parallel)
            check_loaded(want, load_deepseek_v41_weights(want, whole.items()), "whole tensors")
            filled = dict(got.named_parameters())
            assert filled.keys() == dict(want.named_parameters()).keys()
            for name, param in want.named_parameters():
                assert torch.equal(filled[name], param), name
            vocab, rank_width = whole["mtp.2.markov_head.head.weight"].shape
            assert filled["model.mtp.2.markov_head.head.weight"].shape == (vocab // 2, rank_width)
            assert filled["model.mtp.2.markov_head.embed.weight"].shape == (vocab // 2, rank_width)
            assert filled["model.mtp.0.main_proj.weight"].shape == whole["mtp.0.main_proj.weight"].shape


def _drafts(snapshot, steps, dtype=torch.float32, **kwargs):
    """The blocks the head drafts over ``steps``: a list of ``(walk, {request: ids})``."""
    engine, sampler, model = cpu_engine(snapshot[0], dtype, draft_head=True, **kwargs)
    drafts = record_drafts(model)
    for rid in sorted({rid for _, inputs in steps for rid in inputs}):
        add_request(engine, rid)
    logits = [run_step(engine, sampler, walk, inputs) for walk, inputs in steps]
    return logits, drafts


class TestDraftEngineCpu:
    def test_a_step_drafts_a_block_for_each_request(self, snapshot):
        config = snapshot[1]
        ids = random_ids(config, PROMPT + 3, seed=1)
        steps = [("prefill", {"r": ids[:PROMPT]}), *(("decode", {"r": ids[PROMPT + i].view(1)}) for i in range(3))]
        _, drafts = _drafts(snapshot, steps)
        assert len(drafts) == 4
        for draft in drafts:
            assert draft.ids.shape == (1, config.dspark_block_size)
            assert draft.logits.shape == (1, config.dspark_block_size, config.vocab_size)
            assert draft.logits.dtype == draft.confidence.dtype == torch.float32
            assert torch.equal(draft.ids, draft.logits.argmax(-1))
            assert torch.isfinite(draft.confidence).all() and draft.confidence.shape == draft.ids.shape

    def test_two_requests_in_one_step_draft_what_each_drafts_alone(self, snapshot):
        config = snapshot[1]
        a, b = random_ids(config, PROMPT + 4, seed=1), random_ids(config, 9 + 4, seed=2)

        def steps(**prompts):
            ids = {"a": a, "b": b}
            return [("prefill", {rid: ids[rid][:n] for rid, n in prompts.items()}),
                    *(("decode", {rid: ids[rid][n + i].view(1) for rid, n in prompts.items()}) for i in range(4))]

        _, both = _drafts(snapshot, steps(a=PROMPT, b=9))
        _, alone_a = _drafts(snapshot, steps(a=PROMPT))
        _, alone_b = _drafts(snapshot, steps(b=9))
        for got, want_a, want_b in zip(both, alone_a, alone_b, strict=True):
            assert torch.equal(got.ids, torch.cat([want_a.ids, want_b.ids]))
            assert logit_gap(got.logits[0], want_a.logits[0]) < 1e-5
            assert logit_gap(got.logits[1], want_b.logits[0]) < 1e-5
            assert torch.allclose(got.confidence, torch.cat([want_a.confidence, want_b.confidence]), atol=1e-5)

    def test_the_block_after_a_decode_step_is_the_block_after_a_prefill_of_the_same_tokens(self, snapshot):
        """The draft layers' windows hold the same rows whether a prompt or decode steps brought them."""
        ids = random_ids(snapshot[1], PROMPT + 4, seed=3)
        _, stepped = _drafts(snapshot, [
            ("prefill", {"r": ids[:PROMPT]}), *(("decode", {"r": ids[PROMPT + i].view(1)}) for i in range(4))])
        _, whole = _drafts(snapshot, [("prefill", {"r": ids})])
        assert torch.equal(stepped[-1].ids, whole[0].ids)
        assert logit_gap(stepped[-1].logits, whole[0].logits) < 1e-5
        assert torch.allclose(stepped[-1].confidence, whole[0].confidence, atol=1e-5)

    def test_a_prompt_fed_in_two_steps_drafts_what_one_step_drafts(self, snapshot):
        ids = random_ids(snapshot[1], PROMPT, seed=4)
        _, cut = _drafts(snapshot, [("prefill", {"r": ids[:13]}), ("prefill", {"r": ids[13:]})])
        _, whole = _drafts(snapshot, [("prefill", {"r": ids})])
        assert torch.equal(cut[-1].ids, whole[0].ids)
        assert logit_gap(cut[-1].logits, whole[0].logits) < 1e-5

    def test_the_head_changes_no_logit_of_the_backbone(self, snapshot):
        ids = random_ids(snapshot[1], PROMPT + 2, seed=5)
        steps = [("prefill", {"r": ids[:PROMPT]}), *(("decode", {"r": ids[PROMPT + i].view(1)}) for i in range(2))]
        on, _ = _drafts(snapshot, steps)
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        add_request(engine, "r")
        for (walk, inputs), want in zip(steps, on, strict=True):
            assert torch.equal(run_step(engine, sampler, walk, inputs)["r"], want["r"])

    def test_a_finished_request_leaves_no_window_behind(self, snapshot):
        """The next request in the slot drafts as in a fresh engine."""
        config = snapshot[1]
        first, second = random_ids(config, PROMPT, seed=6), random_ids(config, 11, seed=7)
        engine, sampler, model = cpu_engine(snapshot[0], torch.float32, draft_head=True, max_requests=1)
        drafts = record_drafts(model)
        for rid, ids in (("a", first), ("b", second)):
            add_request(engine, rid)
            run_step(engine, sampler, "prefill", {rid: ids})
            engine.remove_request(rid)
        _, fresh = _drafts(snapshot, [("prefill", {"b": second})])
        assert torch.equal(drafts[1].logits, fresh[0].logits)
