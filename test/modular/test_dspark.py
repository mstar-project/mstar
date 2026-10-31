"""A DSpark drafter inside the verifying node, through the engine's own step path on CPU: the
toy trunk taps two of its layers for a randomly initialized ``DSparkDrafter``
(``dspark_toy``), which keeps its context as KV of its own, level with the target's, and drafts
each block in one pass.

Greedy verify must emit what plain greedy decode emits, whatever the drafter proposes, at every
draft count it can run, batched over prompts of different lengths; the drafter must draft at
every step what a drafter recomputed from scratch (float64, no cache, the taps from a dense
target pass) drafts, which only holds if its context KV took the right taps at the right
positions and the verdict rewound it; at temperature > 0 the stream follows the target's law.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_verify_toy as toy  # noqa: E402
import dspark_toy as ds  # noqa: E402
import test_draft_verify_sampled as sampled  # noqa: E402

from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.model.components.dspark import DSpark, DSparkConfig  # noqa: E402

CPU = torch.device("cpu")
PROMPTS = {0: [3, 9, 12, 40], 1: [5, 11], 2: [20, 21, 22, 23, 24, 25], 3: [60, 2, 33]}
MAX_TOKENS = 16


@pytest.fixture(autouse=True)
def _threads():
    # the comparisons are exact; keep the CPU reductions' order fixed
    torch.set_num_threads(2)


def _run(k: int, prompts=PROMPTS, max_tokens: int = MAX_TOKENS, batched: bool = True, log: bool = False, **sub_kwargs):
    sub = ds.ToyDSparkSubmodule(k=k, **sub_kwargs)
    eng = ds.engine(sub, CPU)
    drv = ds.Driver(eng, sub)
    drv.add_batch(prompts, max_tokens)
    if log:
        sub.draft_log = []
    out = drv.run(list(prompts), batched=batched)
    return out, eng, drv, sub


def _reference(prompts=PROMPTS, max_tokens: int = MAX_TOKENS):
    return {rid: toy.reference(p, max_tokens) for rid, p in prompts.items()}


@pytest.mark.parametrize("k", [1, 4, 7])
def test_greedy_verify_of_dspark_drafts_emits_what_plain_decode_emits(k):
    out, eng, drv, _ = _run(k)
    assert out == _reference()
    dv = eng._resources[toy.DV]
    assert dv.rows_verified > 0


def test_one_request_a_step_emits_what_plain_decode_emits():
    out, *_ = _run(3, batched=False)
    assert out == _reference()


def _drafts_against_reference(k: int, oracle: bool = False, **sub_kwargs):
    """Each verify step's drafts, row by row, against a drafter recomputed from scratch; returns
    (rows checked, rows equal, rows that parted where the reference's top two were within 1e-3).
    ``oracle``: some of each row's drafts replaced by the greedy stream's, so steps keep some."""
    sub = ds.ToyDSparkSubmodule(k=k, **sub_kwargs)
    eng = ds.engine(sub, CPU)
    drv = ds.Driver(eng, sub)
    drv.add_batch(PROMPTS, MAX_TOKENS)
    sub.draft_log = []
    if oracle:
        sub.oracle = {rid: p + toy.reference(p, MAX_TOKENS + k + 1, ignore_eos=True) for rid, p in PROMPTS.items()}
    live, checked, equal, ties = list(PROMPTS), 0, 0, 0
    while live:
        kept = {rid: PROMPTS[rid] + drv.emitted[rid] for rid in live}
        sub.kept.update({rid: len(tokens) for rid, tokens in kept.items()})
        n_before = len(sub.draft_log)
        drv.decode(live)
        if len(sub.draft_log) > n_before:
            anchors, drafts = sub.draft_log[-1]
            for row, rid in enumerate(live):
                assert int(anchors[row]) == kept[rid][-1]
                got = drafts[row].tolist()
                want, margins = ds.reference_drafts(sub.drafter, kept[rid], len(got))
                checked += 1
                if got == want:
                    equal += 1
                else:
                    first = next(i for i in range(len(got)) if got[i] != want[i])
                    ties += margins[first] < 1e-3
        live = [rid for rid in live if rid not in drv.stops]
    return checked, equal, ties, eng


def test_after_partial_acceptances_the_drafter_still_drafts_from_the_right_context():
    """Some drafts replaced by the greedy stream's, so steps keep 0 to k of them: the drafter's
    context must hold the kept rows' taps and nothing of the rejected ones."""
    checked, equal, ties, eng = _drafts_against_reference(4, oracle=True)
    hist = eng._resources[toy.DV].accepted_hist.tolist()
    assert sum(1 for count in hist[1:] if count) >= 3, hist
    assert equal + ties == checked > 10, (checked, equal, ties)


def test_both_kvs_stay_inside_their_footprints(monkeypatch, caplog):
    from mstar.engine.resources.kv import manager as kv_manager

    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    out, eng, *_ = _run(5)
    assert out == _reference()
    assert "against a footprint" not in caplog.text
    for key in (toy.KV, ds.DKV):
        for rid in PROMPTS:
            stream = eng._resources[key]._streams[rid]["main"]
            assert len(stream.page_indices) <= stream.footprint, (key, rid)


def test_without_the_drafters_kv_rewind_its_drafts_go_wrong(monkeypatch):
    real = KVManager.rewind

    def rewind(self, verdicts):
        if self.name != ds.DKV:
            real(self, verdicts)

    monkeypatch.setattr(KVManager, "rewind", rewind)
    checked, equal, ties, _ = _drafts_against_reference(3)
    assert equal + ties < checked


# ── temperature > 0: the sampled verify (DV-2) ──


def _sampled_streams(k: int, sample: bool, n: int, chunk: int = 256, max_tokens: int = 3):
    out, engines = {}, []
    for start in range(0, n, chunk):
        sub = ds.ToyDSparkSubmodule(k=k, sample=sample, max_num_pages=2048)
        eng = ds.engine(sub, CPU, sampled_verify=True, max_num_pages=2048)
        drv = ds.Driver(eng, sub)
        rids = list(range(start, min(start + chunk, n)))
        drv.add_batch({rid: sampled.PROMPT for rid in rids}, max_tokens, ignore_eos=True,
                      seeds={rid: 1 + rid for rid in rids},
                      temperature=sampled.TEMPERATURE, top_k=sampled.TOP_K)
        out.update(drv.run(rids, max_steps=400))
        engines.append(eng)
    return out, engines


@pytest.mark.parametrize(("k", "sample"), [(2, True), (3, False)], ids=["sampled_drafts", "argmax_drafts"])
def test_the_sampled_stream_follows_the_target(k, sample):
    """The joint law of the two tokens after the prefill's, over requests with their own seeds,
    against the float64 target: with the Markov head's drafts sampled as the request samples
    (each draft conditioned on the sampled one before it) and verified against their
    distributions, and with its argmax drafts."""
    n = 1536
    streams, engines = _sampled_streams(k, sample, n)
    t1, counts = sampled._first_two(streams)
    assert sampled._chi2(counts, sampled._expected(t1, n)) > sampled.P_MIN, counts
    rows = sum(eng._resources[toy.DV].rows_verified for eng in engines)
    assert rows > 0


# ── declarations and configs ──


QWEN3_8B = {  # deepseek-ai/dspark_qwen3_8b_block7's config.json
    "architectures": ["Qwen3DSparkModel"], "block_size": 7, "confidence_head_with_markov": True,
    "enable_confidence_head": True, "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096,
    "intermediate_size": 12288, "layer_types": ["full_attention"] * 5, "markov_head_type": "vanilla",
    "markov_rank": 256, "mask_token_id": 151669, "max_position_embeddings": 40960, "model_type": "qwen3",
    "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8, "num_target_layers": 36,
    "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "sliding_window": None, "target_layer_ids": [1, 9, 17, 25, 33], "use_sliding_window": False,
    "vocab_size": 151936,
}


def test_configs_read_from_each_checkpoint_format():
    q = DSparkConfig.from_hf(QWEN3_8B)
    assert q.taps == (2, 10, 18, 26, 34) and q.block_size == 7 and not q.causal
    assert q.backbone == "qwen3" and q.rope_scaling is None and q.rope_theta == 1e6
    decl = DSpark.for_target(q, 7, max_seq_len=8192, target_layers=36, target_hidden_size=4096)
    assert decl.policy_fields() == {"taps": (2, 10, 18, 26, 34)}
    assert decl.verify_steps(7)["dspark_attn"].hide_last == 1 and "dspark_attn" not in decl.verify_steps(0)
    with pytest.raises(ValueError, match="block_size"):
        DSpark.for_target(q, 8, max_seq_len=8192, target_layers=36, target_hidden_size=4096)
    with pytest.raises(ValueError, match="target layers"):
        DSpark.for_target(q, 7, max_seq_len=8192, target_layers=28, target_hidden_size=4096)
    # DeepSeek-V4.1: the target's own config; MTP blocks, the family's
    v41 = DSparkConfig.from_hf({"model_type": "deepseek_v41", "text_config": {
        "hidden_size": 5120, "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
        "vocab_size": 129280, "num_nextn_predict_layers": 3, "dspark_block_size": 5,
        "dspark_noise_token_id": 128799, "dspark_target_layer_ids": [37, 38, 39], "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3, "sliding_window": 128,
        "moe_intermediate_size": 2304,
    }})
    assert v41.taps == (37, 38, 39) and v41.num_layers == 3 and v41.block_size == 5
    assert (v41.num_experts, v41.num_experts_per_tok, v41.sliding_window) == (128, 3, 128)
    assert v41.backbone == "family"
    # Kimi K3, Qwen3-style dense layers with YaRN (RadixArk/Kimi-K3-DSpark)
    k3 = DSparkConfig.from_hf({
        "markov_rank": 256, "model_type": "qwen3", "hidden_size": 7168, "num_hidden_layers": 5,
        "num_attention_heads": 64, "num_key_value_heads": 16, "head_dim": 64, "intermediate_size": 14336,
        "vocab_size": 163840, "block_size": 7, "enable_confidence_head": True, "confidence_head_with_markov": True,
        "dflash_config": {"mask_token_id": 163824, "target_layer_ids": [7, 23, 51, 67, 83]},
        "rope_parameters": {"factor": 16.0, "original_max_position_embeddings": 65536, "rope_theta": 10000.0,
                            "rope_type": "yarn"},
    })
    assert k3.taps == (8, 24, 52, 68, 84) and k3.mask_token_id == 163824 and k3.rope_scaling["rope_type"] == "yarn"
    with pytest.raises(ValueError, match="rope_scaling"):
        DSpark.for_target(k3, 7, max_seq_len=8192, target_layers=93, target_hidden_size=7168)
    # the speculators format: taps already name the stream entering each layer
    spec = DSparkConfig.from_hf({
        "speculators_model_type": "dspark", "aux_hidden_state_layer_ids": [24, 48, 72, 88, 92], "block_size": 8,
        "markov_rank": 256, "mask_token_id": 163837, "enable_confidence_head": True,
        "transformer_layer_config": {k: v for k, v in QWEN3_8B.items() if k not in (
            "target_layer_ids", "markov_rank", "mask_token_id", "block_size")},
    })
    assert spec.taps == (24, 48, 72, 88, 92) and spec.block_size == 8

