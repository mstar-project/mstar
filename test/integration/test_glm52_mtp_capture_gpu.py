"""GLM-5.2 MTP as one captured graph per step, on one GPU: the reduced model at the real MLA
latent dims (FlashInfer's MLA kernel), captured through the engine's ``AcceleratorGraphRunner`` and
replayed the way the engine replays it, against the same model eager and plain decode; and at a
temperature, against plain decode's samples."""
from collections import Counter

import pytest
import test_glm52_capture_gpu as capture_gpu
import torch
from test_glm52_capture_gpu import PROMPTS, _load, _Node

from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.resources import DRAFT_VERIFY
from mstar.model.glm52.config import KV_RESOURCE, SAMPLER_RESOURCE
from mstar.model.glm52.submodules import Glm52LLMSubmodule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU",
)

MAX_TOKENS = 12


class _MtpNode(_Node):
    """``_Node`` plus the engine's cut of a verify step's tokens and its stop
    (``Engine._emit_verified``), and streams that take every token a step emits."""

    def __init__(self, *args, seed_oracle: dict[str, list[int]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # the right d1 written into each request's seed before every decode step: the
        # captured drafts are the graph's own, so this is how a replay sees accepted rows
        self.seed_oracle = seed_oracle

    def info(self, rid):
        info = super().info(rid)
        info.rid_handle = rid  # the worker stamps it; MTP state is keyed by it
        return info

    def step(self, walk, batch):
        outs = super().step(walk, batch)
        policy = self.sub.get_draft_policy(walk)
        self.stops = set()
        if policy is not None:
            rids = list(batch)
            infos = [info for info, _ in batch.values()]
            rows = self.resources[DRAFT_VERIFY].emitted_rows(
                rids, [info.max_tokens for info in infos],
                [info.resource_configs[SAMPLER_RESOURCE].ignore_eos for info in infos], policy,
            )
            for rid, tokens in zip(rids, torch.from_numpy(rows.tokens).split(rows.counts.tolist()), strict=True):
                outs[rid]["new_token"] = [tokens]
            self.stops = {rid for rid, done in zip(rids, rows.done.tolist(), strict=True) if done}
        return outs

    def generate(self) -> dict[str, list[int]]:
        infos = {rid: self.info(rid) for rid in self.prompts}
        for rid in self.prompts:
            self.runner.ingest_request(rid, self.request_configs.get(rid, self.overrides))
        streams: dict[str, list[int]] = {rid: [] for rid in self.prompts}
        texts = {}
        for rid, prompt in self.prompts.items():
            ids = torch.tensor(prompt, dtype=torch.long, device="cuda:0")
            out = self.step("prefill", {rid: (infos[rid], ids)})[rid]
            self.sub.postprocess(rid, infos[rid], out)
            streams[rid] += out["new_token"][0].tolist()
            texts[rid] = out["text_inputs"][0].clone()
            assert not self.sub.check_stop(rid, infos[rid], out)
        live = list(self.prompts)
        for _ in range(MAX_TOKENS + 2):
            if not live:
                break
            for rid in live:
                infos[rid].graph_walk = "decode"
                if self.seed_oracle is not None:
                    ref = self.seed_oracle[rid]
                    nxt = len(streams[rid])
                    self.sub._mtp_seed_draft[self.sub._mtp_slot[rid]] = ref[nxt] if nxt < len(ref) else 0
            outs = self.step("decode", {rid: (infos[rid], texts[rid]) for rid in live})
            for rid in list(live):
                out = outs[rid]
                self.sub.postprocess(rid, infos[rid], out)
                streams[rid] += out["new_token"][0].tolist()
                texts[rid] = out["text_inputs"][0].clone()
                if rid in self.stops:
                    live.remove(rid)
        return streams


@pytest.fixture
def eager_capture():
    # capture the eager forward: these streams are compared with eager, and Inductor fuses
    # bf16 chains without eager's intermediate rounding (the compiled capture has its own test)
    return {"graph_compile": False}


def _mtp_cfg(k: int) -> dict:
    # the MTP layer's position lands FULL (4 = offset-1 + freq)
    return {"mtp_num_draft_tokens": k, "num_hidden_layers": 4}


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("oracle", [False, True], ids=["mtp", "seed_oracle"])
def test_captured_mtp_matches_eager_and_plain_decode(tmp_path, monkeypatch, eager_capture, k, oracle):
    """Two requests decoded as one batch, captured at bs 1/2/4 (the batch runs the bs-2
    bucket; the single prefills bs 1): captured == eager MTP == plain decode, token for token.
    With the seed oracle every step's d1 is right, so replays verify accepted rows too."""
    monkeypatch.setattr(AcceleratorGraphRunner, "CAPTURE_BATCH_SIZES", [1, 2, 4])
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 2, 4])
    model, submodule = _load(tmp_path, monkeypatch, fp8=False, cfg_overrides={**_mtp_cfg(0), **eager_capture})
    plain = _Node(model, submodule, capture=False, prompts=PROMPTS)
    reference = plain.generate()
    plain.close()

    streams = {}
    for capture in (False, True):
        ckpt = tmp_path / f"k{k}-{int(capture)}"
        ckpt.mkdir()
        model, submodule = _load(ckpt, monkeypatch, fp8=False, cfg_overrides={**_mtp_cfg(k), **eager_capture})
        node = _MtpNode(model, submodule, capture=capture, prompts=PROMPTS,
                        seed_oracle=reference if oracle else None)
        if capture:
            assert node.cg.any_graphs and node.cg.dropped_buckets == []
        streams[capture] = node.generate()
        accepted = node.resources[DRAFT_VERIFY].drafts_accepted
        node.close()
        if oracle:
            assert accepted > 0, "the seed oracle's d1 never got accepted"

    for rid, ref in reference.items():
        assert len(ref) == MAX_TOKENS and len(set(ref)) > 2, (rid, ref)
        assert streams[False][rid] == ref, ("eager", rid, streams[False][rid], ref)
        assert streams[True][rid] == ref, ("captured", rid, streams[True][rid], ref)


TEMPERATURE, TOP_K = 0.7, 4


def _sampled(node, first_seed: int) -> None:
    """Every request of ``node`` at TEMPERATURE / TOP_K, each with its own seed."""
    for i, rid in enumerate(node.prompts):
        configs = node.model.get_request_resource_configs(
            {}, model_kwargs={"temperature": TEMPERATURE, "top_k": TOP_K, "ignore_eos": True},
        )
        configs[SAMPLER_RESOURCE].apply_conductor_config(seed=first_seed + i, resource_key=SAMPLER_RESOURCE)
        node.request_configs[rid] = configs


def _copies(n: int) -> dict[str, list[int]]:
    """``n`` requests of each prompt, named prompt-copy."""
    return {f"{rid}-{i}": prompt for i in range(n) for rid, prompt in PROMPTS.items()}


def test_mtp_at_a_temperature_keeps_drafts_and_replays_as_eager(tmp_path, monkeypatch, eager_capture):
    """At temperature 0.7 (top-k 4), each request its own seed, with the seed oracle's d1 (the
    greedy stream's next token, which the target holds while the sampled stream is still on
    the greedy one): drafts are kept, the streams are sampled, not greedy, and the captured
    steps (prefill and decode) draw what the eager ones draw."""
    monkeypatch.setattr(AcceleratorGraphRunner, "CAPTURE_BATCH_SIZES", [1, 16, 32])
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 16, 32])
    monkeypatch.setattr(capture_gpu, "KV_YAML", {"resources": {KV_RESOURCE: {"max_num_pages": 256, "page_size": 16}}})
    model, submodule = _load(tmp_path, monkeypatch, fp8=False, cfg_overrides={**_mtp_cfg(0), **eager_capture})
    plain = _Node(model, submodule, capture=False, prompts=PROMPTS)
    greedy = plain.generate()
    plain.close()
    # the reduced model's top 4 are near even at 0.7, so a few percent of rows keep a draft:
    # 32 requests make that several
    prompts = _copies(16)
    oracle = {rid: greedy[rid.split("-")[0]] for rid in prompts}
    streams = {}
    for capture in (False, True):
        ckpt = tmp_path / f"mtp{int(capture)}"
        ckpt.mkdir()
        model, submodule = _load(ckpt, monkeypatch, fp8=False, cfg_overrides={**_mtp_cfg(3), **eager_capture})
        node = _MtpNode(model, submodule, capture=capture, prompts=prompts, seed_oracle=oracle)
        if capture:
            assert node.cg.any_graphs and node.cg.dropped_buckets == []
        _sampled(node, 100)
        streams[capture] = node.generate()
        dv = node.resources[DRAFT_VERIFY]
        node.close()
        assert 0 < dv.drafts_accepted < 3 * dv.rows_verified, dv.stats()
    assert all(len(stream) == MAX_TOKENS for stream in streams[False].values()), streams
    assert any(stream != oracle[rid] for rid, stream in streams[False].items()), "every stream was the greedy one"
    assert streams[True] == streams[False]


def test_mtp_samples_what_plain_decode_samples(tmp_path, monkeypatch, eager_capture):
    """The first two tokens of many requests at temperature 0.7 (top-k 4), each request its
    own seed: MTP (k = 3, the seed oracle's d1) against plain decode, as a two-sample
    chi-square over the token pairs. Eager, so a step takes any batch."""
    monkeypatch.setattr(capture_gpu, "KV_YAML", {"resources": {KV_RESOURCE: {"max_num_pages": 2200, "page_size": 16}}})
    monkeypatch.setattr(capture_gpu, "MAX_TOKENS", 2)
    monkeypatch.setitem(globals(), "MAX_TOKENS", 2)
    n = 512
    ckpt = tmp_path / "plain"
    ckpt.mkdir()
    model, submodule = _load(ckpt, monkeypatch, fp8=False, cfg_overrides={**_mtp_cfg(0), **eager_capture})
    plain = _Node(model, submodule, capture=False, prompts=PROMPTS)
    greedy = plain.generate()
    plain.close()
    nodes = {0: _Node(model, submodule, capture=False, prompts=_copies(n))}
    ckpt = tmp_path / "mtp"
    ckpt.mkdir()
    model, submodule = _load(ckpt, monkeypatch, fp8=False, cfg_overrides={**_mtp_cfg(3), **eager_capture})
    prompts = _copies(n)
    oracle = {rid: greedy[rid.split("-")[0]] for rid in prompts}
    nodes[3] = _MtpNode(model, submodule, capture=False, prompts=prompts, seed_oracle=oracle)
    pairs = {}
    for k, node in nodes.items():
        # disjoint seeds, so the two arms are independent samples
        _sampled(node, 10_000 * (k + 1))
        streams = node.generate()
        if k:
            accepted = node.resources[DRAFT_VERIFY].drafts_accepted
        node.close()
        pairs[k] = {
            name: Counter(tuple(s[:2]) for rid, s in streams.items() if rid.startswith(name + "-"))
            for name in PROMPTS
        }
    assert accepted > 0
    for name in PROMPTS:
        cells = sorted(set(pairs[0][name]) | set(pairs[3][name]))
        table = torch.tensor([[pairs[k][name][c] for c in cells] for k in (0, 3)], dtype=torch.float64)
        assert _two_sample_p(table) > 1e-4, (name, cells, table)


def _two_sample_p(table: torch.Tensor) -> float:
    """The chi-square homogeneity p-value of a 2 x C count table, columns expecting fewer
    than 5 pooled."""
    expected = table.sum(1, keepdim=True) * table.sum(0, keepdim=True) / table.sum()
    small = expected.min(0).values < 5
    if small.any():
        table = torch.cat([table[:, ~small], table[:, small].sum(1, keepdim=True)], 1)
        expected = table.sum(1, keepdim=True) * table.sum(0, keepdim=True) / table.sum()
    stat = ((table - expected) ** 2 / expected).sum()
    df = table.shape[1] - 1
    if df == 0:
        return 1.0
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64), stat / 2))


def test_padded_capture_rows_leave_real_seeds_alone(tmp_path, monkeypatch, eager_capture):
    """A batch of two replayed in the bs-4 bucket: the padding rows write the sink slot, so
    the real requests' seeds and streams are the unpadded ones."""
    monkeypatch.setattr(AcceleratorGraphRunner, "CAPTURE_BATCH_SIZES", [1, 4])
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 4])
    streams = {}
    for capture in (False, True):
        ckpt = tmp_path / f"pad{int(capture)}"
        ckpt.mkdir()
        model, submodule = _load(ckpt, monkeypatch, fp8=False, cfg_overrides={**_mtp_cfg(3), **eager_capture})
        node = _MtpNode(model, submodule, capture=capture, prompts=PROMPTS)
        streams[capture] = node.generate()
        node.close()
    assert streams[True] == streams[False]


def test_compiled_capture_matches_compiled_plain_decode(tmp_path, monkeypatch):
    """The serving configuration: the captured forward compiled (Inductor), MTP against plain
    decode compiled the same way."""
    monkeypatch.setattr(AcceleratorGraphRunner, "CAPTURE_BATCH_SIZES", [1, 2])
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 2])
    streams = {}
    for k in (0, 3):
        ckpt = tmp_path / f"compiled{k}"
        ckpt.mkdir()
        model, submodule = _load(ckpt, monkeypatch, fp8=False, cfg_overrides=_mtp_cfg(k))
        node = (_MtpNode if k else _Node)(model, submodule, capture=True, prompts=PROMPTS)
        assert node.cg.any_graphs and node.cg.dropped_buckets == []
        streams[k] = node.generate()
        node.close()
    assert streams[3] == streams[0]
