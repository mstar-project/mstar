"""GLM-5.2 MTP as one captured graph per step, on one GPU: the reduced model at the real MLA
latent dims (FlashInfer's MLA kernel), captured through the engine's ``CudaGraphRunner`` and
replayed the way the engine replays it, against the same model eager and plain decode."""
import pytest
import torch
from test_glm52_capture_gpu import PROMPTS, _load, _Node

from mstar.engine.cuda_graph_runner import CudaGraphRunner
from mstar.model.glm52.submodules import Glm52LLMSubmodule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU",
)

MAX_TOKENS = 12


class _MtpNode(_Node):
    """``_Node`` plus the engine's host cut of a verify step (``unpack_packed_outputs``), and
    streams that take every token a step emits."""

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
        unpacked = self.sub.unpack_packed_outputs(
            static_output={}, request_ids=list(batch), real_seq_lens=[], inputs=[],
            per_request_info={rid: info for rid, (info, _) in batch.items()},
        )
        for rid, rid_out in unpacked.items():
            outs[rid].update(rid_out)
        return outs

    def generate(self) -> dict[str, list[int]]:
        infos = {rid: self.info(rid) for rid in self.prompts}
        for rid in self.prompts:
            self.runner.ingest_request(rid, self.overrides)
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
                if self.sub.check_stop(rid, infos[rid], out):
                    live.remove(rid)
        return streams


@pytest.fixture
def eager_capture(monkeypatch):
    # capture the eager forward: these streams are compared with eager, and Inductor fuses
    # bf16 chains without eager's intermediate rounding (the compiled capture has its own test)
    monkeypatch.setenv("MSTAR_GLM52_GRAPH_COMPILE", "0")


def _mtp_cfg(k: int) -> dict:
    # the MTP layer's position lands FULL (4 = offset-1 + freq)
    return {"mtp_num_draft_tokens": k, "num_hidden_layers": 4}


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("oracle", [False, True], ids=["mtp", "seed_oracle"])
def test_captured_mtp_matches_eager_and_plain_decode(tmp_path, monkeypatch, eager_capture, k, oracle):
    """Two requests decoded as one batch, captured at bs 1/2/4 (the batch runs the bs-2
    bucket; the single prefills bs 1): captured == eager MTP == plain decode, token for token.
    With the seed oracle every step's d1 is right, so replays verify accepted rows too."""
    monkeypatch.setattr(CudaGraphRunner, "CAPTURE_BATCH_SIZES", [1, 2, 4])
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 2, 4])
    model, submodule = _load(tmp_path, monkeypatch, fp8=False, cfg_overrides=_mtp_cfg(0))
    plain = _Node(model, submodule, capture=False, prompts=PROMPTS)
    reference = plain.generate()
    plain.close()

    streams = {}
    for capture in (False, True):
        ckpt = tmp_path / f"k{k}-{int(capture)}"
        ckpt.mkdir()
        model, submodule = _load(ckpt, monkeypatch, fp8=False, cfg_overrides=_mtp_cfg(k))
        node = _MtpNode(model, submodule, capture=capture, prompts=PROMPTS,
                        seed_oracle=reference if oracle else None)
        if capture:
            assert node.cg.any_graphs and node.cg.dropped_buckets == []
        streams[capture] = node.generate()
        accepted = submodule._mtp_stat_emitted - submodule._mtp_stat_steps
        node.close()
        if oracle:
            assert accepted > 0, "the seed oracle's d1 never got accepted"

    for rid, ref in reference.items():
        assert len(ref) == MAX_TOKENS and len(set(ref)) > 2, (rid, ref)
        assert streams[False][rid] == ref, ("eager", rid, streams[False][rid], ref)
        assert streams[True][rid] == ref, ("captured", rid, streams[True][rid], ref)


def test_padded_capture_rows_leave_real_seeds_alone(tmp_path, monkeypatch, eager_capture):
    """A batch of two replayed in the bs-4 bucket: the padding rows write the sink slot, so
    the real requests' seeds and streams are the unpadded ones."""
    monkeypatch.setattr(CudaGraphRunner, "CAPTURE_BATCH_SIZES", [1, 4])
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 4])
    streams = {}
    for capture in (False, True):
        ckpt = tmp_path / f"pad{int(capture)}"
        ckpt.mkdir()
        model, submodule = _load(ckpt, monkeypatch, fp8=False, cfg_overrides=_mtp_cfg(3))
        node = _MtpNode(model, submodule, capture=capture, prompts=PROMPTS)
        streams[capture] = node.generate()
        node.close()
    assert streams[True] == streams[False]


def test_compiled_capture_matches_compiled_plain_decode(tmp_path, monkeypatch):
    """The serving configuration: the captured forward compiled (Inductor), MTP against plain
    decode compiled the same way."""
    monkeypatch.setattr(CudaGraphRunner, "CAPTURE_BATCH_SIZES", [1, 2])
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
