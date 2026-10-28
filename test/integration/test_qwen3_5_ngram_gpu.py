"""Qwen3.5 with n-gram drafting through the engine's step path, on real weights, on a GPU.

Greedy decode with drafts must emit what plain greedy decode emits: every verify step runs
the trunk over a block of tokens (attention over the paged KV, the gated delta rule from each
slot's state without writing it), keeps the drafts the model agrees with, rewinds the KV and
positions past them and settles each GDN layer's state and conv window after the kept prefix.
Code-editing prompts, whose replies copy their prompt, so most steps keep drafts.

bf16 reductions over a block of rows tile differently from one row, so a token whose top two
logits tie in plain decode (bf16 holds a logit near 16 to 1/16) can come out the other way.
So the streams must agree up to the first such tie: a stream that leaves plain decode where
plain decode's choice was clear is wrong (a missed rewind does that within a few tokens of
the first rejected draft).

Needs CUDA, FlashInfer and the cached Qwen/Qwen3.5-0.8B checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("flashinfer")


def _checkpoint() -> str | None:
    if os.environ.get("QWEN3_5_CKPT"):
        return os.environ["QWEN3_5_CKPT"]
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    snaps = sorted((home / "hub/models--Qwen--Qwen3.5-0.8B/snapshots").glob("*/model.safetensors*"))
    return str(snaps[0].parent) if snaps else None


CKPT = _checkpoint()
if CKPT is None:
    pytest.skip("Qwen/Qwen3.5-0.8B is not cached", allow_module_level=True)

from mstar.communication.tensors import LocalTransferEngine  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.communication import CommGroup, JointGroups  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import NGRAM, SamplingReqConfig, StepContext  # noqa: E402
from mstar.engine.resources.draft_verify.config import DRAFT_VERIFY  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig  # noqa: E402
from mstar.engine.resources.kv.transfer import TransferEngineInfo  # noqa: E402
from mstar.engine.resources.sampler.resource import SamplerResource  # noqa: E402
from mstar.model.qwen3_5.config import KV_CACHE, SAMPLER  # noqa: E402
from mstar.model.qwen3_5.qwen3_5_model import TEXT_PART, Qwen3_5DenseModel  # noqa: E402
from mstar.model.qwen3_5.submodules import LLMSubmodule  # noqa: E402

GPU = torch.device("cuda", 0)
MAX_TOKENS = 96
# a plain-decode choice whose top two logits are this close may flip under another tiling
TIE = 0.25
CODE = '''def tokens_per_step(rows, accepted):
    total = 0
    for row in rows:
        total += accepted[row] + 1
    return total / len(rows)


def acceptance(rows, accepted, drafts):
    total = 0
    for row in rows:
        total += accepted[row]
    return total / (len(rows) * drafts)
'''
PROMPTS = [
    f"Rename `rows` to `batch` everywhere in this code and reply with the whole code only:\n\n{CODE}",
    f"Add a one-line docstring to each function and reply with the whole code only:\n\n{CODE}",
    "Repeat this list exactly, then continue it for three more items: alpha, beta, gamma, delta, "
    "epsilon, zeta, eta, theta",
]


class _Groups:
    def __init__(self):
        self._comm = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())

    def get_joint_group_for_node(self, node):
        return self._comm

    def get_instance_world_size_for_node(self, node):
        return 1

    def get_instance_rank_for_node(self, node):
        return 0

    def resource_needs_remote_transfer(self, nodes, local):
        return False


@pytest.fixture(scope="module", autouse=True)
def _small_ladders():
    # a few buckets, so a capture takes seconds
    saved = {name: getattr(LLMSubmodule, name) for name in (
        "DECODE_CAPTURE_BATCH_SIZES", "PREFILL_TOKEN_BUCKETS", "PREFILL_CAPTURE_BATCH_SIZES",
        "PREFILL_VISION_TOKEN_BUCKETS", "PREFILL_VISION_CAPTURE_BATCH_SIZES",
    )}
    LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4]
    LLMSubmodule.PREFILL_TOKEN_BUCKETS = [128]
    LLMSubmodule.PREFILL_CAPTURE_BATCH_SIZES = [1]
    LLMSubmodule.PREFILL_VISION_TOKEN_BUCKETS = [64]
    LLMSubmodule.PREFILL_VISION_CAPTURE_BATCH_SIZES = [1]
    yield
    for name, value in saved.items():
        setattr(LLMSubmodule, name, value)


class _Serve:
    """The worker's part for the LLM node: ingest, a prefill step, then decode steps through
    the engine's own exec and stop check until every request stops."""

    def __init__(self, draft: dict | None, captured: bool):
        self.model = Qwen3_5DenseModel(CKPT, draft=draft)
        self.sub = self.model.get_submodule("LLM", device=str(GPU))
        specs = [spec for spec in self.model.get_node_resources() if "LLM" in spec.nodes]
        self.eng = Engine(None, autocast_dtype=torch.bfloat16)
        self.eng.load_model(
            submodules={"LLM": self.sub}, specs=specs, parallel_groups=_Groups(), device=GPU,
            transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
            kv_cache_type=torch.bfloat16, model=self.model,
        )
        if captured:
            self.eng.warmup()
        self.drafting = draft is not None
        self.stop_ids = self.model.config.stop_token_ids
        self.infos: dict[int, CurrentForwardPassInfo] = {}
        self.emitted: dict[int, list[int]] = {}
        self.next: dict[int, torch.Tensor] = {}
        self.leases = []

    def add(self, prompts: list[str]) -> None:
        inputs = {}
        for rid, text in enumerate(prompts):
            out = self.model.process_prompt(text, ["text"], ["text"], enable_thinking=False)
            (ids,) = out.new_input_tensors["text_inputs"]
            kv = KVReqConfig(needed_labels=["main"])
            kv.apply_conductor_config(
                kv_layout={label: layout.row() for label, layout in out.metadata["kv_layout"][KV_CACHE].items()},
                loop_caps=self.model.get_loop_caps(max_output_tokens=MAX_TOKENS),
            )
            configs = {KV_CACHE: kv, SAMPLER: SamplingReqConfig(temperature=0.0)}
            self.eng.add_request(rid, configs)
            self.infos[rid] = CurrentForwardPassInfo(
                request_id=f"r{rid}", rid_handle=rid, graph_walk="prefill_text", fwd_index=0,
                random_seed=0, max_tokens=MAX_TOKENS, resource_configs=configs,
                step_metadata={"is_prefill": True, "last_prefill": True, "prefill_order": (TEXT_PART,)},
            )
            inputs[rid] = ids.to(GPU)
        for rid, ids in inputs.items():  # one prefill a step, as the scheduler runs this walk
            out = self._step("prefill_text", {rid: ids})[rid]
            self.emitted[rid] = out["new_token"][0].reshape(-1).tolist()
            self.next[rid] = out["new_token"][0]
            self.infos[rid].graph_walk = "decode"
            self.infos[rid].step_metadata = {"is_prefill": False, "last_prefill": False, "prefill_order": ()}

    def _step(self, walk: str, inputs: dict[int, torch.Tensor]):
        rids = list(inputs)
        batch = ExecutingBatch(
            node_name="LLM",
            per_request_info={rid: self.infos[rid] for rid in rids},
            step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
            per_request_input_tensors={rid: {"text_inputs": [inputs[rid]]} for rid in rids},
        )
        self.eng.prepare_inputs(batch)
        outputs = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        self.leases.append(batch.step_context.slot_lease)
        self.stops = self.eng.check_stop_for_batch(batch, outputs.per_rid_outputs)
        return outputs.per_rid_outputs

    def run(self, max_steps: int = 400) -> dict[int, list[int]]:
        live = [rid for rid, tokens in self.emitted.items() if self.drafting or tokens[-1] not in self.stop_ids]
        for _ in range(max_steps):
            if not live:
                break
            out = self._step("decode", {rid: self.next[rid] for rid in live})
            stopped = set(self.stops or ())
            for rid in live:
                tokens = out[rid]["new_token"][0].reshape(-1).tolist()
                self.emitted[rid] += tokens
                self.next[rid] = out[rid]["text_inputs"][0]
                if not self.drafting:
                    # plain decode: the model's own stop, as its check_stop says it
                    counts = self.infos[rid].dynamic_loop_iter_counts
                    counts["decode_loop"] = counts.get("decode_loop", 0) + 1
                    if tokens[-1] in self.stop_ids or len(self.emitted[rid]) >= MAX_TOKENS:
                        stopped.add(rid)
            live = [rid for rid in live if rid not in stopped]
        assert not live, f"requests {live} never stopped"
        return dict(self.emitted)


def _decode(draft, captured):
    serve = _Serve(draft, captured)
    serve.add(PROMPTS)
    out = serve.run()
    stats = None
    if draft is not None:
        stats = serve.eng._resources[DRAFT_VERIFY], serve.eng._resources[NGRAM]
    leases = serve.leases[len(PROMPTS):]
    del serve
    torch.cuda.empty_cache()
    return out, stats, leases


@pytest.fixture(scope="module")
def plain():
    """Plain greedy decode's streams, and each emitted token's top-two logit margin."""
    margins: dict[int, list[float]] = {}
    sample = SamplerResource.sample

    def recording(self, request_ids, logits, **kwargs):
        top = logits.float().topk(2, dim=-1).values
        for rid, gap in zip(request_ids, (top[:, 0] - top[:, 1]).tolist(), strict=True):
            margins.setdefault(rid, []).append(gap)
        return sample(self, request_ids, logits, **kwargs)

    SamplerResource.sample = recording
    try:
        out, _, _ = _decode(None, captured=False)
    finally:
        SamplerResource.sample = sample
    assert all(len(tokens) > 20 for tokens in out.values()), out
    return out, margins


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
@pytest.mark.parametrize("k", [2, 4])
def test_ngram_greedy_emits_what_plain_greedy_emits(k, captured, plain):
    streams, margins = plain
    out, (dv, prop), leases = _decode({"method": "ngram", "max_drafts": k, "sampled_verify": False}, captured)
    if captured:
        assert leases and all(lease is not None for lease in leases), "a decode step ran eager"
    for rid, want in streams.items():
        got = out[rid]
        apart = next((i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), None)
        if apart is None:
            assert got == want, (rid, dv.stats())
        else:
            # the first difference is at a tie of plain decode's, and nothing before it differs
            assert margins[rid][apart] <= TIE, (rid, apart, margins[rid][apart], dv.stats())
    # the replies copy their prompts: most verify steps keep a draft
    assert dv.drafts_accepted > 0.5 * dv.rows_verified, dv.stats()
    assert prop.rows_matched > 0
