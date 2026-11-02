"""Qwen3-8B with DeepSeek's DSpark drafter (``deepseek-ai/dspark_qwen3_8b_block7``) on the text
LM, through the engine's step path, on real weights, on a GPU.

Greedy decode with drafts must emit what plain greedy decode emits, up to the first token whose
top two logits tie in plain decode (a bf16 reduction over a block of rows tiles differently
from one row): every verify step runs the drafter's block over its context KV, the target over
the drafts, keeps the drafts the target agrees with, and writes the drafter's context from the
target's taps, rewound with the target's KV. The trained drafter has to keep drafts, which only
happens when its taps, context and block positions are the ones it was trained on.

A draft model (Qwen3-0.6B) takes the same surface (``mstar.model.components.drafters``), and
must emit what plain greedy emits too. So must DSpark with new prompts joining the decode loop
inside its verify steps (the text LM's ``mixed`` walk), each prompt prefilled in the step that
verifies the running requests' drafts.

Needs CUDA, FlashInfer and the cached Qwen/Qwen3-8B and deepseek-ai/dspark_qwen3_8b_block7
(and Qwen/Qwen3-0.6B for the draft model's case).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("flashinfer")


def _cached(repo: str) -> bool:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return any((home / f"hub/models--{repo.replace('/', '--')}/snapshots").glob("*/*.safetensors*"))


TARGET, DRAFTER = "Qwen/Qwen3-8B", "deepseek-ai/dspark_qwen3_8b_block7"
if not (_cached(TARGET) and _cached(DRAFTER)):
    pytest.skip(f"{TARGET} or {DRAFTER} is not cached", allow_module_level=True)

from mstar.communication.tensors import LocalTransferEngine  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.communication import CommGroup, JointGroups  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import DRAFT_VERIFY, SamplingReqConfig, StepContext, apply_yaml_overrides  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig  # noqa: E402
from mstar.engine.resources.kv.transfer import TransferEngineInfo  # noqa: E402
from mstar.engine.resources.sampler.resource import SamplerResource  # noqa: E402
from mstar.model.submodule_base import END_PARTITION  # noqa: E402
from mstar.model.text_lm.config import KV_CACHE, LLM_NODE, SAMPLER  # noqa: E402
from mstar.model.text_lm.submodules import TextLMSubmodule  # noqa: E402
from mstar.model.text_lm.text_lm_model import TextLMModel  # noqa: E402

GPU = torch.device("cuda", 0)
# long enough that every row's context crosses 128 tokens, FA2's smallest KV chunk
MAX_TOKENS = 160
TIE = 0.25
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in three sentences why the sky is blue.",
    "List the planets of the solar system in order from the Sun, one per line.",
    "Describe the history of the Roman Empire from its founding to its fall.",
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
def _small_ladder():
    saved = (
        TextLMSubmodule.DECODE_CAPTURE_BATCH_SIZES, TextLMSubmodule.MIXED_CAPTURE_BATCH_SIZES,
        TextLMSubmodule.MIXED_PROMPT_TOKENS,
    )
    TextLMSubmodule.DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4]
    TextLMSubmodule.MIXED_CAPTURE_BATCH_SIZES = [2, 4]
    TextLMSubmodule.MIXED_PROMPT_TOKENS = [64, 128]
    yield
    (
        TextLMSubmodule.DECODE_CAPTURE_BATCH_SIZES, TextLMSubmodule.MIXED_CAPTURE_BATCH_SIZES,
        TextLMSubmodule.MIXED_PROMPT_TOKENS,
    ) = saved


class _Serve:
    """The worker's part for the LLM node: ingest, a prefill step, then decode steps through the
    engine's own exec and stop check until every request stops."""

    def __init__(self, draft: dict | None, captured: bool):
        self.model = TextLMModel(TARGET, max_seq_len=4096, draft=draft, accelerator_graph=captured)
        self.sub = self.model.get_submodule(LLM_NODE, device=str(GPU), autocast_dtype=torch.bfloat16)
        specs = [spec for spec in self.model.get_node_resources() if LLM_NODE in spec.nodes]
        pools = {KV_CACHE: {"max_num_pages": 64}}
        if draft is not None:
            pools["dspark_kv" if draft["method"] == "dspark" else "draft_kv"] = {"max_num_pages": 64}
        apply_yaml_overrides(specs, {"resources": pools})
        self.eng = Engine(None, autocast_dtype=torch.bfloat16)
        self.eng.load_model(
            submodules={LLM_NODE: self.sub}, specs=specs, parallel_groups=_Groups(), device=GPU,
            transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
            kv_cache_type=torch.bfloat16, model=self.model,
        )
        if captured:
            self.eng.warmup()
        self.drafting = draft is not None
        self.stop_ids = self.model.stop_token_ids
        self.infos: dict[int, CurrentForwardPassInfo] = {}
        self.emitted: dict[int, list[int]] = {}
        self.next: dict[int, torch.Tensor] = {}
        self.leases = []

    def _ingest(self, rid: int, text: str) -> torch.Tensor:
        out = self.model.process_prompt(
            text, ["text"], ["text"], enable_thinking=False, max_output_tokens=MAX_TOKENS,
        )
        (ids,) = out.new_input_tensors["text_inputs"]
        configs = {SAMPLER: SamplingReqConfig(temperature=0.0)}
        for key, layouts in out.metadata["kv_layout"].items():
            kv = KVReqConfig(needed_labels=["main"])
            kv.apply_conductor_config(
                kv_layout={label: layout.row() for label, layout in layouts.items()},
                loop_caps=self.model.get_loop_caps(max_output_tokens=MAX_TOKENS),
            )
            configs[key] = kv
        self.eng.add_request(rid, configs)
        self.infos[rid] = CurrentForwardPassInfo(
            request_id=f"r{rid}", rid_handle=rid, graph_walk="prefill", fwd_index=0,
            random_seed=0, max_tokens=MAX_TOKENS, resource_configs=configs,
        )
        return ids.to(GPU)

    def add(self, prompts: list[str]) -> None:
        inputs = {rid: self._ingest(rid, text) for rid, text in enumerate(prompts)}
        out = self._step("prefill", inputs)
        for rid in inputs:
            self._prefilled(rid, out[rid])

    def _prefilled(self, rid: int, out: dict) -> None:
        self.emitted[rid] = out["new_token"][0].reshape(-1).tolist()
        # a draft model's decode loop starts from the prompt's last token and the sample
        self.next[rid] = out.get("decode_input", out["new_token"])[0]
        self.infos[rid].graph_walk = "decode"

    def _step(self, walk: str, inputs: dict[int, torch.Tensor], walks: dict[int, str] | None = None):
        rids = list(inputs)
        if walks:
            # as the worker's batch: a mixed verify step runs its verify rows first
            rids = self.eng.order_split_rows(LLM_NODE, walk, rids, walks)
        batch = ExecutingBatch(
            node_name=LLM_NODE,
            per_request_info={rid: self.infos[rid] for rid in rids},
            step_context=StepContext(
                request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False, request_walks=walks or {},
            ),
            per_request_input_tensors={rid: {"text_inputs": [inputs[rid]]} for rid in rids},
        )
        self.eng.prepare_inputs(batch)
        outputs = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        self.leases.append(batch.step_context.slot_lease)
        self.stops = self.eng.check_stop_for_batch(batch, outputs.per_rid_outputs)
        return self._stored(outputs, rids)

    @staticmethod
    def _stored(outputs, rids) -> dict:
        """What the worker's store sends on: a batch's row views ahead of its per-request outputs."""
        out = {rid: dict(outputs.per_rid_outputs.get(rid, {})) for rid in rids}
        order = list(outputs.row_request_ids or rids)
        for name, views in (outputs.row_views or {}).items():
            for rid, view in zip(order, views, strict=False):
                if rid in out:
                    out[rid][name] = [view]
        return out

    def run(self, max_steps: int = 400, joins: dict[int, dict[int, str]] | None = None) -> dict[int, list[int]]:
        """Decode until every request stops; ``joins``: step -> the prompts (rid -> text) that
        join the loop in that step, prefilled inside it (the ``mixed`` walk)."""
        # a stop sampled at the prefill ends the request there, as the runtime ends its partition
        prefill_stops = self.stops or {}
        live = [rid for rid in self.emitted if END_PARTITION not in prefill_stops.get(rid, ())]
        joins = dict(joins or {})
        for step in range(max_steps):
            if not live and not joins:
                break
            joining = {rid: self._ingest(rid, text) for rid, text in joins.pop(step, {}).items()}
            inputs = {rid: self.next[rid] for rid in live}
            if joining:
                walks = {**dict.fromkeys(inputs, "decode"), **dict.fromkeys(joining, "prefill")}
                out = self._step("mixed", {**inputs, **joining}, walks)
                for rid in joining:
                    self._prefilled(rid, out[rid])
                    if END_PARTITION not in (self.stops or {}).get(rid, ()):
                        live.append(rid)
            elif live:
                out = self._step("decode", inputs)
            else:
                continue
            stopped = set(self.stops or ())
            for rid in inputs:
                tokens = out[rid]["new_token"][0].reshape(-1).tolist()
                self.emitted[rid] += tokens
                if self.drafting:
                    self.next[rid] = out[rid]["text_inputs"][0]
                else:
                    self.next[rid] = out[rid]["new_token"][0]
                    counts = self.infos[rid].dynamic_loop_iter_counts
                    counts["decode_loop"] = counts.get("decode_loop", 0) + 1
                    if tokens[-1] in self.stop_ids or len(self.emitted[rid]) >= MAX_TOKENS:
                        stopped.add(rid)
            live = [rid for rid in live if rid not in stopped]
        assert not live, f"requests {live} never stopped"
        return dict(self.emitted)


def _decode(draft, captured, join: bool = False):
    serve = _Serve(draft, captured)
    if join:
        # two prompts start the loop; the others join it in its second and fourth steps
        serve.add(PROMPTS[:2])
        out = serve.run(joins={1: {2: PROMPTS[2]}, 3: {3: PROMPTS[3]}})
    else:
        serve.add(PROMPTS)
        out = serve.run()
    dv = serve.eng._resources[DRAFT_VERIFY] if draft is not None else None
    stats = None if dv is None else (dv.stats(), dv.rows_verified, dv.drafts_accepted)
    # the prefill's, then the decode steps'
    leases = list(serve.leases)
    del serve, dv
    gc.collect()
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
def test_dspark_greedy_emits_what_plain_greedy_emits(captured, plain):
    streams, margins = plain
    draft = {"method": "dspark", "model": DRAFTER, "max_drafts": 7, "sampled_verify": False}
    out, (stats, rows, accepted), leases = _decode(draft, captured)
    print(f"[dspark captured={captured}] {stats}")
    if captured:
        # the prefill replays a packed capture too: the drafter's context write is the target's rows
        assert all(lease is not None and lease.bucket is not None for lease in leases) and len(leases) > 1
    for rid, want in streams.items():
        got = out[rid]
        apart = next((i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), None)
        if apart is None:
            assert got == want, (rid, stats)
        else:
            assert margins[rid][apart] <= TIE, (rid, apart, margins[rid][apart], stats)
    # a trained drafter on its own target keeps drafts: taps, context and positions as trained
    assert rows and accepted > 1.5 * rows, stats


@pytest.mark.skipif(not _cached("Qwen/Qwen3-0.6B"), reason="Qwen/Qwen3-0.6B is not cached")
def test_a_draft_model_through_the_shared_surface_emits_what_plain_greedy_emits(plain):
    streams, margins = plain
    draft = {"method": "draft_model", "model": "Qwen/Qwen3-0.6B", "max_drafts": 4, "sampled_verify": False}
    out, (stats, rows, accepted), leases = _decode(draft, captured=True)
    print(f"[draft_model] {stats}")
    for rid, want in streams.items():
        got = out[rid]
        apart = next((i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), None)
        if apart is None:
            assert got == want, (rid, stats)
        else:
            assert margins[rid][apart] <= TIE, (rid, apart, margins[rid][apart], stats)
    assert rows and accepted > 0.5 * rows, stats


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
def test_dspark_prompts_joining_inside_verify_steps_emit_what_plain_greedy_emits(captured, plain):
    streams, margins = plain
    draft = {"method": "dspark", "model": DRAFTER, "max_drafts": 7, "sampled_verify": False}
    out, (stats, rows, accepted), leases = _decode(draft, captured, join=True)
    print(f"[dspark mixed captured={captured}] {stats}")
    if captured:
        # the steps the prompts joined replayed a mixed capture, the drafts' verify and both prefills
        assert all(lease is not None and lease.bucket is not None for lease in leases)
        assert sum(lease.bucket.graph_walk == "mixed" for lease in leases) == 2
    for rid, want in streams.items():
        got = out[rid]
        apart = next((i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), None)
        if apart is None:
            assert got == want, (rid, stats)
        else:
            assert margins[rid][apart] <= TIE, (rid, apart, margins[rid][apart], stats)
    assert rows and accepted > 1.5 * rows, stats

