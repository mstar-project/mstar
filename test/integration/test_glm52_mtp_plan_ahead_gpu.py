"""GLM-5.2's MTP verify steps planned ahead of the previous step's verdict, as the worker's plan
thread runs them, then corrected for the rewind (``StepRunner.settle_preplan``), on one GPU.

The reduced model at the real MLA latent dims (FlashInfer's MLA kernel) runs through the engine,
its decode captured at buckets of 1, 2 and 4 rows. Each decode step is pre-planned on a slot of
its own from the lengths the step before committed, as if it kept its whole block; the runner
then waits for that verdict, rewinds, and the KV, positions and MLA attention correct what they
staged (MLA plans its wrappers again). Random weights reject nearly every draft, so most steps
have lengths to put right; the seed oracle makes each step's first draft right, so others keep
part of their block. Every latent row a request keeps, in every layer and the MTP plane, must
equal bitwise what the same steps planned inline after the rewind wrote, and the streams too;
with the MLA correction left out the cache differs. Skips without CUDA.
"""
import pytest
import torch
from test_glm52_capture_gpu import KV_YAML, PROMPTS, _load, _Node
from test_glm52_mtp_capture_gpu import _mtp_cfg

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import DRAFT_VERIFY, KVReqConfig, StepContext, apply_yaml_overrides
from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
from mstar.engine.resources.kv import manager as kv_manager
from mstar.engine.resources.kv.config import PrefixSpan, StreamLayout
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.glm52.config import ATTN_RESOURCE, KV_RESOURCE
from mstar.model.glm52.submodules import Glm52LLMSubmodule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU",
)

DEVICE, DTYPE, NODE = torch.device("cuda:0"), torch.bfloat16, "LLM"
MAX_TOKENS = 12


class _Groups:
    """A one-rank node's lockstep instance, as the engine reads it."""

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


class _Driver:
    """The worker's part for one node, as ``draft_verify_toy.Driver``: ingest and prefill each
    prompt, then decode steps of every live request until each stops, through the engine's
    exec and stop check. ``plan_ahead`` pre-plans each decode step as the plan thread does:
    after the step before it committed, before anything reads that step's verdict."""

    def __init__(self, model, sub, eng, plan_ahead: bool, oracle: dict[int, list[int]] | None):
        self.sub, self.eng, self.plan_ahead, self.oracle = sub, eng, plan_ahead, oracle
        self.configs = model.get_request_resource_configs(
            {}, model_kwargs={"temperature": 0.0, "ignore_eos": True},
        )
        self.emitted: dict[int, list[int]] = {}
        self.next: dict[int, torch.Tensor] = {}
        self.planned_ahead = self.decode_steps = 0

    def _step(self, walk: str, inputs: dict[int, torch.Tensor]):
        rids = list(inputs)
        batch = ExecutingBatch(
            node_name=NODE,
            per_request_info={rid: CurrentForwardPassInfo(
                request_id=f"r{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0,
                random_seed=0, max_tokens=MAX_TOKENS, resource_configs=self.configs,
            ) for rid in rids},
            step_context=StepContext(
                request_ids=tuple(rids), graph_walk=walk, slot=None if self.plan_ahead else 0,
                capture=False,
            ),
            per_request_input_tensors={rid: {"text_inputs": [inputs[rid]]} for rid in rids},
        )
        if walk == "decode":
            self.decode_steps += 1
            if self.plan_ahead and self.eng.reserve_replay_slot(batch) is not None:
                self.planned_ahead += bool(self.eng.pre_plan_for_batch(batch))
        self.eng.prepare_inputs(batch)
        outputs = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        self.stops = self.eng.check_stop_for_batch(batch, outputs.per_rid_outputs)
        return outputs.per_rid_outputs

    def run(self, prompts: dict[str, list[int]]) -> dict[int, list[int]]:
        lookahead = self.sub.config.mtp_kv_lookahead
        for rid, prompt in enumerate(prompts.values()):
            self.eng.add_request(rid, {**self.configs, KV_RESOURCE: KVReqConfig(
                kv_layout={"main": StreamLayout(
                    spans=(PrefixSpan(len(prompt), len(prompt), "prefill"),),
                    growth=(("decode_loop", 1),), lookahead=lookahead,
                )},
                loop_caps={"decode_loop": MAX_TOKENS - 1},
            )})
            out = self._step("prefill", {rid: torch.tensor(prompt, device=DEVICE)})[rid]
            self.emitted[rid] = out["new_token"][0].reshape(-1).tolist()
            self.next[rid] = out["text_inputs"][0]
        live = list(self.emitted)
        for _ in range(2 * MAX_TOKENS):
            if not live:
                break
            if self.oracle is not None:
                # each request's next reference token as its first draft (the seed's d1), which
                # a replay reads in place
                for rid in live:
                    ref, nxt = self.oracle[rid], len(self.emitted[rid])
                    self.sub._mtp_seed_draft[self.sub._mtp_slot[rid]] = ref[nxt] if nxt < len(ref) else 0
            out = self._step("decode", {rid: self.next[rid] for rid in live})
            for rid in live:
                self.emitted[rid] += out[rid]["new_token"][0].reshape(-1).tolist()
                self.next[rid] = out[rid]["text_inputs"][0]
            live = [rid for rid in live if rid not in self.stops]
        assert not live, f"requests {live} never stopped"
        return self.emitted


def _run(tmp_path, monkeypatch, k: int, plan_ahead: bool, oracle=None):
    monkeypatch.setattr(AcceleratorGraphRunner, "CAPTURE_BATCH_SIZES", [1, 2, 4])
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 2, 4])
    tmp_path.mkdir(parents=True)
    model, sub = _load(tmp_path, monkeypatch, fp8=False,
                       cfg_overrides={**_mtp_cfg(k), "graph_compile": False})
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, KV_YAML)
    eng = Engine(None, autocast_dtype=DTYPE)
    eng.load_model(
        submodules={NODE: sub}, specs=specs, parallel_groups=_Groups(), device=DEVICE,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        kv_cache_type=DTYPE, model=model,
    )
    eng.warmup()
    drv = _Driver(model, sub, eng, plan_ahead, oracle)
    with torch.no_grad():
        out = drv.run(PROMPTS)
    torch.cuda.synchronize()
    return out, eng, drv


def _plain(tmp_path, monkeypatch) -> dict[int, list[int]]:
    """Plain greedy decode of the same model, eager (the capture test's ``_Node``), by the
    driver's request numbers."""
    tmp_path.mkdir(parents=True)
    model, sub = _load(tmp_path, monkeypatch, fp8=False,
                       cfg_overrides={**_mtp_cfg(0), "graph_compile": False})
    node = _Node(model, sub, capture=False, prompts=PROMPTS)
    streams = node.generate()
    node.close()
    return {i: streams[name] for i, name in enumerate(PROMPTS)}


def _kept_rows(eng) -> dict:
    """Every latent row each request keeps, per layer (the MTP plane included)."""
    kv = eng._resources[KV_RESOURCE]
    page = kv.kv_cache.page_size
    rows = {}
    for rid, streams in kv._streams.items():
        if not isinstance(rid, int) or rid < 0:
            continue
        stream = streams["main"]
        pages = torch.tensor(stream.page_indices[: -(-stream.stored_len // page)], device=DEVICE)
        for layer in range(kv.config.num_layers):
            view = kv.layer_view(layer)[pages]  # [pages, page, latent]
            rows[(rid, layer)] = view.reshape(-1, view.shape[-1])[: stream.stored_len].clone()
    return rows


def _same_rows(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


@pytest.fixture
def replans(monkeypatch):
    """Counts the MLA corrections that had a staged plan to put right."""
    count = {"staged": 0}
    real = FlashInferMLAManager.correct_preplan

    def counting(self, verdicts, step, ctx):
        count["staged"] += self._preplanned
        return real(self, verdicts, step, ctx)

    monkeypatch.setattr(FlashInferMLAManager, "correct_preplan", counting)
    return count


@pytest.mark.parametrize(("k", "oracle"), [(1, False), (3, False), (3, True)],
                         ids=["k1", "k3", "k3_seed_oracle"])
def test_steps_planned_ahead_keep_what_steps_planned_after_the_rewind_keep(
    tmp_path, monkeypatch, replans, k, oracle,
):
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    reference = None
    if oracle:
        reference = _plain(tmp_path / "plain", monkeypatch)
    out, eng, drv = _run(tmp_path / "ahead", monkeypatch, k, plan_ahead=True, oracle=reference)
    # every decode step was planned ahead of its predecessor's verdict
    assert drv.planned_ahead == drv.decode_steps > 0
    assert eng._resources[ATTN_RESOURCE].corrects_preplan and replans["staged"] > 0
    behind, behind_eng, behind_drv = _run(tmp_path / "behind", monkeypatch, k, plan_ahead=False,
                                          oracle=reference)
    assert behind_drv.planned_ahead == 0
    assert out == behind
    if reference is not None:
        assert out == reference
    assert _same_rows(_kept_rows(eng), _kept_rows(behind_eng))
    dv, behind_dv = eng._resources[DRAFT_VERIFY], behind_eng._resources[DRAFT_VERIFY]
    assert dv.accepted_hist.tolist() == behind_dv.accepted_hist.tolist()
    # rows rejected drafts, so the corrections had lengths to put right
    assert dv.drafts_accepted < k * dv.rows_verified, dv.stats()
    if oracle:
        assert dv.drafts_accepted > 0


def test_the_mla_correction_is_needed(tmp_path, monkeypatch):
    """With the staged MLA plans kept at the untrimmed lengths, the rows the cache keeps are no
    longer the ones steps planned after the rewind write."""
    behind, behind_eng, _ = _run(tmp_path / "behind", monkeypatch, 3, plan_ahead=False)
    monkeypatch.setattr(FlashInferMLAManager, "correct_preplan", lambda self, verdicts, step, ctx: True)
    out, eng, drv = _run(tmp_path / "ahead", monkeypatch, 3, plan_ahead=True)
    assert drv.planned_ahead > 0
    assert not _same_rows(_kept_rows(eng), _kept_rows(behind_eng))
