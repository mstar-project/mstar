"""glm5_next MTP on CUDA (sm90, FlashInfer MLA, fused KDA), on the engine's draft-verify: the
captured verify step emits what the eager one does, and both emit what greedy decode without
MTP does. The harness steps the engine's resources as the engine does, the verdict's cut
(``DraftVerifyResource.emitted_rows``) included."""
import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.resources import SamplingReqConfig, StepRunner
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.draft_verify.startup import verify_walk
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.resources.step import StepContext
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import (
    KDA_STATE,
    KV_CACHE,
    SAMPLER,
    Glm5NextModelConfig,
    build_indexer_types,
    build_layer_types,
    build_mlp_layer_types,
)
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.model.glm5_next.weight_loader import restore_fp32_params
from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine
from mstar.utils.fp8_block import Fp8BlockQuantConfig

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_decode._HAS_TRITON
         and torch.cuda.get_device_capability()[0] == 9),
    reason="needs sm90 (FlashInfer MLA) + triton",
)

DEV = torch.device("cuda", 0)
MAX_TOKENS = 24


def _config(k, long_context=False):
    n = 5  # layers 0-2 dense KDA, 3 MLA + MoE, 4 KDA + MoE; MTP = layer 5
    long = dict(dsa_long_context=True, index_topk=64, max_seq_len=1024,
                prefill_window_tokens=48) if long_context else {}
    return Glm5NextModelConfig(
        **long,
        vocab_size=512, hidden_size=256, num_hidden_layers=n,
        num_attention_heads=8, num_key_value_heads=8,
        layer_types=build_layer_types(n), mlp_layer_types=build_mlp_layer_types(n, 3),
        indexer_types=build_indexer_types(n),
        linear_num_heads=2, linear_head_dim=128,
        q_lora_rank=128, kv_lora_rank=512, qk_nope_head_dim=64, v_head_dim=64,
        index_n_heads=2, index_head_dim=64,
        intermediate_size=512, moe_intermediate_size=256, n_routed_experts=16,
        num_experts_per_tok=4, eos_token_ids=(1,), pad_token_id=0,
        quantization_config=Fp8BlockQuantConfig(), moe_quant_kernel="auto",
        prefill_graphs=True, prefill_token_buckets=[256], prefill_capture_batch_sizes=[2, 4],
        mtp_num_draft_tokens=k,
    )


@torch.no_grad()
def _randomize(lm):
    for name, p in lm.named_parameters():
        if p.dtype == torch.uint8:  # fp8 experts
            p.copy_((torch.randn(p.shape, device=p.device) * 0.05)
                    .to(torch.float8_e4m3fn).view(torch.uint8))
        elif "scale_inv" in name:
            p.fill_(1.0)
        elif name.endswith("A_log"):
            p.fill_(0.0)
        elif name.endswith("norm.weight"):
            p.normal_(1.0, 0.1)
        elif p.is_floating_point():
            p.normal_(0.0, 0.05)


class _NoTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def remove_request(self, request_id):
        del request_id


class _Engine:
    """The submodule, its real resources and a AcceleratorGraphRunner, stepped like Engine
    (prepare -> declare -> admit, whose runner rewinds the last verify step -> plan ->
    preprocess -> run -> commit -> the verdict's cut). Requests ignore EOS, so every step
    emits until MAX_TOKENS."""

    def __init__(self, k, seed=0, long_context=False):
        cfg = _config(k, long_context)
        with torch.device("meta"):
            lm = Glm5NextForCausalLM(cfg)
        lm = lm.to(torch.bfloat16)
        lm.to_empty(device=DEV)
        restore_fp32_params(lm)
        torch.manual_seed(seed)  # the trunk's parameters come first: the same trunk for any k
        _randomize(lm)
        process_weights_after_loading(lm, DEV)
        self.lm = lm.eval().requires_grad_(False)
        model = Glm5NextModel("x", config_variant="reduced", kda_max_requests=8)
        model.config = cfg
        specs = model.get_node_resources()
        by_key = resolve_spec_dependencies(specs)
        self.resources = {}
        for spec in specs:
            if hasattr(getattr(spec, "config", None), "max_num_pages"):
                spec.config.max_num_pages = 128
            info = EngineResourceInfo(device=DEV, dependencies={
                key: by_key[key] for key in spec.depends_on()})
            self.resources[spec.resource_key] = build_resource(spec, info)
        self.runner = StepRunner(self.resources, node_resources={"LLM": list(self.resources)})
        self.sub = Glm5NextLLMSubmodule(self.lm, cfg)
        self.sub.bind_node_resources(self.resources)
        # what the engine checks and binds when it loads a node that verifies drafts
        self.walk = verify_walk("LLM", self.sub, self.resources, "decode")
        trivial = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        self.cg = AcceleratorGraphRunner(
            submodule_name="LLM", submodule=self.sub, resources=self.resources,
            step_runner=self.runner, device=DEV, autocast_dtype=None,
            joint_comm_group=trivial, num_slots=2,
        )
        self.cg.warmup_and_capture(None)

    def info(self, rid, walk):
        return CurrentForwardPassInfo(
            request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
            max_tokens=MAX_TOKENS,
            resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=True)},
            dynamic_loop_iter_counts={"decode_loop": 0},
        )

    def preplan(self, walk, rids, slot):
        """TP async's order (Engine.pre_plan_for_batch): lease, declare on the bucket's
        template rows, admit and plan, all before ``prepare_inputs``."""
        lease = self.cg.lease_slot(walk, bs=len(rids), slot=slot)
        assert lease is not None, "no batched capture fits"
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=slot, capture=False,
                          slot_lease=lease)
        ctx.set_padded_rids(self.cg.step_ids(lease, rids))
        step = self.sub.declare_step(walk, list(ctx.padded_request_ids),
                                     self.cg.declare_inputs_for(lease), slot_lease=lease)
        ctx.is_preplan = True
        step.set_ctx(ctx)
        assert self.runner.pre_admit(step).ok
        self.runner.pre_plan(step)
        ctx.is_preplan = False
        return lease

    def step(self, walk, prompts, captured, slot=0, preplan=False):
        """One step; returns ({rid: emitted tokens}, {rid: next input}, lease)."""
        rids = list(prompts)
        if walk == "prefill":
            for rid in rids:
                self.runner.ingest_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=True)})
        lease = self.preplan(walk, rids, slot) if preplan else None
        infos = {rid: self.info(rid, walk) for rid in rids}
        inputs = [self.sub.prepare_inputs(walk, infos[rid], {"text_inputs": [ids]})
                  for rid, ids in prompts.items()]
        if captured and lease is None:
            lease = self.cg.lease_slot(walk, bs=len(rids), slot=slot,
                                       num_tokens=sum(i.input_seq_len for i in inputs))
            assert lease is not None, "no captured bucket fits"
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=slot, capture=False,
                          slot_lease=lease)
        if lease is not None:
            inputs = self.cg.pad_inputs(lease, inputs)
            infos = self.cg.step_metadata(lease, rids, infos)
            ctx.set_padded_rids(self.cg.step_ids(lease, rids))
        padded = list(ctx.padded_request_ids)
        step = self.sub.declare_step(walk, padded, inputs, slot_lease=lease)
        step.set_ctx(ctx)
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        ei = ModelInputsFromEngine(request_ids=padded, per_request_info=infos,
                                   resources=self.resources, captured=lease is not None,
                                   step=step)
        pre = self.sub.preprocess(walk, ei, inputs)
        with torch.no_grad():
            raw = (self.cg.run_forward(lease, pre) if lease is not None
                   else self.sub.forward_batched(walk, ei, **pre))
        self.runner.commit(step)
        raw = BatchedModelOutput.coerce(raw)
        out_ids = self.cg.slot_for(lease).dummy_rids if lease is not None else padded
        cut = {}
        if self.walk is not None and walk == "decode":
            # Engine._emit_verified's cut of the verdict
            rows = self.walk.resource.emitted_rows(
                rids, [MAX_TOKENS] * len(rids), [True] * len(rids), self.walk.policy)
            ends = rows.counts.cumsum().tolist()
            cut = {rid: rows.tokens[end - n:end].tolist()
                   for rid, n, end in zip(rows.request_ids, rows.counts.tolist(), ends, strict=True)}
        emitted, nxt = {}, {}
        for i, rid in enumerate(rids):
            # a prefill step hands its tokens over as rows, an MTP step per request
            out = raw.get(out_ids[i]) or {
                name: [t[i : i + 1]] for name, t in (raw.row_outputs or {}).items()}
            emitted[rid] = cut[rid] if rid in cut else out["new_token"][0].tolist()
            nxt[rid] = out.get("text_inputs", out.get("new_token"))[0].clone().to(DEV)
        if lease is not None:
            self.cg.release(lease, len(rids))
        return emitted, nxt, lease

    def generate(self, prompts, captured_decode, captured_prefill=False, preplan=False):
        emitted, nxt, _ = self.step("prefill", prompts, captured=captured_prefill)
        seqs = dict(emitted)
        step = 0
        while min(len(s) for s in seqs.values()) < MAX_TOKENS:
            got, nxt, lease = self.step("decode", nxt, captured=captured_decode, slot=step % 2,
                                        preplan=preplan)
            if captured_decode:
                assert lease is not None
            for rid, toks in got.items():
                seqs[rid] += toks
            step += 1
        for rid in prompts:
            self.runner.remove_request(rid)
            self.sub.request_states.pop(rid, None)  # the engine drops it on removal
        return {rid: s[:MAX_TOKENS] for rid, s in seqs.items()}, step


@pytest.fixture(scope="module", autouse=True)
def _no_transfer():
    from mstar.engine.resources.kv import manager as kv_mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kv_mod, "KVTransferManager", _NoTransfer)
        yield


def _prompts(lengths, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {f"r{i}": torch.randint(2, 512, (n,), generator=g).to(DEV)
            for i, n in enumerate(lengths)}


# "three" replays on the 4-row graph: a padding row in every decode step
LENGTHS = {"one": [37], "two": [60, 13], "three": [30, 45, 20]}


def _split(a, b):
    """First index where two token lists differ, None if equal."""
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)


@pytest.fixture(scope="module")
def greedy():
    eng = _Engine(k=0)
    out = {n: eng.generate(_prompts(lengths), captured_decode=True)[0]
           for n, lengths in LENGTHS.items()}
    # the control: request r0 of "two" decoded alone, without MTP
    alone = eng.generate({"r0": _prompts(LENGTHS["two"])["r0"]}, captured_decode=True)[0]
    out["two_r0_alone"] = alone["r0"]
    return out


def test_greedy_alone_vs_batched_control(greedy):
    """Without MTP, the same prompt decoded alone and next to another request: where this
    splits, bf16 near-ties decide, not MTP. Reported, not asserted."""
    print("k=0 r0 alone vs batched: first split",
          _split(greedy["two_r0_alone"], greedy["two"]["r0"]))


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("batch", list(LENGTHS))
def test_captured_verify_matches_eager_and_greedy(greedy, k, batch):
    lengths = LENGTHS[batch]
    eng = _Engine(k=k)
    walks = {key.graph_walk for key in eng.cg._buckets}
    assert walks == {"prefill", "decode"}, walks
    eager, steps_e = eng.generate(_prompts(lengths), captured_decode=False)
    graph, steps_g = eng.generate(_prompts(lengths), captured_decode=True)
    # capture must not change a single token
    assert graph == eager, ("graph", graph, "eager", eager)
    assert steps_g == steps_e
    # The trunk sees k + 1 rows per request under MTP, one without (other GEMM shapes and
    # KV splits), so a near-tie in this random model can flip a token after a few steps. The
    # fp32 CPU test pins exact equality. A broken verify step splits at once.
    for rid in eager:
        split = _split(eager[rid], greedy[batch][rid])
        print(f"k={k} {batch} {rid}: MTP vs greedy first split {split}")
        assert split is None or split >= 4, (rid, split, eager[rid], greedy[batch][rid])
    # The captured prefill too (the MTP pass over the prompt inside the graph). A padded
    # bucket moves the prefill GEMMs' rounding (test_glm5next_prefill_graph), so the same
    # rule as against greedy.
    full, _ = eng.generate(_prompts(lengths), captured_decode=True, captured_prefill=True)
    for rid in eager:
        split = _split(full[rid], eager[rid])
        print(f"k={k} {batch} {rid}: fully captured vs eager first split {split}")
        assert split is None or split >= 4, (rid, split, full[rid], eager[rid])
    # TP async's order: each decode step admitted and planned before its prepare_inputs; the
    # admit's rewind waits for the last verdict.
    ahead, _ = eng.generate(_prompts(lengths), captured_decode=True, preplan=True)
    assert ahead == graph, ("preplanned", ahead, "graph", graph)
    kv, pool = eng.resources[KV_CACHE], eng.resources[KDA_STATE]
    # the graph runner's padding streams stay resident; the requests' are gone
    assert not any(rid in kv._streams for rid in _prompts(lengths))
    assert pool.num_free_slots == pool.config.usable_slots


# past index_topk (64), several windows; "three" replays bucket 4 with a padding request
LONG_LENGTHS = {"one": [150], "two": [100, 70], "three": [100, 70, 90]}


@pytest.fixture(scope="module")
def greedy_long():
    eng = _Engine(k=0, long_context=True)
    return {n: eng.generate(_prompts(lengths, seed=3), captured_decode=True)[0]
            for n, lengths in LONG_LENGTHS.items()}


@pytest.fixture(scope="module")
def greedy_long_captured():
    """Greedy with the prefill captured too, for the batches the one 256-token bucket holds.
    A captured prefill rounds apart from the eager windows (its KDA and GEMM shapes differ),
    so each is compared with its own kind."""
    eng = _Engine(k=0, long_context=True)
    return {n: eng.generate(_prompts(lengths, seed=3), captured_decode=True,
                            captured_prefill=True)[0]
            for n, lengths in LONG_LENGTHS.items() if sum(lengths) <= 256}


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("batch", list(LONG_LENGTHS))
def test_long_context_captured_verify_matches_eager_and_greedy(greedy_long, greedy_long_captured,
                                                              k, batch):
    """dsa_long_context: the engine rewinds each verify step's rejected rows before the next
    step admits, so the host plans every block at its request's true rows, and the MTP layer
    selects with its own indexer. Captured == eager, both == greedy without MTP (the same
    near-tie rule as above), and the pre-planned order == captured."""
    lengths = LONG_LENGTHS[batch]
    eng = _Engine(k=k, long_context=True)
    assert eng.sub.config.dsa_long_context and eng.walk is not None
    eager, steps_e = eng.generate(_prompts(lengths, seed=3), captured_decode=False)
    graph, steps_g = eng.generate(_prompts(lengths, seed=3), captured_decode=True)
    assert graph == eager, ("graph", graph, "eager", eager)
    assert steps_g == steps_e
    for rid in eager:
        split = _split(eager[rid], greedy_long[batch][rid])
        print(f"long k={k} {batch} {rid}: MTP vs greedy first split {split}")
        assert split is None or split >= 4, (rid, split, eager[rid], greedy_long[batch][rid])
    ahead, _ = eng.generate(_prompts(lengths, seed=3), captured_decode=True, preplan=True)
    assert ahead == graph, ("preplanned", ahead, "graph", graph)
    if batch in greedy_long_captured:  # a captured prefill seeds the drafter too
        full, _ = eng.generate(_prompts(lengths, seed=3), captured_decode=True,
                               captured_prefill=True)
        for rid in eager:
            split = _split(full[rid], greedy_long_captured[batch][rid])
            print(f"long k={k} {batch} {rid}: fully captured vs its greedy first split {split}")
            assert split is None or split >= 4, (rid, split, full[rid],
                                                 greedy_long_captured[batch][rid])
