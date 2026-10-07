"""Captured prefill (``prefill_graphs``) against the eager prefill — CUDA, sm90 (FlashInfer MLA).

Two levels. The fused KDA prefill over a bucket-sized ``varlen_layout`` against the exact
one: the kernels work per span, so real rows and slots must match bit for bit.
And the whole submodule captured by the engine's ``CudaGraphRunner`` and replayed the way
``Engine._exec_single`` drives a leased step, against the eager forward of the same prompts.
"""
import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.cuda_graph_runner import CudaGraphRunner
from mstar.engine.resources import SamplingReqConfig, StepRunner
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn import kda_triton
from mstar.engine.resources.recurrent.pool import SINK_SLOT
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.resources.step import StepContext
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import (
    KDA_STATE,
    SAMPLER,
    Glm5NextModelConfig,
    build_indexer_types,
    build_layer_types,
    build_mlp_layer_types,
)
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.kda import Glm5NextKdaConfig, Glm5NextLinearAttention
from mstar.model.glm5_next.quantization import Fp8BlockQuantConfig
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.model.glm5_next.weight_loader import restore_fp32_params
from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_decode._HAS_TRITON
         and torch.cuda.get_device_capability()[0] == 9),
    reason="needs sm90 (FlashInfer MLA) + triton",
)

DEV = torch.device("cuda", 0)


# --- the fused KDA prefill over a static layout -----------------------------


def _kda_inputs(total, heads=2, head_dim=128, slots=6, seed=0):
    torch.manual_seed(seed)
    kda = Glm5NextLinearAttention(Glm5NextKdaConfig(
        hidden_size=256, linear_num_heads=heads, linear_head_dim=head_dim)).to(DEV)
    with torch.no_grad():
        for name, p in kda.named_parameters():
            if name.endswith("A_log"):
                p.normal_(0.0, 0.5)
            elif name.endswith("dt_bias"):
                p.uniform_(-8.0, 2.0)
            elif "norm" in name:
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.05)
    width = 3 * heads * head_dim + heads + 2 * head_dim
    proj = torch.randn(total, width, device=DEV).to(torch.bfloat16)
    rec = torch.randn(slots, heads, head_dim, head_dim, device=DEV) * 0.1
    conv = torch.randn(slots, 3 * heads * head_dim, 3, device=DEV).to(torch.bfloat16)
    return kda, proj, rec, conv


def _run(kda, proj, rec, conv, varlen):
    return kda_triton.kda_prefill(
        proj, conv, kda.conv1d.weight.view(kda.conv_dim, -1),
        kda.forget_gate.f_b_proj.weight, kda.g_b_proj.weight, kda.forget_gate.dt_bias,
        kda.forget_gate.A_log, kda.o_norm.weight, rec, varlen,
        num_heads=kda.num_heads, head_dim=kda.head_dim,
        lower_bound=kda.forget_gate.gate_lower_bound, scale=kda.head_dim ** -0.5,
        norm_eps=kda.o_norm.variance_epsilon,
        cfg=kda_triton.PrefillConfig(scan_bv=32),
    )


def _varlen(lens, slots, num_tokens, fixed):
    layout = kda_triton.varlen_layout(lens, num_tokens if fixed else None)
    return kda_triton.varlen_view(
        torch.tensor(layout, dtype=torch.int32, device=DEV),
        torch.tensor(slots, dtype=torch.int32, device=DEV), len(lens), num_tokens)


@pytest.mark.parametrize("lengths, pad_rows, bucket", [
    ([300], 0, 512),
    ([100, 37], 1, 256),
    ([64, 1, 130], 1, 512),
    ([256], 0, 256),
])
def test_static_layout_matches_the_exact_one(lengths, pad_rows, bucket):
    """Real rows and real slots bit-exact; the padding rows' sink slot and the other slots
    untouched but for the sink."""
    kda, proj, rec, conv = _kda_inputs(bucket)
    real = sum(lengths)
    slots = list(range(1, len(lengths) + 1))

    rec_a, conv_a = rec.clone(), conv.clone()
    exact = _run(kda, proj[:real], rec_a, conv_a, _varlen(lengths, slots, real, fixed=False))

    rec_b, conv_b = rec.clone(), conv.clone()
    varlen = _varlen(lengths + [0] * pad_rows, slots + [SINK_SLOT] * pad_rows, bucket,
                     fixed=True)
    assert varlen.num_chunks == bucket // 64 + len(lengths) + pad_rows
    static = _run(kda, proj, rec_b, conv_b, varlen)

    assert torch.equal(static[:real], exact)
    live = list(range(1, len(lengths) + 1))
    assert torch.equal(rec_b[live], rec_a[live]) and torch.equal(conv_b[live], conv_a[live])
    rest = list(range(len(lengths) + 1, rec.shape[0]))
    assert torch.equal(rec_b[rest], rec[rest]) and torch.equal(conv_b[rest], conv[rest])


# --- the submodule captured by the engine's runner ---------------------------


def _config(buckets, batch_sizes):
    n = 5  # layers 0-2 dense KDA, 3 MLA + MoE, 4 KDA + MoE
    return Glm5NextModelConfig(
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
        prefill_graphs=True, prefill_token_buckets=buckets,
        prefill_capture_batch_sizes=batch_sizes,
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


class _Engine:
    """The submodule, its real resources and a CudaGraphRunner, stepped like Engine."""

    def __init__(self, buckets, batch_sizes, seed=0):
        torch.manual_seed(seed)
        self.cfg = cfg = _config(buckets, batch_sizes)
        with torch.device("meta"):
            lm = Glm5NextForCausalLM(cfg)
        lm = lm.to(torch.bfloat16)
        lm.to_empty(device=DEV)
        restore_fp32_params(lm)
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
                k: by_key[k] for k in spec.depends_on()})
            self.resources[spec.resource_key] = build_resource(spec, info)
        self.runner = StepRunner(self.resources, node_resources={"LLM": list(self.resources)})
        self.sub = Glm5NextLLMSubmodule(self.lm, cfg)
        self.sub.bind_node_resources(self.resources)
        trivial = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        self.cg = CudaGraphRunner(
            submodule_name="LLM", submodule=self.sub, resources=self.resources,
            step_runner=self.runner, device=DEV, autocast_dtype=None,
            joint_comm_group=trivial, num_slots=1,
        )
        self.cg.warmup_and_capture()

    def info(self, rid, walk):
        return CurrentForwardPassInfo(
            request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=8,
            resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)},
            dynamic_loop_iter_counts={"decode_loop": 0},
        )

    def step(self, walk, prompts, captured):
        """One step over ``prompts`` (rid -> ids); returns (tokens, lease)."""
        rids = list(prompts)
        for rid in rids:
            if walk == "prefill" and rid not in self.resources[KDA_STATE]._slots:
                self.runner.ingest_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})
        infos = {rid: self.info(rid, walk) for rid in rids}
        inputs = [self.sub.prepare_inputs(walk, infos[rid], {"text_inputs": [ids]})
                  for rid, ids in prompts.items()]
        lease = None
        if captured:
            lease = self.cg.lease_slot(walk, bs=len(rids), slot=0,
                                       num_tokens=sum(i.input_seq_len for i in inputs))
            assert lease is not None, "no captured bucket fits"
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False,
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
        rows = raw.row_outputs["new_token"]
        tokens = {rid: rows[i : i + 1].clone() for i, rid in enumerate(rids)}
        if lease is not None:
            self.cg.release(lease, len(rids))
        return tokens, lease

    def state(self, rid):
        pool = self.resources[KDA_STATE]
        slot = pool._slots[rid]["main"].index
        return pool._blocks["state"][:, slot].clone(), pool._blocks["conv"][:, slot].clone()


class _NoTransfer:
    """A resource built by hand (no engine) has no transfer engine to hand the KV manager."""

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


@pytest.fixture(scope="module")
def engine():
    from mstar.engine.resources.kv import manager as kv_mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kv_mod, "KVTransferManager", _NoTransfer)
        # rows 2 only, so a one-prompt step replays with a padding row
        yield _Engine(buckets=[256, 512], batch_sizes=[2])


def _prompts(tag, lengths, seed):
    g = torch.Generator().manual_seed(seed)
    return {f"{tag}{i}": torch.randint(2, 512, (n,), generator=g).to(DEV)
            for i, n in enumerate(lengths)}


def test_no_prefill_capture_without_the_fused_kda():
    from mstar.engine.resources.kv import manager as kv_mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kv_mod, "KVTransferManager", _NoTransfer)
        mp.setattr(fused_decode, "_ENABLED", False)
        eng = _Engine(buckets=[256], batch_sizes=[2])
    assert not any(k.graph_walk == "prefill" for k in eng.cg._buckets)


def test_prefill_buckets_are_captured(engine):
    walks = {(k.graph_walk, k.bs, k.num_tokens) for k in engine.cg._buckets}
    assert {("prefill", 2, 256), ("prefill", 2, 512)} <= walks


@pytest.mark.parametrize("lengths", [
    [128, 128],   # exact fit: the replay runs the eager shapes
    [100, 180],   # padded tokens
    [300],        # padded tokens and a padding row
    [137],
])
def test_captured_prefill_matches_eager(engine, lengths):
    seed = sum(lengths)
    eager = _prompts("e", lengths, seed)
    graph = {k.replace("e", "g", 1): v for k, v in eager.items()}
    tok_e, _ = engine.step("prefill", eager, captured=False)
    tok_g, lease = engine.step("prefill", graph, captured=True)
    assert lease.bucket.num_tokens >= sum(lengths)
    for (re, rg) in zip(eager, graph, strict=True):
        assert torch.equal(tok_e[re], tok_g[rg]), (re, tok_e[re], tok_g[rg])
        (rec_e, conv_e), (rec_g, conv_g) = engine.state(re), engine.state(rg)
        exact = sum(lengths) == lease.bucket.num_tokens and len(lengths) == lease.bucket.bs
        if exact:
            assert torch.equal(rec_g, rec_e) and torch.equal(conv_g, conv_e)
        else:  # the dense GEMMs see more rows, so cuBLAS may block them differently
            torch.testing.assert_close(rec_g, rec_e, rtol=2e-2, atol=2e-3)
            torch.testing.assert_close(conv_g.float(), conv_e.float(), rtol=2e-2, atol=2e-2)
    # the decode step after reads both paths' state and KV the same way
    nxt_e, _ = engine.step("decode", {r: tok_e[r].view(1) for r in eager}, captured=False)
    nxt_g, _ = engine.step("decode", {r: tok_g[r].view(1) for r in graph}, captured=False)
    for re, rg in zip(eager, graph, strict=True):
        assert torch.equal(nxt_e[re], nxt_g[rg])
    for rid in [*eager, *graph]:
        engine.runner.remove_request(rid)


def test_nan_router_rows_stay_in_range():
    logits = torch.randn(4, 16, device=DEV)
    logits[1] = float("nan")
    w, ids = fused_decode.router_topk(logits, torch.zeros(16, device=DEV), top_k=4, scale=1.0,
                                      normalize=True)
    assert int(ids.max()) < 16 and int(ids.min()) >= 0
    ref = torch.topk(torch.sigmoid(logits[0]), 4).indices.sort().values
    assert torch.equal(ids[0].sort().values, ref)
