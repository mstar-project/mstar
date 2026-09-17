"""GLM-5.3-Flash serving submodule on the resource-pool engine — CPU, reduced
config, no GPU deps.
"""
from __future__ import annotations

import logging
import sys
import types

sys.path.insert(0, ".")

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig
from mstar.engine.cuda_graph_runner import DEFAULT_CAPTURE_BATCH_SIZES
from mstar.engine.resources import (
    SINK_SLOT,
    AttentionStep,
    KVStep,
    SamplerStep,
    SamplingReqConfig,
    Segment,
    SlotStatePlan,
    SlotStateStep,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo, Resource, build_resource
from mstar.engine.resources.kv import manager as kv_mod
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.resources.step import StepContext
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock
from mstar.model.glm5_next.config import (
    ATTN,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeSubmodule,
)

MAX_SLOTS = 4
DEVICE = torch.device("cpu")


# --- stubs ------------------------------------------------------------------


class _GreedySampler(Resource):
    """Argmax in place of the Triton sampler; the Resource hooks are no-ops,
    so the runner sweeps its step like the real one's."""

    @classmethod
    def build(cls, spec, info):
        return cls()

    def sample(self, request_ids, logits, **kwargs):
        return logits.argmax(-1)


class _StubTransfer:
    """The KV manager builds a transfer manager from ``transfer_engine_info``;
    a resource built by hand (no engine) has none."""

    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self):
        return None

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(
        rmsnorm=lambda x, w, eps=1e-6: (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        ).to(x.dtype) * w.to(x.dtype)
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setattr(kv_mod, "KVTransferManager", _StubTransfer)


# --- fixtures ---------------------------------------------------------------


def _build_model(
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    max_slots: int = MAX_SLOTS,
    mtp_num_draft_tokens: int = 0,
) -> tuple[Glm5NextModel, Glm5NextForCausalLM, Glm5NextModelConfig]:
    """Reduced-config model with small random weights, finalized the way the loader
    finalizes it (``process_weights_after_loading`` resolves the MoE dispatch and builds
    the absorbed MLA projections).
    """
    torch.manual_seed(seed)
    model = Glm5NextModel(
        "x", config_variant="reduced", kda_conv_dtype=dtype,
        kda_max_requests=max_slots, mtp_num_draft_tokens=mtp_num_draft_tokens,
    )
    cfg = model.config
    lm = Glm5NextForCausalLM(cfg)
    with torch.no_grad():
        for param in lm.parameters():
            if param.is_floating_point():
                param.normal_(0, 0.02)
    if dtype != torch.float32:
        lm.to(dtype)
    process_weights_after_loading(lm, DEVICE)
    lm.eval()
    lm.requires_grad_(False)
    return model, lm, cfg


def _info(
    rid: str,
    walk: str = "decode",
    max_tokens: int = 64,
    ignore_eos: bool = False,
    decode_iters: int | None = 0,
) -> CurrentForwardPassInfo:
    counts = {} if decode_iters is None else {"decode_loop": decode_iters}
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=max_tokens,
        resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts=counts,
    )


def _padding_input() -> ARNodeInputs:
    """What ``BatchedCudaGraphConfig.get_node_inputs`` hands a padding row:
    a clone of the config's single-request decode input (one token)."""
    return ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=1)


class _Harness:
    """The submodule bound to the model's real resources, driven by a
    ``StepRunner`` the way the engine drives it."""

    def __init__(self, seed: int = 0, dtype: torch.dtype = torch.float32, max_slots: int = MAX_SLOTS):
        self.model, self.lm, self.cfg = _build_model(seed=seed, dtype=dtype, max_slots=max_slots)
        self.sub = Glm5NextLLMSubmodule(self.lm, self.cfg)
        specs = self.model.get_node_resources()
        by_key = resolve_spec_dependencies(specs)
        self.resources: dict[str, Resource] = {}
        for spec in specs:
            if spec.resource_key == SAMPLER:
                self.resources[SAMPLER] = _GreedySampler()
                continue
            info = EngineResourceInfo(
                device=DEVICE, kv_dtype=dtype,
                dependencies={k: by_key[k] for k in spec.depends_on()},
            )
            self.resources[spec.resource_key] = build_resource(spec, info)
        self.runner = StepRunner(self.resources, node_resources={"LLM": list(self.resources)})
        self.sub.bind_node_resources(self.resources)
        self.kv = self.resources[KV_CACHE]
        self.kda = self.resources[KDA_STATE]
        self.last_step: SubmoduleStep | None = None

    def ingest(self, *rids: str) -> None:
        for rid in rids:
            self.runner.ingest_request(rid, {})

    def remove(self, *rids: str) -> None:
        for rid in rids:
            self.runner.remove_request(rid)

    def prepare(self, walk: str, rid: str, ids: torch.Tensor) -> ARNodeInputs:
        return self.sub.prepare_inputs(walk, _info(rid, walk), {"text_inputs": [ids]})

    def step(
        self,
        walk: str,
        inputs: dict[str, torch.Tensor],
        padded: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """One engine step over ``inputs`` (rid -> token ids); ``padded`` is
        the captured-replay addressing (real rids first, then the padding
        rids), each padding row carrying the capture config's one-token
        input. Returns the emitted token per rid the forward answered for.
        """
        rids = list(inputs)
        node_inputs = [self.prepare(walk, rid, ids) for rid, ids in inputs.items()]
        step_rids = rids if padded is None else list(padded)
        assert step_rids[: len(rids)] == rids, "padded addressing puts the real rids first"
        node_inputs.extend(_padding_input() for _ in step_rids[len(rids):])

        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
        if padded is not None:
            ctx.set_padded_rids(tuple(step_rids))
        step = self.sub.declare_step(walk, step_rids, node_inputs)
        step.set_ctx(ctx)
        admit = self.runner.admit(step)
        assert admit.ok, admit.reason
        self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=step_rids, per_request_info={}, resources=self.resources, step=step,
        )
        pre = self.sub.preprocess(walk, engine_inputs, node_inputs)
        # the engine runs exec under no_grad (engine.py); the MLA layers'
        # absorbed projections are built from live params and carry grad_fn
        with torch.no_grad():
            out = self.sub.forward_batched(walk, engine_inputs, **pre)
        self.runner.commit(step)
        self.last_step = step
        return {rid: out[rid]["new_token"][0] for rid in out}


def _moe_blocks(lm: Glm5NextForCausalLM) -> list[Glm5NextSparseMoeBlock]:
    return [m for m in lm.modules() if isinstance(m, Glm5NextSparseMoeBlock)]


# --- (1) surface + construction ---------------------------------------------


class TestSurface:
    def test_exposes_concrete_arnode_surface(self):
        # Every method the engine drives is overridden ON this class, not left
        # as the base stub. nn.Module's metaclass is not ABCMeta, so
        # __abstractmethods__ is never populated — check the override set.
        for name in (
            "prepare_inputs", "declare_step", "preprocess", "forward",
            "forward_batched", "can_batch", "postprocess", "check_stop",
            "max_batch_size", "to", "get_cuda_graph_configs",
        ):
            assert name in Glm5NextLLMSubmodule.__dict__, name
            assert callable(getattr(Glm5NextLLMSubmodule, name)), name
        # cleanup_request is deliberately the base one: the submodule holds
        # no per-request state of its own (the resources hold it all).
        assert "cleanup_request" not in Glm5NextLLMSubmodule.__dict__

        _model, lm, cfg = _build_model(seed=37)
        sub = Glm5NextLLMSubmodule(language_model=lm, config=cfg)
        assert isinstance(sub, ARNodeSubmodule)
        assert isinstance(sub, NodeSubmodule)
        assert sub.can_batch(None, []) is True
        assert sub.language_model is lm
        assert sub.lm_head is lm.lm_head
        assert sub.config is cfg
        # unbound: no resources yet, so no pool to size a batch by
        assert sub.node_resources == {}
        assert sub.max_batch_size("decode") is None

    def test_constructs_mtp_off_and_on(self):
        for k in (0, 2):
            _model, lm, cfg = _build_model(seed=30, mtp_num_draft_tokens=k)
            sub = Glm5NextLLMSubmodule(language_model=lm, config=cfg)
            # The walk stays plain single-token regardless of k: the
            # submodule carries no MTP machinery, so both builds are valid.
            assert (lm.mtp is None) == (k == 0)
            assert sub.can_batch(None, []) is True

    def test_torch_compile_disabled_by_default(self):
        # The eager prefill forward hosts the KDA span loop; compiling the
        # captured decode is opt-in.
        assert Glm5NextLLMSubmodule.disable_torch_compile is True

    def test_create_submodule_builds_on_reduced_config(self, tmp_path, monkeypatch):
        """The real meta-build -> to_empty(cpu) -> load -> process_weights_
        after_loading -> wrap sequence runs end to end on CPU, with the
        checkpoint read stubbed out (random-init the meta-materialized module
        instead)."""
        model = Glm5NextModel("zai-org/GLM-5.3-Flash", config_variant="reduced")
        # A bare dir: no index.json (generic driver branch), no config.json
        # (_maybe_apply_checkpoint_quant_config is a no-op, bf16 stands).
        monkeypatch.setattr(model, "_resolve_checkpoint", lambda: str(tmp_path))

        def _fake_load(self, language_model, source, device, tp_group):
            with torch.no_grad():
                for param in language_model.parameters():
                    if param.is_floating_point():
                        param.normal_(0, 0.02)

        monkeypatch.setattr(Glm5NextModel, "_load_checkpoint", _fake_load)

        sub = model._create_submodule("LLM", device="cpu")
        assert isinstance(sub, Glm5NextLLMSubmodule)
        assert sub.config is model.config
        assert sub.lm_head is sub.language_model.lm_head
        assert not sub.language_model.training
        # The declared conv-tail dtype follows the loaded projections, so the
        # slot-state spec the engine builds next matches the layer.
        assert model.kda_conv_dtype == torch.float32
        kda_spec = next(s for s in model.get_node_resources() if s.resource_key == KDA_STATE)
        assert kda_spec.config.tensors["conv"].dtype == torch.float32
        assert sub.node_resources == {}

        # Non-LLM nodes have no submodule; a dummy build (no checkpoint) is None.
        assert model._create_submodule("EMBED", device="cpu") is None
        monkeypatch.setattr(model, "_resolve_checkpoint", lambda: None)
        assert model._create_submodule("LLM", device="cpu") is None


# --- (1) declare_step + preprocess ------------------------------------------


class TestDeclareStep:
    def test_prefill_declares_chunk_walk_and_tracks_prompts(self):
        _model, lm, cfg = _build_model(seed=1)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        ids = {"a": torch.arange(5), "b": torch.arange(3)}
        inputs = [ARNodeInputs(input_ids=t, input_seq_len=t.shape[0]) for t in ids.values()]

        step = sub.declare_step("prefill", list(ids), inputs)

        assert isinstance(step, SubmoduleStep)
        assert step.segments == [Segment("a", LABEL, 5), Segment("b", LABEL, 3)]
        assert step.cg_key_info is None
        assert set(step.keys()) == {KV_CACHE, ATTN, SAMPLER, KDA_STATE}
        assert isinstance(step.get(KV_CACHE), KVStep)
        assert isinstance(step.get(ATTN), AttentionStep)
        assert step.get(ATTN).causal is True
        sampler = step.get(SAMPLER)
        assert isinstance(sampler, SamplerStep)
        assert sampler.apply_penalty is True
        # the prompt goes to the sampler's seen-token mask, by identity
        assert list(sampler.prefill_tracked_tokens) == ["a", "b"]
        assert sampler.prefill_tracked_tokens["a"] is ids["a"]
        assert sampler.prefill_tracked_tokens["b"] is ids["b"]
        kda = step.get(KDA_STATE)
        assert isinstance(kda, SlotStateStep)
        assert kda.mode == "chunk"
        assert kda.commit is True
        # the envelope hands its segment list to every resource step
        for key in (KV_CACHE, ATTN, SAMPLER, KDA_STATE):
            assert step.get(key).segments is step.segments, key

    def test_decode_declares_single_token_step_and_tracks_nothing(self):
        _model, lm, cfg = _build_model(seed=1)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        inputs = [ARNodeInputs(input_ids=torch.tensor([7]), input_seq_len=1) for _ in range(3)]

        step = sub.declare_step("decode", ["a", "b", "c"], inputs)

        assert step.segments == [Segment(r, LABEL, 1) for r in ("a", "b", "c")]
        assert step.get(KDA_STATE).mode == "step"
        assert step.get(SAMPLER).prefill_tracked_tokens == {}
        assert step.get(ATTN).causal is True

    def test_padding_rows_declare_their_own_segments(self):
        # Under a lease the engine declares over the PADDED rids; the
        # submodule declares a segment per row, padding included, and lets
        # the resources tell real from padding by ctx.request_ids.
        _model, lm, cfg = _build_model(seed=1)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        rids = ["r0", "__cg_x_0__", "__cg_x_1__"]
        inputs = [ARNodeInputs(input_ids=torch.tensor([3]), input_seq_len=1), _padding_input(), _padding_input()]
        step = sub.declare_step("decode", rids, inputs)
        assert [s.request_id for s in step.segments] == rids
        assert [s.span for s in step.segments] == [1, 1, 1]

    def test_rid_input_mismatch_is_loud(self):
        _model, lm, cfg = _build_model(seed=1)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        with pytest.raises(ValueError):
            sub.declare_step("decode", ["a", "b"], [_padding_input()])


class TestPreprocess:
    def test_returns_only_input_ids(self):
        # The one per-replay-varying input; everything else the forward reads
        # is a resource plan, which is what lets a captured decode repoint.
        _model, lm, cfg = _build_model(seed=2)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        a, b = torch.tensor([1, 2, 3]), torch.tensor([9])
        inputs = [ARNodeInputs(input_ids=a, input_seq_len=3), ARNodeInputs(input_ids=b, input_seq_len=1)]
        engine_inputs = ModelInputsFromEngine(request_ids=["a", "b"], per_request_info={})
        for walk in ("prefill", "decode"):
            pre = sub.preprocess(walk, engine_inputs, inputs)
            assert set(pre) == {"input_ids"}
            assert torch.equal(pre["input_ids"], torch.tensor([1, 2, 3, 9]))
            assert pre["input_ids"].dtype == torch.long

    def test_prepare_inputs_passes_at_index_topk_unbound(self):
        # No resources bound (loader-time submodule): committed = 0, so the
        # whole budget is the prompt.
        _model, lm, cfg = _build_model(seed=2)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        limit = cfg.index_topk
        ids = torch.zeros(limit, dtype=torch.long)
        out = sub.prepare_inputs("prefill", _info("r", "prefill"), {"text_inputs": [ids]})
        assert isinstance(out, ARNodeInputs)
        assert out.input_ids is ids
        assert out.input_seq_len == limit
        with pytest.raises(RuntimeError, match="index_topk"):
            sub.prepare_inputs(
                "prefill", _info("r", "prefill"),
                {"text_inputs": [torch.zeros(limit + 1, dtype=torch.long)]},
            )

    def test_prepare_inputs_counts_committed_tokens(self):
        """The cap is on the CONTEXT: committed (from the KDA resource) plus
        this step's tokens. Refusing raises, which fails only that rid — and
        leaks nothing, because the guard runs before any admit."""
        h = _Harness(seed=3)
        h.ingest("r", "other")
        limit = h.cfg.index_topk
        h.step("prefill", {"r": torch.arange(4)})
        assert h.kda.committed("r") == 4
        tracked_before = h.kda.tracked_requests()
        free_before = h.kda.num_free

        # boundary: 4 committed + (limit - 4) new == limit passes
        out = h.prepare("decode", "r", torch.zeros(limit - 4, dtype=torch.long))
        assert out.input_seq_len == limit - 4
        # one past it is refused, naming the regime
        with pytest.raises(RuntimeError, match=r"exceeds index_topk"):
            h.prepare("decode", "r", torch.zeros(limit - 3, dtype=torch.long))
        # another request with no history still gets the full budget
        assert h.prepare("prefill", "other", torch.zeros(limit, dtype=torch.long)).input_seq_len == limit

        assert h.kda.tracked_requests() == tracked_before
        assert h.kda.num_free == free_before
        assert h.kda.committed("r") == 4
        h.remove("r", "other")


# --- (3) capture policy ------------------------------------------------------


class TestCudaGraphs:
    def test_reference_moe_returns_no_graphs(self):
        # fp8 reference dispatch (per-hit-expert loop hosting .nonzero()) is
        # uncapturable; so is the naive bf16 per-expert loop under TP > 1.
        _model, lm, cfg = _build_model(seed=33)
        cfg.quantization_config = Glm5NextModelConfig.reduced_fp8().quantization_config
        assert cfg.moe_fp8_resident is True
        sub = Glm5NextLLMSubmodule(lm, cfg)
        assert all(blk._use_fused is False for blk in _moe_blocks(lm))
        assert sub._moe_resolved_fused() is False
        assert sub._moe_capture_blocked(tp_world_size=1) is True
        assert sub.get_cuda_graph_configs(DEVICE, tp_world_size=8) == []
        assert sub.get_cuda_graph_configs(DEVICE, tp_world_size=1) == []

        _model, lm, cfg = _build_model(seed=33)
        assert cfg.quantization_config is None
        sub = Glm5NextLLMSubmodule(lm, cfg)
        assert sub._moe_capture_blocked(tp_world_size=2) is True
        assert sub.get_cuda_graph_configs(DEVICE, tp_world_size=2) == []
        # bf16 on one rank has no host loop in the way: capturable
        assert sub._moe_capture_blocked(tp_world_size=1) is False

    @pytest.mark.parametrize("max_slots", [3, MAX_SLOTS])
    def test_fused_moe_returns_one_decode_graph_clamped_to_pool(self, max_slots, monkeypatch):
        """The capture flip is entirely ``_use_fused``-driven."""
        monkeypatch.delenv("MSTAR_GLM53_GRAPH_COMPILE", raising=False)
        h = _Harness(seed=34, max_slots=max_slots)
        sub, lm = h.sub, h.lm
        sub.config.quantization_config = Glm5NextModelConfig.reduced_fp8().quantization_config
        assert sub.get_cuda_graph_configs(DEVICE, tp_world_size=8) == []

        for blk in _moe_blocks(lm):
            blk._use_fused = True
        assert sub._moe_resolved_fused() is True
        assert sub._moe_capture_blocked(tp_world_size=8) is False

        configs = sub.get_cuda_graph_configs(DEVICE, tp_world_size=8)
        assert len(configs) == 1
        (config,) = configs
        assert isinstance(config, BatchedCudaGraphConfig)
        assert config.capture_graph_walk == "decode"
        assert config.replay_graph_walks == ["decode"]
        assert config.capture_forward_method == "forward_batched"
        assert config.additional_key_info is None
        assert config.compile is False
        expected = [b for b in DEFAULT_CAPTURE_BATCH_SIZES if b <= max_slots]
        assert config.capture_batch_sizes == expected
        assert max(config.capture_batch_sizes) <= h.kda.max_slots == max_slots
        # one token per row: the padding rows clone this
        single = config.single_request_inputs
        assert isinstance(single, ARNodeInputs)
        assert single.input_seq_len == 1
        assert single.input_ids.shape == (1,) and single.input_ids.dtype == torch.long
        assert single.input_ids.device == DEVICE
        assert config.get_total_tokens(4) == [4]

    def test_unbound_submodule_keeps_every_default_bucket(self, monkeypatch):
        # No slot pool to clamp by until the engine binds the resources.
        monkeypatch.delenv("MSTAR_GLM53_GRAPH_COMPILE", raising=False)
        _model, lm, cfg = _build_model(seed=34)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        (config,) = sub.get_cuda_graph_configs(DEVICE, tp_world_size=1)
        assert config.capture_batch_sizes == list(DEFAULT_CAPTURE_BATCH_SIZES)

    def test_compile_flag_follows_env(self, monkeypatch):
        _model, lm, cfg = _build_model(seed=34)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        monkeypatch.delenv("MSTAR_GLM53_GRAPH_COMPILE", raising=False)
        assert sub.get_cuda_graph_configs(DEVICE)[0].compile is False
        monkeypatch.setenv("MSTAR_GLM53_GRAPH_COMPILE", "1")
        assert sub.get_cuda_graph_configs(DEVICE)[0].compile is True
        monkeypatch.setenv("MSTAR_GLM53_GRAPH_COMPILE", "0")
        assert sub.get_cuda_graph_configs(DEVICE)[0].compile is False

    def test_max_batch_size_is_the_slot_pool(self):
        h = _Harness(seed=36, max_slots=3)
        assert h.kda.max_slots == 3
        assert h.sub.max_batch_size("decode") == 3
        assert h.sub.max_batch_size("prefill") == 3


# --- (4) slow-postprocess path ----------------------------------------------


class TestPostprocessAndStop:
    def test_postprocess_rebinds_new_token_to_text_inputs(self):
        _model, lm, cfg = _build_model(seed=4)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        token = torch.tensor([17])
        outputs = {"new_token": [token]}
        sub.postprocess("r", _info("r"), outputs)
        assert outputs["text_inputs"] is outputs["new_token"]
        assert outputs["text_inputs"][0] is token
        # nothing emitted (admit failure): nothing rebound
        empty = {}
        sub.postprocess("r", _info("r"), empty)
        assert empty == {}

    def test_check_stop_eos_unless_ignored(self):
        _model, lm, cfg = _build_model(seed=4)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        eos = cfg.eos_token_ids[1]
        not_eos = next(t for t in range(cfg.vocab_size) if t not in cfg.eos_token_ids)
        for token in cfg.eos_token_ids:
            assert sub.check_stop("r", _info("r"), {"new_token": [torch.tensor([token])]}) == {"decode_loop"}
        assert sub.check_stop("r", _info("r"), {"new_token": [torch.tensor([not_eos])]}) == set()
        # ignore_eos comes off the sampler's request config
        assert sub.check_stop("r", _info("r", ignore_eos=True), {"new_token": [torch.tensor([eos])]}) == set()
        # nothing emitted: nothing to decide
        assert sub.check_stop("r", _info("r"), {}) == set()

    def test_check_stop_max_tokens_counts_the_prefill_token(self):
        # generated = decode iters + 2 (the prefill-emitted token + this step's)
        _model, lm, cfg = _build_model(seed=4)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        not_eos = next(t for t in range(cfg.vocab_size) if t not in cfg.eos_token_ids)
        out = {"new_token": [torch.tensor([not_eos])]}
        for iters in (0, 1, 2):
            assert sub.check_stop("r", _info("r", max_tokens=5, decode_iters=iters), out) == set(), iters
        assert sub.check_stop("r", _info("r", max_tokens=5, decode_iters=3), out) == {"decode_loop"}
        assert sub.check_stop("r", _info("r", max_tokens=5, decode_iters=9), out) == {"decode_loop"}
        # no loop count yet (first decode step): 0 + 2
        assert sub.check_stop("r", _info("r", max_tokens=2, decode_iters=None), out) == {"decode_loop"}
        assert sub.check_stop("r", _info("r", max_tokens=3, decode_iters=None), out) == set()
        # ignore_eos does not touch the budget
        assert sub.check_stop("r", _info("r", max_tokens=5, ignore_eos=True, decode_iters=3), out) == {"decode_loop"}


class TestDtypeAndCleanup:
    def test_to_refuses_post_load_dtype_cast(self, caplog):
        # The loader left mixed per-param dtypes (fp32 scales / Sinkhorn /
        # router bias next to bf16 compute); the engine's blanket
        # submodule.to(device, autocast_dtype) must not re-narrow them.
        _model, lm, cfg = _build_model(seed=5)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        before = {n: p.dtype for n, p in sub.named_parameters()}
        assert set(before.values()) == {torch.float32}

        with caplog.at_level(logging.INFO, logger="mstar.model.glm5_next.submodules"):
            assert sub.to(torch.float64) is sub
            assert sub.to(dtype=torch.bfloat16) is sub
            assert sub.to("cpu", torch.float64) is sub
        assert {n: p.dtype for n, p in sub.named_parameters()} == before
        assert sum("ignoring post-load dtype cast" in r.message for r in caplog.records) == 3
        # a bare device move still goes through
        assert sub.to("cpu") is sub
        assert sub.to(device="cpu") is sub
        assert {n: p.dtype for n, p in sub.named_parameters()} == before

    def test_cleanup_request_drops_submodule_state_only(self):
        # The submodule keeps no per-request state; cleanup is the base
        # request_states pop. The resources' state is the engine's to remove.
        h = _Harness(seed=6)
        h.ingest("r")
        h.step("prefill", {"r": torch.arange(3)})
        slot = h.kda.slot_of("r")
        assert slot is not None
        h.sub.request_state("r").add("x", 1)
        assert "r" in h.sub.request_states

        h.sub.cleanup_request("r")
        assert "r" not in h.sub.request_states
        assert h.kda.slot_of("r") == slot
        assert h.kda.committed("r") == 3
        assert h.kv._streams["r"][LABEL].stored_len == 3
        h.sub.cleanup_request("r")  # the engine may double-retire

        h.remove("r")
        assert h.kda.slot_of("r") is None
        assert h.kda.num_free == h.kda.max_slots
        assert "r" not in h.kv._streams


# --- (5) lifecycle through the runner ---------------------------------------


class TestEngineSeam:
    def test_prefill_decode_lifecycle_and_chunk_step_parity(self):
        h = _Harness(seed=7)
        h.ingest("r0")
        prompt = torch.tensor([5, 9, 13, 7, 21], dtype=torch.long)

        t1 = h.step("prefill", {"r0": prompt})["r0"]
        assert t1.shape == (1,) and t1.dtype == torch.long
        assert h.kda.committed("r0") == 5
        assert h.kda.slot_of("r0") not in (None, SINK_SLOT)
        assert h.kv._streams["r0"][LABEL].stored_len == 5
        assert h.last_step.get(KDA_STATE).mode == "chunk"

        t2 = h.step("decode", {"r0": t1})["r0"]
        assert h.last_step.get(KDA_STATE).mode == "step"
        t3 = h.step("decode", {"r0": t2})["r0"]
        assert h.kda.committed("r0") == 7
        assert h.kv._streams["r0"][LABEL].stored_len == 7
        assert h.kda.num_free == h.kda.max_slots - 1

        # postprocess seeds the next step from the emission
        outputs = {"new_token": [t3]}
        h.sub.postprocess("r0", _info("r0"), outputs)
        assert outputs["text_inputs"] is outputs["new_token"]

        # the whole sequence prefilled on a fresh request yields the same
        # third token: chunk-vs-step parity of the engine path
        h.ingest("ref")
        t3_ref = h.step("prefill", {"ref": torch.cat([prompt, t1, t2])})["ref"]
        assert t3_ref.item() == t3.item(), (t3_ref.item(), t3.item())

        h.remove("r0", "ref")
        assert h.kda.num_free == h.kda.max_slots
        assert h.kda.tracked_requests() == set()
        assert not h.kv._streams

    def test_batched_two_requests_match_isolated_and_free_all(self):
        """Two requests prefilled AND decoded jointly emit, at every step, the greedy
        tokens each emits alone — per-request slot / KV-stream isolation across a real
        batch, over the >1-row decode slot index.
        """
        lens = {"a": 7, "b": 4}
        num_decode = 4
        torch.manual_seed(39)
        vocab = Glm5NextModelConfig.reduced().vocab_size
        toks = {rid: torch.randint(0, vocab, (n + num_decode,)) for rid, n in lens.items()}

        def _run(h: _Harness, rids: list[str]) -> dict[str, list[int]]:
            # teacher-forced on each rid's own tokens; greedy emission per step
            h.ingest(*rids)
            emitted = {rid: [] for rid in rids}
            out = h.step("prefill", {rid: toks[rid][: lens[rid]] for rid in rids})
            for rid in rids:
                emitted[rid].append(out[rid].item())
            for s in range(num_decode):
                out = h.step("decode", {rid: toks[rid][lens[rid] + s : lens[rid] + s + 1] for rid in rids})
                for rid in rids:
                    emitted[rid].append(out[rid].item())
            return emitted

        joint = _Harness(seed=38, dtype=torch.float64)
        got = _run(joint, ["a", "b"])
        # one step, two rows: the planned slot index carried one slot per rid
        plan: SlotStatePlan = joint.kda.current_plan()
        assert plan.slot_index.tolist() == [joint.kda.slot_of("a"), joint.kda.slot_of("b")]
        for rid, n in lens.items():
            assert joint.kda.committed(rid) == n + num_decode
            assert joint.kv._streams[rid][LABEL].stored_len == n + num_decode
        joint.remove("a", "b")
        assert joint.kda.num_free == joint.kda.max_slots
        assert not joint.kv._streams

        for rid in lens:
            alone = _Harness(seed=38, dtype=torch.float64)  # same weights
            ref = _run(alone, [rid])
            assert got[rid] == ref[rid], (rid, got[rid], ref[rid])

    def test_padded_decode_reads_the_sink_and_commits_only_real_rows(self):
        """A captured replay pads the batch with ``__cg_*`` rows: the engine declares over
        ``padded_request_ids`` and runs the forward over them.
        """
        h = _Harness(seed=8)
        pad = "__cg_x_0__"
        # the dummy-row pool ingests padding rids up front, like real ones
        h.ingest("r0", pad)
        assert h.kda.slot_of(pad) is None  # ingest leases nothing
        prompt = torch.tensor([3, 1, 4, 1, 5, 9], dtype=torch.long)
        t1 = h.step("prefill", {"r0": prompt})["r0"]
        slot = h.kda.slot_of("r0")

        out = h.step("decode", {"r0": t1}, padded=["r0", pad])

        # one entry per padded rid, each a single token
        assert list(out) == ["r0", pad]
        assert all(t.shape == (1,) for t in out.values())
        # the plan: real row on its slot, padding row on the sink
        step = h.last_step
        assert step.ctx.request_ids == ("r0",)
        assert tuple(step.ctx.padded_request_ids) == ("r0", pad)
        assert [s.request_id for s in step.segments] == ["r0", pad]
        plan: SlotStatePlan = step.ctx.plan_results[KDA_STATE]
        assert plan.mode == "step"
        assert plan.slot_index.tolist() == [slot, SINK_SLOT]
        assert plan.request_ids == ["r0", pad]
        assert [s.real for s in plan.spans] == [True, False]
        assert plan.spans[0].ctx_start == len(prompt)
        assert plan.spans[1].ctx_start == 0
        # commit: the real row only; the padding row leased nothing
        assert h.kda.committed("r0") == len(prompt) + 1
        assert h.kda.committed(pad) == 0
        assert h.kda.slot_of(pad) is None
        assert h.kda.num_free == h.kda.max_slots - 1

        # the padding row did not perturb the real row: a fresh request
        # prefilled with the whole sequence lands on the same token
        h.ingest("ref")
        t2_ref = h.step("prefill", {"ref": torch.cat([prompt, t1])})["ref"]
        assert t2_ref.item() == out["r0"].item()

        # the dummy-row pool hands its rows back after capture
        for resource in h.resources.values():
            resource.reset_request(pad, free=True)
        assert h.kda.committed(pad) == 0
        h.remove("r0", "ref", pad)
        assert h.kda.num_free == h.kda.max_slots
        assert not h.kv._streams
