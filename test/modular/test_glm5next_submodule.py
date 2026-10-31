"""GLM-5.3-Flash's serving submodule on its real resources, on CPU: what it declares and
prepares, its capture policy and batch caps, its stop rule, and padded decode rows."""
from __future__ import annotations

import sys
import types

sys.path.insert(0, ".")

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
from mstar.engine.accelerator_graph_runner import DEFAULT_CAPTURE_BATCH_SIZES
from mstar.engine.resources import (
    SamplingReqConfig,
    Segment,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo, Resource, build_resource
from mstar.engine.resources.kv import manager as kv_mod
from mstar.engine.resources.linear_attn.kda import KDAPlan
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.resources.step import StepContext
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock
from mstar.model.glm5_next.config import (
    ATTN,
    KDA,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine

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
        request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=max_tokens,
        resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts=counts,
    )


def _slot_of(pool: RecurrentStatePool, rid: str) -> int | None:
    slot = pool._slots.get(rid, {}).get(LABEL)
    return None if slot is None else slot.index


def _holders(pool: RecurrentStatePool) -> set[str]:
    return {rid for rid, labels in pool._slots.items() if labels}


def _padding_input() -> ARNodeInputs:
    """What ``BatchedAcceleratorGraphConfig.get_node_inputs`` hands a padding row:
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
        self.pool = self.resources[KDA_STATE]
        self.kda = self.resources[KDA]
        self.last_step: SubmoduleStep | None = None

    def context(self, rid: str) -> int:
        """The tokens the submodule counts as the request's context."""
        state = self.sub.request_states.get(rid)
        return 0 if state is None else state.get("context", 0)

    @property
    def usable(self) -> int:
        return self.pool.config.usable_slots

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
            raw = self.sub.forward_batched(walk, engine_inputs, **pre)
        self.runner.commit(step)
        rows = raw.row_outputs["new_token"]
        out = {rid: {"new_token": [rows[i : i + 1]]} for i, rid in enumerate(step_rids)}
        # the engine's per-request tail, after the forward
        for rid, node_input in zip(rids, node_inputs, strict=False):
            self.sub.postprocess(rid, _info(rid, walk), dict(out[rid]), inputs=node_input)
        self.last_step = step
        return {rid: out[rid]["new_token"][0] for rid in out}


def _moe_blocks(lm: Glm5NextForCausalLM) -> list[Glm5NextSparseMoeBlock]:
    return [m for m in lm.modules() if isinstance(m, Glm5NextSparseMoeBlock)]


# --- construction, declare_step and preprocess ------------------------------


class TestConstruction:
    def test_create_submodule_builds_on_reduced_config(self, tmp_path, monkeypatch):
        """``_create_submodule``'s meta-build, load and wrap on CPU, the checkpoint read
        stubbed by a random init."""
        model = Glm5NextModel("zai-org/GLM-5.3-Flash", config_variant="reduced")
        # a bare dir: no index, and no config.json, so no quantization config
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
        # the declared conv-tail dtype follows the loaded projections, so the pool the engine
        # builds next matches the layer
        assert model.kda_conv_dtype == torch.float32
        kda_spec = next(s for s in model.get_node_resources() if s.resource_key == KDA_STATE)
        assert kda_spec.config.blocks["conv"].dtype == torch.float32
        assert sub.node_resources == {}

        # non-LLM nodes have no submodule, and a dummy build (no checkpoint) is None
        assert model._create_submodule("EMBED", device="cpu") is None
        monkeypatch.setattr(model, "_resolve_checkpoint", lambda: None)
        assert model._create_submodule("LLM", device="cpu") is None


class TestDeclareStep:
    def test_a_step_declares_every_resource_and_tracks_only_the_prompt(self):
        _model, lm, cfg = _build_model(seed=1)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        ids = {"a": torch.arange(5), "b": torch.arange(3)}
        inputs = [ARNodeInputs(input_ids=t, input_seq_len=t.shape[0]) for t in ids.values()]

        step = sub.declare_step("prefill", list(ids), inputs)

        assert step.segments == [Segment("a", LABEL, 5), Segment("b", LABEL, 3)]
        assert set(step.keys()) == {KV_CACHE, ATTN, SAMPLER, KDA_STATE, KDA}
        for key in step.keys():
            assert step.get(key).segments is step.segments, key
        # the prompt goes to the sampler's seen-token mask, for the repetition penalty
        sampler = step.get(SAMPLER)
        assert sampler.apply_penalty is True
        assert list(sampler.prefill_tracked_tokens) == ["a", "b"]
        assert all(sampler.prefill_tracked_tokens[rid] is ids[rid] for rid in ids)

        decode = sub.declare_step("decode", ["a", "b"], [_padding_input(), _padding_input()])
        assert decode.segments == [Segment(rid, LABEL, 1) for rid in ("a", "b")]
        assert decode.get(SAMPLER).prefill_tracked_tokens == {}

    def test_rid_input_mismatch_is_loud(self):
        _model, lm, cfg = _build_model(seed=1)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        with pytest.raises(ValueError):
            sub.declare_step("decode", ["a", "b"], [_padding_input()])


class TestPreprocess:
    def test_returns_only_input_ids(self):
        # the one input that varies per replay; everything else the forward reads is a
        # resource plan, which is what lets a captured decode repoint
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

    def test_prepare_inputs_counts_committed_tokens(self):
        """The cap is on the context, the tokens of the steps that ran plus this step's; the
        refusal runs before any admit, so it leaks nothing."""
        h = _Harness(seed=3)
        h.ingest("r", "other")
        limit = h.cfg.index_topk
        h.step("prefill", {"r": torch.arange(4)})
        assert h.context("r") == 4
        holders_before = _holders(h.pool)
        free_before = h.pool.num_free_slots

        # boundary: 4 committed + (limit - 4) new == limit passes
        out = h.prepare("decode", "r", torch.zeros(limit - 4, dtype=torch.long))
        assert out.input_seq_len == limit - 4
        # one past it is refused
        with pytest.raises(RuntimeError, match=r"exceeds index_topk"):
            h.prepare("decode", "r", torch.zeros(limit - 3, dtype=torch.long))
        # another request with no history still gets the full budget
        assert h.prepare("prefill", "other", torch.zeros(limit, dtype=torch.long)).input_seq_len == limit

        assert _holders(h.pool) == holders_before
        assert h.pool.num_free_slots == free_before
        assert h.context("r") == 4
        h.remove("r", "other")


# --- capture policy and batch caps -------------------------------------------


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
        assert sub.get_accelerator_graph_configs(DEVICE, tp_world_size=8) == []
        assert sub.get_accelerator_graph_configs(DEVICE, tp_world_size=1) == []

        _model, lm, cfg = _build_model(seed=33)
        assert cfg.quantization_config is None
        sub = Glm5NextLLMSubmodule(lm, cfg)
        assert sub._moe_capture_blocked(tp_world_size=2) is True
        assert sub.get_accelerator_graph_configs(DEVICE, tp_world_size=2) == []
        # bf16 on one rank has no host loop in the way: capturable
        assert sub._moe_capture_blocked(tp_world_size=1) is False

    @pytest.mark.parametrize("max_slots", [3, MAX_SLOTS])
    def test_fused_moe_captures_decode_at_the_batch_sizes_the_pool_holds(self, max_slots):
        h = _Harness(seed=34, max_slots=max_slots)
        sub, lm = h.sub, h.lm
        sub.config.quantization_config = Glm5NextModelConfig.reduced_fp8().quantization_config
        assert sub.get_accelerator_graph_configs(DEVICE, tp_world_size=8) == []
        for blk in _moe_blocks(lm):
            blk._use_fused = True

        (config,) = sub.get_accelerator_graph_configs(DEVICE, tp_world_size=8)
        assert isinstance(config, BatchedAcceleratorGraphConfig)
        assert config.capture_graph_walk == "decode"
        assert config.capture_batch_sizes == [b for b in DEFAULT_CAPTURE_BATCH_SIZES if b <= max_slots]
        # the padding rows clone this one-token input
        assert config.single_request_inputs.input_seq_len == 1

    @pytest.mark.parametrize("fused_kda", [False, True])
    def test_prefill_rows_follow_the_captured_prefill(self, monkeypatch, fused_kda):
        """Prefill is captured only on the fused KDA kernels, and a captured prefill holds a
        step to its largest graph's rows: a larger step would run eager and compile the MoE
        decode kernels for its row count mid-serve. An eager prefill is held to the free
        slots alone."""
        from mstar.model.glm5_next import submodules

        h = _Harness(seed=35)
        sub = h.sub
        monkeypatch.setattr(submodules, "_fused_kda", lambda config: fused_kda)
        sub.config.prefill_graphs = True
        walks = [c.capture_graph_walk for c in sub.get_accelerator_graph_configs(DEVICE)]
        assert walks == (["decode", "prefill"] if fused_kda else ["decode"])
        rows = max(sub.config.prefill_capture_batch_sizes)
        assert sub.max_batch_size("prefill") == (rows if fused_kda else h.usable)
        assert sub.max_batch_size("decode") == h.usable
        # no graphs at all (the fp8 reference MoE): prefill runs eager
        sub.config.quantization_config = Glm5NextModelConfig.reduced_fp8().quantization_config
        assert sub.get_accelerator_graph_configs(DEVICE) == []
        assert sub.max_batch_size("prefill") == h.usable


# --- check_stop and dtypes ---------------------------------------------------


class TestCheckStop:
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


class TestDtype:
    def test_to_refuses_post_load_dtype_cast(self):
        # the loader leaves fp32 params beside bf16 ones; the engine's blanket
        # submodule.to(device, dtype) must not narrow them
        _model, lm, cfg = _build_model(seed=5)
        sub = Glm5NextLLMSubmodule(lm, cfg)
        before = {n: p.dtype for n, p in sub.named_parameters()}
        assert set(before.values()) == {torch.float32}

        assert sub.to(torch.float64) is sub
        assert sub.to(dtype=torch.bfloat16) is sub
        assert sub.to("cpu", torch.float64) is sub
        assert {n: p.dtype for n, p in sub.named_parameters()} == before
        # a bare device move still goes through
        assert sub.to("cpu") is sub
        assert sub.to(device="cpu") is sub
        assert {n: p.dtype for n, p in sub.named_parameters()} == before


# --- padded decode rows ------------------------------------------------------


class TestPaddedRows:
    def test_padded_decode_reads_the_sink_and_commits_only_real_rows(self):
        """A captured replay pads the batch with ``__cg_*`` rows, declared and run like real
        ones: they read the sink, and only the real rows commit."""
        h = _Harness(seed=8)
        pad = "__cg_x_0__"
        # the dummy-row pool ingests padding rids up front, like real ones
        h.ingest("r0", pad)
        assert _slot_of(h.pool, pad) is None  # ingest leases nothing
        prompt = torch.tensor([3, 1, 4, 1, 5, 9], dtype=torch.long)
        t1 = h.step("prefill", {"r0": prompt})["r0"]
        slot = _slot_of(h.pool, "r0")

        out = h.step("decode", {"r0": t1}, padded=["r0", pad])

        # one entry per padded rid, each a single token
        assert list(out) == ["r0", pad]
        assert all(t.shape == (1,) for t in out.values())
        # the plan: real row on its slot, padding row on the sink
        step = h.last_step
        assert step.ctx.request_ids == ("r0",)
        assert tuple(step.ctx.padded_request_ids) == ("r0", pad)
        assert [s.request_id for s in step.segments] == ["r0", pad]
        plan: KDAPlan = step.ctx.plan_results[KDA][LABEL]
        assert plan.is_decode and plan.spans == (1, 1)
        assert plan.slot_ids.tolist() == [slot, SINK_SLOT]
        # the real row resumes its state; the padding row reads the sink as zeros
        assert plan.has_state.tolist() == [True, False]
        # commit: the real row only; the padding row leased nothing
        assert h.context("r0") == len(prompt) + 1
        assert h.context(pad) == 0
        assert _slot_of(h.pool, pad) is None
        assert h.pool.num_free_slots == h.usable - 1

        # the padding row did not perturb the real row: a fresh request
        # prefilled with the whole sequence lands on the same token
        h.ingest("ref")
        t2_ref = h.step("prefill", {"ref": torch.cat([prompt, t1])})["ref"]
        assert t2_ref.item() == out["r0"].item()

        # the dummy-row pool hands its rows back after capture
        for resource in h.resources.values():
            resource.reset_request(pad, free=True)
        h.remove("r0", "ref", pad)
        assert h.pool.num_free_slots == h.usable
        assert not h.kv._streams
