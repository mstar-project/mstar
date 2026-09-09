"""GLM-5.3-Flash assembled-model tests on the resource-pool engine — CPU,
reduced config, no GPU deps.

Port of the lane's ``test_glm5next_model.py`` (deleted-engine era: flat
token batches over ``glm52._testing.ReferenceCacheHandle`` + model-owned KDA
state). Every numerical parity assertion is kept; what changed is the
harness: the model is driven the way the v1 engine drives it — its
``get_node_resources()`` specs are built into the REAL resources (MLA-layout
KV cache, MLA attention backend on its fp32 SDPA fallback, slot-state
resource, sampler), bound into the layers, and every step goes
declare -> admit -> plan -> forward -> commit through a ``StepRunner``.

What this file pins:

1. Prefill vs stepwise decode emit the same logits through the full
   hybrid stack (KDA chunk kernel vs recurrent step, MLA over the paged
   latent cache, mHC threading, clamped MoE) at batch > 1 — including
   across a KDA-state snapshot/restore + KV rewind, which replays
   bit-exactly (the M2 verify-rewind primitive), now through
   ``Glm5NextKdaStateAccess`` over the engine's slot pool.
2. The slot-state lifecycle in its v1 form: lease at first admit, a
   single-token step before any chunk refused, commit after the forward
   (a planned-but-never-run step commits nothing — what replaces
   ``abort_context``), reset, and idempotent removal.
3. The ``layer_types`` schedule is honored structurally AND behaviorally
   (KDA layers never touch the KV resource; the two full-attention layers
   write and read compact planes 0 and 1, once each, in order).
4. The registry entry resolves and ``Glm5NextModel`` declares the four
   node resources (KV planes = full-attention count, MLA layout, latent =
   kv_lora_rank + the 64-wide zero kpe pad; KDA pool shapes).
5. MTP-off and MTP-on both construct; the layer-45 module's call contract
   holds against the real resources (it addresses its own draft plane);
   the loader's expected parameter tree matches the assembled module tree
   name-for-name and a raw checkpoint stream round-trips bit-for-bit.
6. The engine-path smoke (prepare_inputs -> declare -> preprocess ->
   forward_batched with a greedy stand-in for the Triton sampler): the
   same token trajectory generated stepwise and prefilled whole
   (``test_glm5next_engine_path.py`` folded in; that file stays).

Dropped, with no v1 analogue (each noted again at the site):
``alloc_kda_state`` double-alloc refusal (a slot lease is idempotent by
design), ``prefill_context(ctx_starts=)`` desync refusal (the resource owns
the committed counter — a caller cannot name a gap), ``abort_context``
(nothing to roll back: commit runs after the forward),
``get_node_engine_types``/``EngineType`` (one engine now), and the
``mla_absorb=False`` naive fallback (the MLA backend's SDPA fallback serves
reduced configs on the absorbed path).

Per lane ground rule 4, the flashinfer CPU stub is a fixture
(``run_rms_norm`` imports flashinfer per call, so a monkeypatched
``sys.modules`` entry scopes it to each test) — never a module-level
``sys.modules`` write. The KV transfer manager is stubbed the same way (no
transfer engine on a laptop).
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.engine.resources import (  # noqa: E402
    AttentionSpec,
    AttnBackend,
    KVLayout,
    KVSpec,
    SamplerSpec,
    SamplingReqConfig,
    SlotStateSpec,
    SlotStateStep,
    StepRunner,
    resolve_spec_dependencies,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource  # noqa: E402
from mstar.engine.resources.kv import manager as kv_mod  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.model.glm5_next.components.attention import (  # noqa: E402
    Glm5NextKdaAttention,
    Glm5NextMLAAttention,
)
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM  # noqa: E402
from mstar.model.glm5_next.components.moe import (  # noqa: E402
    Glm5NextGatedMLP,
    Glm5NextSparseMoeBlock,
)
from mstar.model.glm5_next.config import (  # noqa: E402
    ATTN,
    FULL_ATTENTION,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    LINEAR_ATTENTION,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.glm5_next.glm5_next_model import (  # noqa: E402
    Glm5NextModel,
    process_weights_after_loading,
)
from mstar.model.glm5_next.kda_state import (  # noqa: E402
    CONV,
    RECURRENT,
    Glm5NextKdaStateAccess,
)
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine  # noqa: E402

# The chunked-prefill and recurrent-decode paths are mathematically equal but
# numerically distinct (different reduction structure). In float64 their max
# logits delta on this reduced geometry is ~4e-7 on EVERY platform — that is
# the algorithmic-equivalence signal the cross-path parity test asserts, so it
# runs in float64 (LOGITS_ATOL_F64). In fp32 the delta is platform-dependent
# and much larger through the reduced config's ~3x/layer noise gain: ~4e-7 on
# Apple-Accelerate (dev laptop) but ~5e-3 on x86-MKL (Laude CI box, 2026-08-31)
# — an fp32 tolerance that held on the laptop failed box-side, and papering it
# with a ~1e-2 fp32 atol would erode the bug margin (real bugs — gate wiring,
# state desync, op order — land at O(0.1-1)). LOGITS_ATOL (fp32) stays for the
# same-path bitwise/greedy checks that are platform-stable.
#
# The MLA backend's SDPA fallback computes in fp32 whatever the model dtype
# (as the lane's ReferenceCacheHandle did), so the float64 run isolates the
# KDA chunk-vs-recurrent structure exactly as before.
LOGITS_ATOL = 1e-3
LOGITS_ATOL_F64 = 1e-5
_WEIGHT_STD = 0.03

# Small pages so a reduced prefill spans several: the paged latent gather of
# the MLA fallback and the page-boundary bookkeeping of the KV resource are
# on the path the parity tests measure. (The serve YAML tunes these under
# ``resources: kv_cache:`` — this is the same override hook.)
_PAGE_SIZE = 8
_MAX_PAGES = 32


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


class _StubTransfer:
    """No transfer engine on a laptop: the KV resource's transfer manager is
    replaced per test (``KVManager.__init__`` builds one unconditionally)."""

    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self):
        return None

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _cpu_flashinfer(monkeypatch):
    """CPU ``flashinfer.norm.rmsnorm`` so RMSNorm-backed forwards run here.

    Forced per test (not deferred to an import-time guard) — the glm52
    suite's lesson: on a box with real flashinfer an earlier import wins
    and these CPU tensors hit GPU kernels."""
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setattr(kv_mod, "KVTransferManager", _StubTransfer)


def _randomize(model: Glm5NextForCausalLM, dtype: torch.dtype) -> None:
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point() and param.ndim >= 2:
                param.data.normal_(0, _WEIGHT_STD)
    if dtype != torch.float32:
        model.to(dtype)
    # The real load sequence: load -> process_weights_after_loading -> eval.
    # The MLA layers build their absorbed w_kc/w_vc (+ fused q/kv_a) here;
    # after the dtype cast so they carry the model dtype.
    process_weights_after_loading(model, torch.device("cpu"))
    model.eval()
    model.requires_grad_(False)


def _build_model(
    seed: int, mtp_num_draft_tokens: int = 0, dtype: torch.dtype = torch.float32
) -> tuple[Glm5NextForCausalLM, Glm5NextModelConfig]:
    """Reduced-config model with small random weights, no engine attached.

    ``dtype`` widens the whole model — float64 for the cross-path parity test
    (see LOGITS_ATOL_F64), float32 elsewhere.
    """
    torch.manual_seed(seed)
    cfg = Glm5NextModelConfig.reduced()
    cfg.mtp_num_draft_tokens = mtp_num_draft_tokens
    model = Glm5NextForCausalLM(cfg)
    _randomize(model, dtype)
    return model, cfg


class _GreedySampler:
    """Argmax in place of the Triton sampler for the folded engine-path
    smoke (``forward_batched`` samples inside the forward)."""

    def sample(self, request_ids, logits, **kwargs):
        return logits.argmax(-1)


class _Harness:
    """A reduced ``Glm5NextForCausalLM`` behind the REAL v1 resources.

    Built exactly as the engine builds them: ``Glm5NextModel.get_node_resources``
    -> ``resolve_spec_dependencies`` -> ``build_resource`` per spec ->
    ``StepRunner`` -> ``bind_node_resources`` into the layers. Each step is
    the engine's protocol (``declare_step`` -> admit -> plan -> forward ->
    commit); the sampler step is dropped from the declaration — the Triton
    sampler cannot run on CPU and these tests want raw logits — so the
    forward is the trunk + lm_head, the same tensors
    ``Glm5NextLLMSubmodule._forward`` hands the sampler.
    """

    def __init__(
        self,
        seed: int,
        dtype: torch.dtype = torch.float32,
        mtp_num_draft_tokens: int = 0,
        max_slots: int = 4,
    ) -> None:
        torch.manual_seed(seed)
        self.model = Glm5NextModel(
            "x", config_variant="reduced", kda_conv_dtype=dtype,
            kda_max_requests=max_slots, mtp_num_draft_tokens=mtp_num_draft_tokens,
        )
        self.cfg = self.model.config
        self.lm = Glm5NextForCausalLM(self.cfg)
        _randomize(self.lm, dtype)
        self.sub = Glm5NextLLMSubmodule(self.lm, self.cfg)

        specs = self.model.get_node_resources()
        for spec in specs:
            if isinstance(spec, KVSpec):
                spec.apply_yaml_overrides(page_size=_PAGE_SIZE, max_num_pages=_MAX_PAGES)
        by_key = resolve_spec_dependencies(specs)
        self.resources = {}
        for spec in specs:
            info = EngineResourceInfo(
                device=torch.device("cpu"), kv_dtype=dtype,
                dependencies={k: by_key[k] for k in spec.depends_on()},
            )
            self.resources[spec.resource_key] = build_resource(spec, info)
        self.runner = StepRunner(self.resources, node_resources={"LLM": list(self.resources)})
        self.sub.bind_node_resources(self.resources)

        self.kv = self.resources[KV_CACHE]
        self.attn = self.resources[ATTN]
        self.kda = self.resources[KDA_STATE]
        self.kda_access = Glm5NextKdaStateAccess(self.kda)

    # -- request lifecycle ------------------------------------------------

    def ingest(self, *request_ids: str) -> None:
        for rid in request_ids:
            self.runner.ingest_request(rid, {})

    def remove(self, *request_ids: str) -> None:
        for rid in request_ids:
            self.runner.remove_request(rid)

    def stored_len(self, rid: str) -> int:
        return self.kv._streams[rid][LABEL].stored_len

    def rewind_kv(self, rid: str, n: int) -> None:
        """The lane's ``handle.rewind_seq_lens``: v1 has no rewind on the KV
        resource yet (the M2 verify loop will add one), and a paged rewind is
        the stream's ``stored_len`` — its pages stay leased as a high-water
        mark and heal by overwrite."""
        self.kv._streams[rid][LABEL].stored_len -= n

    # -- one engine step --------------------------------------------------

    def open_step(self, walk: str, request_ids: list[str], token_rows: list[torch.Tensor]):
        """declare -> admit -> plan; returns ``(step, node_inputs)`` with the
        resources planned for a forward. ``close_step`` commits."""
        node_inputs = [
            ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0]) for ids in token_rows
        ]
        step = self.sub.declare_step(walk, request_ids, node_inputs)
        step.set_ctx(StepContext(
            request_ids=tuple(request_ids), graph_walk=walk, slot=0, capture=False,
        ))
        step.steps.pop(SAMPLER)  # see the class docstring
        outcome = self.runner.admit(step)
        assert outcome.ok, outcome.reason
        self.runner.plan(step)
        return step, node_inputs

    def close_step(self, step) -> None:
        self.runner.commit(step)

    def engine_inputs(self, request_ids: list[str]) -> ModelInputsFromEngine:
        return ModelInputsFromEngine(
            request_ids=list(request_ids), per_request_info={}, resources=self.resources,
        )

    def _logits_step(self, walk, request_ids, token_rows) -> torch.Tensor:
        step, node_inputs = self.open_step(walk, request_ids, token_rows)
        pre = self.sub.preprocess(walk, self.engine_inputs(request_ids), node_inputs)
        logits = self.lm(pre["input_ids"])  # (T, vocab): trunk + lm_head
        self.close_step(step)
        return logits

    def prefill(self, request_ids, token_rows) -> list[torch.Tensor]:
        """One flattened prefill step; returns per-request logit chunks."""
        logits = self._logits_step("prefill", request_ids, token_rows)
        out, row = [], 0
        for rows in token_rows:
            out.append(logits[row : row + rows.shape[0]])
            row += rows.shape[0]
        return out

    def decode(self, request_ids, tokens: torch.Tensor) -> torch.Tensor:
        """One joint single-token step; ``tokens`` is ``[bs]``."""
        return self._logits_step(
            "decode", request_ids, [tokens[i : i + 1] for i in range(len(request_ids))],
        )


# --- (1) prefill == stepwise decode, batched, incl. save/restore ------------


@torch.no_grad()
def test_prefill_matches_stepwise_decode_batched():
    """Chunked prefill and recurrent decode agree over the same tokens.

    Reference: each request's FULL sequence prefilled alone in one chunk.
    Subject: both requests prefilled together (flat varlen batch), then
    teacher-forced joint decode steps — exercising per-request KDA slot
    isolation, the batched gather/scatter decode path by the planned slot
    index, and the paged MLA latent cache, at batch > 1 (lane generality
    rule).
    """
    # float64: isolates the chunk-vs-recurrent algorithmic equivalence from
    # platform fp32 rounding (which the reduced config amplifies ~3x/layer).
    h = _Harness(seed=10, dtype=torch.float64)
    cfg = h.cfg
    prefill_lens = {"r0": 11, "r1": 7}
    num_decode = 4

    torch.manual_seed(11)
    tokens = {
        rid: torch.randint(0, cfg.vocab_size, (length + num_decode,))
        for rid, length in prefill_lens.items()
    }

    # Reference: one full-sequence prefill per request, isolated slot.
    reference = {}
    for rid, ids in tokens.items():
        ref_id = f"ref-{rid}"
        h.ingest(ref_id)
        (reference[rid],) = h.prefill([ref_id], [ids])
        assert h.kda.committed(ref_id) == ids.shape[0] == h.stored_len(ref_id)
        h.remove(ref_id)
    assert h.kda.tracked_requests() == set()  # no slot leaks
    assert h.kda.num_free == h.kda.max_slots

    # Subject: joint prefill, then joint stepwise decode.
    request_ids = list(prefill_lens)
    h.ingest(*request_ids)
    prefill_logits = h.prefill(
        request_ids, [tokens[rid][: prefill_lens[rid]] for rid in request_ids])
    slots = {rid: h.kda.slot_of(rid) for rid in request_ids}
    assert len(set(slots.values())) == 2 and 0 not in slots.values()  # not the sink

    for rid, got in zip(request_ids, prefill_logits, strict=True):
        want = reference[rid][: prefill_lens[rid]]
        assert torch.allclose(got, want, atol=LOGITS_ATOL_F64, rtol=0)

    for step in range(num_decode):
        positions = [prefill_lens[rid] + step for rid in request_ids]
        step_tokens = torch.stack(
            [tokens[rid][positions[i]] for i, rid in enumerate(request_ids)])
        logits = h.decode(request_ids, step_tokens)
        for i, rid in enumerate(request_ids):
            want = reference[rid][positions[i]]
            assert torch.allclose(logits[i], want, atol=LOGITS_ATOL_F64, rtol=0), (
                f"{rid} decode step {step}: max|d|="
                f"{(logits[i] - want).abs().max().item():.3e}"
            )
            # Same greedy trajectory — the M1 parity bar in miniature.
            assert int(logits[i].argmax()) == int(want.argmax())

    for rid in request_ids:
        assert h.kda.committed(rid) == prefill_lens[rid] + num_decode
        assert h.stored_len(rid) == prefill_lens[rid] + num_decode
    h.remove(*request_ids)
    assert h.kda.num_free == h.kda.max_slots


@torch.no_grad()
def test_decode_replays_bitwise_across_kda_snapshot_restore():
    """snapshot -> decode k -> restore + KV rewind -> decode k again: the
    replay is BIT-exact (same states, same ops) — the M2 verify-rewind
    primitive (KV-plane rewind is free; recurrent state is not, hence the
    snapshot), and the strongest possible save/restore check. The snapshot
    now reads the engine's slot pool through ``Glm5NextKdaStateAccess``."""
    h = _Harness(seed=12)
    cfg = h.cfg
    prefill_len, num_decode = 9, 3
    torch.manual_seed(13)
    ids = torch.randint(0, cfg.vocab_size, (prefill_len + num_decode,))

    h.ingest("r0")
    h.prefill(["r0"], [ids[:prefill_len]])

    snap = h.kda_access.snapshot("r0")
    assert snap.committed == prefill_len
    assert snap.recurrent.shape == (len(cfg.kda_layer_indices), *h.kda.pool(RECURRENT).shape[2:])

    def run_decode():
        out = []
        for step in range(num_decode):
            pos = prefill_len + step
            out.append(h.decode(["r0"], ids[pos : pos + 1]))
        return torch.cat(out)

    first = run_decode()
    assert h.kda_access.committed_tokens("r0") == prefill_len + num_decode
    assert h.stored_len("r0") == prefill_len + num_decode

    h.kda_access.restore("r0", snap)
    assert h.kda_access.committed_tokens("r0") == prefill_len
    h.rewind_kv("r0", num_decode)  # paged-KV rewind heals by overwrite
    replay = run_decode()

    assert torch.equal(first, replay)
    h.remove("r0")


def test_kda_state_lifecycle_is_explicit_and_loud():
    """The slot-state lifecycle, driven on the model's own declared pool.

    v1 deltas vs the lane's model-owned store, each asserted in its new
    form: the lease is idempotent (no "already allocated" — a re-admitted
    rid keeps its slot), a single-token step before any chunk is refused
    by ``plan``, counts advance at COMMIT (a planned step whose forward
    never ran leaves them untouched — what ``abort_context`` did), a
    ``commit=False`` step reads without advancing, ``reset_request``
    rewinds to 0 and zeroes the slot, removal is idempotent. No analogue:
    ``prefill_context(ctx_starts=)`` desync — the resource owns the
    committed counter, so a caller cannot name a gap.
    """
    h = _Harness(seed=14, max_slots=2)
    kda = h.kda
    ids = torch.zeros(4, dtype=torch.long)
    h.ingest("r0")
    assert kda.slot_of("r0") is None  # ingest registers, admit leases

    # Decode before any committed prefill token is a scheduling bug.
    with pytest.raises(RuntimeError, match="no committed tokens"):
        h.open_step("decode", ["r0"], [ids[:1]])

    step, _ = h.open_step("prefill", ["r0"], [ids])
    slot = kda.slot_of("r0")
    assert slot is not None and slot != 0
    assert kda.committed("r0") == 0  # counts advance at COMMIT, not at plan
    h.close_step(step)
    assert kda.committed("r0") == 4

    # Chunked continue at the exact boundary: the plan reads the committed
    # count as ctx_start; the same slot is kept (idempotent lease).
    step, _ = h.open_step("prefill", ["r0"], [ids[:2]])
    (span,) = kda.current_plan().spans
    assert (span.slot, span.ctx_start, span.q_len, span.real) == (slot, 4, 2, True)
    h.close_step(step)
    assert kda.committed("r0") == 6 and kda.slot_of("r0") == slot

    # A planned-but-dropped step (its forward never ran) commits nothing:
    # both phases. (The lane rolled these back with abort_context.)
    h.open_step("decode", ["r0"], [ids[:1]])
    assert kda.committed("r0") == 6
    h.open_step("prefill", ["r0"], [ids[:3]])
    assert kda.committed("r0") == 6
    # A step declared commit=False (speculative draft) reads and advances
    # nothing.
    step, _ = h.open_step("decode", ["r0"], [ids[:1]])
    step.steps[KDA_STATE] = SlotStateStep(segments=step.steps[KDA_STATE].segments, commit=False)
    h.close_step(step)
    assert kda.committed("r0") == 6

    kda.pool(RECURRENT)[:, slot].fill_(1.0)
    kda.reset_request("r0")
    assert kda.committed("r0") == 0
    assert kda.pool(RECURRENT)[:, slot].abs().sum() == 0  # zeroed, slot kept
    assert kda.slot_of("r0") == slot
    h.remove("r0")
    h.remove("r0")  # idempotent, engine may double-retire
    assert kda.num_free == 2 and kda.tracked_requests() == set()

    # Exhaustion is a retryable admit failure, not an exception.
    h.ingest("a", "b", "c")
    for rid in ("a", "b"):
        h.close_step(h.open_step("prefill", [rid], [ids[:1]])[0])
    node_in = [ARNodeInputs(input_ids=ids[:1], input_seq_len=1)]
    step = h.sub.declare_step("prefill", ["c"], node_in)
    step.set_ctx(StepContext(request_ids=("c",), graph_walk="prefill", slot=0, capture=False))
    step.steps.pop(SAMPLER)
    outcome = h.runner.admit(step)
    assert not outcome.ok and outcome.failed_resource == KDA_STATE
    assert "exhausted" in outcome.reason.message
    h.remove("a", "b", "c")


# --- (2) layer_types schedule honored ---------------------------------------


class _KVSpy:
    """Records which latent plane the MLA layers write and read through
    the KV resource — the v1 form of the lane's ``set_layer_idx`` spy."""

    def __init__(self, kv):
        self.writes: list[int] = []
        self.views: list[int] = []
        write, view = kv.write_latent, kv.layer_view

        def spy_write(latent, layer_idx=None, label=None):
            self.writes.append(layer_idx)
            return write(latent, layer_idx=layer_idx, label=label)

        def spy_view(layer_idx=None):
            self.views.append(layer_idx)
            return view(layer_idx)

        kv.write_latent = spy_write
        kv.layer_view = spy_view


@torch.no_grad()
def test_layer_schedule_honored():
    """idx % 4 == 3 -> MLA (compact planes), else KDA; dense iff idx < 3.

    Behavioral half: over one forward, the KV resource sees exactly the
    two full-attention planes, once each, in order — written then attended
    — and KDA layers never touch it (assembly spec section 6.1). The KDA
    layers' state lands in the slot pool instead: every KDA layer's plane
    of the request's slot is nonzero after the prefill.
    """
    h = _Harness(seed=15)
    cfg, model = h.cfg, h.lm

    assert cfg.layer_types == (
        LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION,
        LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION,
    )
    for idx, layer in enumerate(model.model.layers):
        if idx % 4 == 3:
            assert isinstance(layer.self_attn, Glm5NextMLAAttention)
            assert layer.kv_plane == cfg.full_attn_layer_indices.index(idx)
            assert layer.self_attn.kv_plane == layer.kv_plane
            assert layer.self_attn.indexer is not None  # every FULL owns one
            assert layer.self_attn.kv is h.kv and layer.self_attn.attn is h.attn
            assert layer._kda is None
        else:
            assert isinstance(layer.self_attn, Glm5NextKdaAttention)
            assert layer.kda_pos == cfg.kda_layer_indices.index(idx)
            assert layer._kda.resource is h.kda
        if idx < cfg.first_k_dense_replace:
            assert isinstance(layer.mlp, Glm5NextGatedMLP)
        else:
            assert isinstance(layer.mlp, Glm5NextSparseMoeBlock)
        # mHC on every trunk layer, both sites.
        assert layer.attn_hc.fn.shape == (
            (2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult * cfg.hidden_size)
        assert layer.ffn_hc.scale.shape == (3,)

    # The KV pool holds exactly the full-attention planes.
    assert h.kv.kv_cache.num_layers == len(cfg.full_attn_layer_indices) == 2

    spy = _KVSpy(h.kv)
    h.ingest("r0")
    h.prefill(["r0"], [torch.randint(0, cfg.vocab_size, (6,))])
    assert spy.writes == [0, 1]
    assert spy.views == [0, 1]
    slot = h.kda.slot_of("r0")
    recurrent, conv = h.kda.pool(RECURRENT), h.kda.pool(CONV)
    for kda_pos in range(len(cfg.kda_layer_indices)):
        assert recurrent[kda_pos, slot].abs().sum() > 0
        assert conv[kda_pos, slot].abs().sum() > 0
    # ... and only that slot: the sink and the free slots stay zero.
    others = [s for s in range(h.kda.max_slots + 1) if s != slot]
    assert recurrent[:, others].abs().sum() == 0
    h.remove("r0")

    # The real 45-layer schedule, config-level (never build full dims here).
    full = Glm5NextModelConfig()
    assert full.full_attn_layer_indices == tuple(range(3, 45, 4))
    assert len(full.kda_layer_indices) == 34
    assert [full.layer_types[i] for i in (0, 2, 3, 4, 43, 44)] == [
        LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION,
        LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION,
    ]


# --- (3) registry + Model construction --------------------------------------


def _specs_by_key(model: Glm5NextModel) -> dict:
    specs = model.get_node_resources()
    assert [type(s) for s in specs] == [KVSpec, AttentionSpec, SamplerSpec, SlotStateSpec]
    assert all(s.nodes == {"LLM"} for s in specs)
    return {s.resource_key: s for s in specs}


def test_glm5next_model_constructs_from_config():
    """The four node resources, full-size and reduced.

    Replaces the lane's ``get_kv_cache_config`` / ``get_node_engine_types``
    assertions: the KV cache is a ``KVSpec`` in the MLA layout, the backend
    an ``AttentionSpec`` (MLA, ckv = kv_lora_rank, scale = qk_head_dim
    ** -0.5), plus the sampler and the KDA slot pool — there is one engine
    now, so ``EngineType`` has no analogue.
    """
    model = object.__new__(Glm5NextModel)
    model.config = Glm5NextModelConfig()
    model.kda_max_requests = 32
    model.kda_conv_dtype = torch.bfloat16
    specs = _specs_by_key(model)

    kv = specs[KV_CACHE].config
    # 11 compact latent planes (full-attention layers only), one latent head.
    # head_dim = ckv(512) + kpe(64): NoPE's real rope is 0, but the cache pads
    # the kpe slot to 64 zeros so the capturable FlashInfer MLA kernel accepts
    # it (mla_ckv_dim stays the true 512). See config.mla_cache_kpe /
    # wiki/glm53-decode-capture.
    assert kv.layout == KVLayout.MLA
    assert kv.num_layers == 11
    assert kv.num_kv_heads == 1
    assert kv.head_dim == 576
    assert kv.num_qo_heads == 64
    assert kv.max_seq_len == model.config.index_topk == 2048
    assert kv.page_size == 128  # the FlashInfer MLA kernel's page size

    attn = specs[ATTN]
    assert attn.config.kv_cache == KV_CACHE and attn.depends_on() == {KV_CACHE}
    assert attn.config.backend == AttnBackend.MLA
    assert attn.config.mla_ckv_dim == 512
    assert attn.config.softmax_scale == pytest.approx(256 ** -0.5)

    sampler = specs[SAMPLER]
    assert sampler.vocab_size == 154880 and sampler.enable_repetion_penalty

    kda = specs[KDA_STATE].config
    assert kda.max_slots == 32
    rec, conv = kda.tensors[RECURRENT], kda.tensors[CONV]
    assert rec.shape == (34, 64, 128, 128) and rec.dtype == torch.float32
    assert conv.shape == (34, 3 * 64 * 128, 3) and conv.dtype == torch.bfloat16
    assert rec.slot_dim == conv.slot_dim == 1  # layer-major pools
    assert rec.pool_shape(32) == (34, 33, 64, 128, 128)  # + the sink slot
    specs[KDA_STATE].apply_yaml_overrides(max_slots=8)
    assert kda.max_slots == 8

    model.config.mtp_num_draft_tokens = 2
    assert _specs_by_key(model)[KV_CACHE].config.num_layers == 12  # + draft plane 11

    from mstar.graph.base import Loop

    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "decode"}
    assert isinstance(walks["decode"], Loop)

    # Real constructor path (no tokenizer IO at init — lazy like glm52).
    constructed = Glm5NextModel(
        "zai-org/GLM-5.3-Flash", config_variant="reduced", kda_max_requests=4)
    assert constructed.config.num_hidden_layers == 8
    reduced = _specs_by_key(constructed)
    kv_reduced = reduced[KV_CACHE].config
    # Same absorbed path as full-size (the lane's mla_absorb=False naive
    # fallback is gone): 2 planes, one latent head of 32 + 64.
    assert kv_reduced.layout == KVLayout.MLA
    assert kv_reduced.num_layers == 2
    assert kv_reduced.num_kv_heads == 1
    assert kv_reduced.head_dim == constructed.config.kv_lora_rank + 64 == 96
    assert kv_reduced.num_qo_heads == constructed.config.num_attention_heads == 4
    assert reduced[ATTN].config.mla_ckv_dim == 32
    assert reduced[KDA_STATE].config.max_slots == 4
    assert reduced[KDA_STATE].config.tensors[RECURRENT].shape == (6, 4, 16, 16)
    assert reduced[KDA_STATE].config.tensors[CONV].shape == (6, 3 * 4 * 16, 3)
    # Every spec builds into its resource on CPU (the harness relies on it).
    by_key = resolve_spec_dependencies(list(reduced.values()))
    for spec in reduced.values():
        info = EngineResourceInfo(
            device=torch.device("cpu"), kv_dtype=torch.float32,
            dependencies={k: by_key[k] for k in spec.depends_on()},
        )
        assert isinstance(build_resource(spec, info), spec.resource_class)

    # Per-request sampler config: model_kwargs over the config defaults.
    (req,) = constructed.get_request_resource_configs({}, {"temperature": 0.0, "top_k": 3}).values()
    assert isinstance(req, SamplingReqConfig)
    assert req.temperature == 0.0 and req.top_k == 3
    assert req.top_p == constructed.config.top_p

    with pytest.raises(NotImplementedError, match="dsa_long_context"):
        Glm5NextModel("x", config_variant="reduced", dsa_long_context=True)


def test_glm5next_registered():
    from mstar.model import registry

    # The registry is lazy now: a (module, class) pair resolved on demand.
    assert registry.MODEL_REGISTRY["glm5_next"] == (
        "mstar.model.glm5_next.glm5_next_model", "Glm5NextModel")
    assert registry.get_model_class("glm5_next") is Glm5NextModel
    assert registry.HF_MODELS["glm5_next"]["model_path_hf"] == "zai-org/GLM-5.3-Flash"


# --- (4) MTP off / structure present ----------------------------------------


@torch.no_grad()
def test_mtp_off_and_on_both_construct():
    model_off, _ = _build_model(seed=16, mtp_num_draft_tokens=0)
    assert model_off.mtp is None

    h = _Harness(seed=16, mtp_num_draft_tokens=2)
    model_on, cfg = h.lm, h.cfg
    mtp = model_on.mtp
    assert mtp is not None
    # Draft plane is the COMPACT index after the trunk's full-attn planes
    # (2 for the reduced config, 11 full-size) — never layer index 45.
    assert mtp.kv_plane == len(cfg.full_attn_layer_indices)
    assert h.kv.kv_cache.num_layers == mtp.kv_plane + 1  # the spec declared it
    # Plain-residual layer: the checkpoint has no hc tensors at layer 45.
    assert not hasattr(mtp.transformer_layer, "attn_hc")
    assert isinstance(mtp.transformer_layer.self_attn, Glm5NextMLAAttention)
    assert mtp.transformer_layer.self_attn.indexer is not None  # own FULL indexer
    assert mtp.transformer_layer.self_attn.kv is h.kv  # bound like the trunk

    # glm52 draft-loop call contract: (embeds, prev_hidden) -> (head_input
    # for the caller-owned lm_head, raw hidden to chain), run inside a
    # planned engine step like any layer.
    spy = _KVSpy(h.kv)
    h.ingest("r0")
    step, _ = h.open_step("prefill", ["r0"], [torch.tensor([5])])
    embeds = model_on.model.embed_tokens(torch.tensor([5]))
    prev_hidden = torch.randn(1, cfg.hidden_size)
    head_input, raw_hidden = mtp(embeds, prev_hidden)
    h.close_step(step)
    assert head_input.shape == raw_hidden.shape == (1, cfg.hidden_size)
    assert spy.writes == spy.views == [mtp.kv_plane]  # the layer set its own plane
    assert torch.equal(head_input, mtp.shared_head(raw_hidden))
    assert model_on.lm_head(head_input).shape == (1, cfg.vocab_size)
    h.remove("r0")


# --- (5) loader <-> model seam, executed -----------------------------------


def _raw_checkpoint_stream(model, cfg):
    """Invert the glm5_next weight map: (raw HF checkpoint name, tensor).

    The inverse of ``weight_loader``'s remap + stacked rules, applied to
    the assembled model's parameters — layer-flat mHC names, flat
    ForgetGate names, three per-branch conv tensors, per-expert
    gate/up/down, ``shared_experts``, the MTP module back under
    ``layers.<num_hidden_layers>``, and the ``model.language_model.``
    prefix. Feeding this stream back through the REAL pipeline must
    reproduce every parameter bit-for-bit.
    """
    import re

    n = cfg.num_hidden_layers
    hc_re = re.compile(r"\.(attn|ffn)_hc\.(fn|base|scale)$")

    for name, param in model.named_parameters():
        tensor = param.data
        if name == "lm_head.weight":
            yield name, tensor.clone()
            continue
        if name.startswith("mtp."):
            sub = name[len("mtp."):]
            if sub.startswith("transformer_layer."):
                sub = sub[len("transformer_layer."):]
            name = f"model.layers.{n}.{sub}"

        name = hc_re.sub(lambda m: f".hc_{m.group(1)}_{m.group(2)}", name)
        name = name.replace(".self_attn.forget_gate.", ".self_attn.")
        name = name.replace(".shared_expert.", ".shared_experts.")
        raw = "model.language_model." + name[len("model."):]

        if raw.endswith(".self_attn.conv1d.weight"):
            qkv = tensor.shape[0] // 3
            for branch, tag in enumerate(("q", "k", "v")):
                yield (
                    raw.replace(".conv1d.", f".{tag}_conv1d."),
                    tensor[branch * qkv : (branch + 1) * qkv].clone(),
                )
        elif raw.endswith(".mlp.experts.gate_up_proj"):
            inter = tensor.shape[1] // 2
            base = raw[: -len("gate_up_proj")]
            for e in range(tensor.shape[0]):
                yield f"{base}{e}.gate_proj.weight", tensor[e, :inter].clone()
                yield f"{base}{e}.up_proj.weight", tensor[e, inter:].clone()
        elif raw.endswith(".mlp.experts.down_proj"):
            base = raw[: -len("down_proj")]
            for e in range(tensor.shape[0]):
                yield f"{base}{e}.down_proj.weight", tensor[e].clone()
        elif raw.endswith(".gate_up_proj.weight"):
            inter = tensor.shape[0] // 2
            yield raw.replace(".gate_up_proj.", ".gate_proj."), tensor[:inter].clone()
            yield raw.replace(".gate_up_proj.", ".up_proj."), tensor[inter:].clone()
        else:
            yield raw, tensor.clone()


@torch.no_grad()
def test_loader_roundtrip_through_real_pipeline():
    """Raw HF-named stream -> load_weights -> every parameter lands.

    Executes the actual remapper, stacked rules, and every attached
    weight_loader (fused conv placement, expert stacking, merged gate_up)
    against the assembled module tree — the closest thing to a checkpoint
    load that runs without 306 GB. Vision keys must be skipped, never
    loaded; with drafting off, the layer-45 stream must be skipped too.
    """
    model_a, cfg = _build_model(seed=20, mtp_num_draft_tokens=2)
    stream = list(_raw_checkpoint_stream(model_a, cfg))
    stream.append(("model.visual.patch_embed.proj.weight", torch.zeros(4, 4)))

    model_b, _ = _build_model(seed=21, mtp_num_draft_tokens=2)
    loaded = model_b.load_weights(iter(stream))
    all_params = {name for name, _ in model_b.named_parameters()}
    assert loaded == all_params  # everything hit, nothing stray

    params_a = dict(model_a.named_parameters())
    for name, param_b in model_b.named_parameters():
        assert torch.equal(param_b, params_a[name]), name

    # Conv fusion order is q|k|v by the module's cat order — pin the
    # loader's placement against the raw stream, not just the roundtrip.
    kda = model_b.model.layers[0].self_attn
    qkv = kda.qkv_dim
    raw = dict(stream)
    assert torch.equal(
        kda.conv1d.weight[:qkv],
        raw["model.language_model.layers.0.self_attn.q_conv1d.weight"])
    assert torch.equal(
        kda.conv1d.weight[2 * qkv :],
        raw["model.language_model.layers.0.self_attn.v_conv1d.weight"])
    # Flat checkpoint ForgetGate names landed on the nested module.
    assert torch.equal(
        model_b.model.layers[0].self_attn.forget_gate.dt_bias,
        raw["model.language_model.layers.0.self_attn.dt_bias"])

    # Drafting off: the same stream loads the trunk and drops layer 45.
    model_c, _ = _build_model(seed=22, mtp_num_draft_tokens=0)
    loaded_c = model_c.load_weights(iter(stream))
    assert loaded_c == {name for name, _ in model_c.named_parameters()}
    assert not any(name.startswith("mtp.") for name in loaded_c)


def test_full_geometry_names_resolve_through_pipeline():
    """Representative full-config (45-layer) raw names resolve to targets
    the loader expects — pins the forget_gate remap fix at the geometry
    the real index has (the reduced roundtrip can't see layer 44)."""
    from mstar.model.glm5_next.weight_loader import (
        expected_parameter_paths,
        resolve_index_names,
    )

    cfg = Glm5NextModelConfig()  # config only — never build full dims
    names = [
        "model.language_model.layers.44.self_attn.dt_bias",  # KDA, flat
        "model.language_model.layers.44.self_attn.A_log",
        "model.language_model.layers.44.self_attn.f_a_proj.weight",
        "model.language_model.layers.44.self_attn.q_conv1d.weight",
        "model.language_model.layers.44.hc_attn_fn",
        "model.language_model.layers.43.self_attn.q_a_proj.weight",  # MLA
        "model.language_model.layers.43.self_attn.indexer.index_kpool_compress_ape",
        "model.language_model.layers.43.mlp.experts.287.up_proj.weight",
        "model.language_model.layers.43.mlp.shared_experts.gate_proj.weight",
        "model.language_model.layers.45.enorm.weight",  # MTP glue
        "model.language_model.layers.45.self_attn.o_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",  # vision -> skip
    ]
    resolved = resolve_index_names(names, cfg, load_mtp=True)
    assert resolved["unmapped"] == ()
    assert resolved["skip_vision"] == ("model.visual.blocks.0.attn.qkv.weight",)
    expected = expected_parameter_paths(cfg, load_mtp=True)
    targets = {entry.rsplit(" -> ", 1)[1] for entry in resolved["loaded"]}
    assert targets <= set(expected), targets - set(expected)
    assert "model.layers.44.self_attn.forget_gate.dt_bias" in targets
    assert "mtp.transformer_layer.self_attn.o_proj.weight" in targets


# --- (6) the one MLP-math delta: the SwiGLU clamp ---------------------------


@torch.no_grad()
def test_swiglu_clamp_engages_on_every_mlp_path():
    """gate.clamp(max=L), up.clamp(-L, L), THEN silu — dense/shared MLP and
    routed experts alike. Verified against an inline reference at a limit
    small enough that unclamped math visibly diverges."""
    from mstar.model.glm5_next.components.moe import Glm5NextMoEGate

    torch.manual_seed(31)
    cfg = Glm5NextModelConfig.reduced()
    cfg.swiglu_limit = 1.0  # engage hard at test scales
    limit = cfg.swiglu_limit

    def clamped_swiglu(x, gate_up_w, down_w):
        gate, up = (x @ gate_up_w.T).chunk(2, dim=-1)
        clamped = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(
            -limit, limit)
        return clamped @ down_w.T

    x = torch.randn(5, cfg.hidden_size) * 3.0  # drives |pre-acts| >> 1

    mlp = Glm5NextGatedMLP(
        hidden_size=cfg.hidden_size, intermediate_size=32,
        swiglu_limit=limit)
    mlp.gate_up_proj.weight.data.normal_(0, 0.5)
    mlp.down_proj.weight.data.normal_(0, 0.5)
    want = clamped_swiglu(x, mlp.gate_up_proj.weight, mlp.down_proj.weight)
    assert torch.equal(mlp(x), want)
    unclamped = (
        torch.nn.functional.silu(
            (x @ mlp.gate_up_proj.weight.T).chunk(2, -1)[0])
        * (x @ mlp.gate_up_proj.weight.T).chunk(2, -1)[1]
    ) @ mlp.down_proj.weight.T
    assert not torch.allclose(want, unclamped, atol=1e-2)  # clamp is live

    block = Glm5NextSparseMoeBlock(cfg)
    assert isinstance(block.gate, Glm5NextMoEGate)
    block.gate.weight.data.normal_(0, 0.3)
    block.experts.gate_up_proj.data.normal_(0, 0.5)
    block.experts.down_proj.data.normal_(0, 0.5)
    block.shared_expert.gate_up_proj.weight.data.normal_(0, 0.5)
    block.shared_expert.down_proj.weight.data.normal_(0, 0.5)

    topk_weights, topk_ids = block.gate(x)
    topk_weights = topk_weights.to(x.dtype)
    want = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            want[t] += topk_weights[t, k] * clamped_swiglu(
                x[t : t + 1],
                block.experts.gate_up_proj[e], block.experts.down_proj[e],
            )[0]
    want = want + clamped_swiglu(
        x, block.shared_expert.gate_up_proj.weight,
        block.shared_expert.down_proj.weight)
    assert torch.allclose(block(x), want, atol=1e-5, rtol=0)


def test_moe_quant_kernel_resolution_and_clamp_guard():
    """glm52 kimi quant_kernel semantics + the new SwiGLU-clamp-capability
    guard. ``process_weights_after_loading`` resolves ``_use_fused``:
    ``"reference"`` (the default) and ``"auto"`` without CUDA keep the clamped
    reference loop (False); explicit ``"triton"`` on CPU must NOT silently
    serve the unclamped fused kernel, so it raises -- and the message names
    ``swiglu_limit``, the clamp guard that replaced the old blanket refusal.
    A non-fp8 (bf16) block never fuses whatever the knob says."""
    cfg = Glm5NextModelConfig.reduced_fp8()
    assert cfg.moe_quant_kernel == "reference"
    blk = Glm5NextSparseMoeBlock(cfg)
    blk.process_weights_after_loading("cpu")
    assert blk._use_fused is False

    cfg_auto = Glm5NextModelConfig.reduced_fp8()
    cfg_auto.moe_quant_kernel = "auto"
    blk_auto = Glm5NextSparseMoeBlock(cfg_auto)
    blk_auto.process_weights_after_loading("cpu")
    assert blk_auto._use_fused is False  # no CUDA here -> reference

    cfg_triton = Glm5NextModelConfig.reduced_fp8()
    cfg_triton.moe_quant_kernel = "triton"
    blk_triton = Glm5NextSparseMoeBlock(cfg_triton)
    with pytest.raises(RuntimeError, match="swiglu_limit"):
        blk_triton.process_weights_after_loading("cpu")

    cfg_bf16 = Glm5NextModelConfig.reduced()  # no quantization_config
    cfg_bf16.moe_quant_kernel = "auto"
    blk_bf16 = Glm5NextSparseMoeBlock(cfg_bf16)
    blk_bf16.process_weights_after_loading("cpu")
    assert blk_bf16.fp8_experts is False and blk_bf16._use_fused is False


def test_module_tree_matches_loader_expectations():
    """The assembled tree == the loader's expected target set, name for
    name — the loader <-> model integration seam, pinned without a
    checkpoint. A drift on either side (a renamed module, a remap change)
    fails here first instead of at the 306 GB load."""
    from mstar.model.glm5_next.weight_loader import expected_parameter_paths

    for k, load_mtp in ((0, False), (2, True)):
        model, cfg = _build_model(seed=17, mtp_num_draft_tokens=k)
        expected = expected_parameter_paths(cfg, load_mtp=load_mtp)
        actual = {name for name, _ in model.named_parameters()}
        assert actual == set(expected), (
            f"k={k}: only in model: {sorted(actual - set(expected))[:5]} | "
            f"only in loader expectations: "
            f"{sorted(set(expected) - actual)[:5]}"
        )


# --- (7) the submodule's engine path, end to end ----------------------------


@torch.no_grad()
def test_submodule_engine_path_prefill_then_decode():
    """``test_glm5next_engine_path.py`` folded in: the submodule's own seam
    (prepare_inputs -> declare_step -> preprocess -> forward_batched) with
    a greedy stand-in for the Triton sampler. Two requests interleave; the
    third token of a stepwise trajectory equals the token a fresh request
    gets from prefilling that trajectory whole, and the context guard
    (``index_topk``) refuses the request that would cross it."""
    from types import SimpleNamespace

    h = _Harness(seed=0)
    sub, resources = h.sub, h.resources
    resources[SAMPLER] = _GreedySampler()  # forward_batched samples in-forward

    def step(walk, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rids = list(inputs)
        node_in = [
            sub.prepare_inputs(walk, SimpleNamespace(request_id=rid), {"text_inputs": [ids]})
            for rid, ids in inputs.items()
        ]
        st = sub.declare_step(walk, rids, node_in)
        st.set_ctx(StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False))
        st.steps.pop(SAMPLER)
        assert h.runner.admit(st).ok
        h.runner.plan(st)
        ei = h.engine_inputs(rids)
        out = sub.forward_batched(walk, ei, **sub.preprocess(walk, ei, node_in))
        h.runner.commit(st)
        return {rid: out[rid]["new_token"][0] for rid in rids}

    prompts = {
        "r0": torch.tensor([5, 9, 13, 7, 21], dtype=torch.long),
        "r1": torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=torch.long),
    }
    h.ingest("r0", "r1")
    t1 = step("prefill", prompts)
    t2 = step("decode", t1)
    t3 = step("decode", t2)
    for rid, prompt in prompts.items():
        assert h.kda.committed(rid) == prompt.shape[0] + 2
        assert h.stored_len(rid) == prompt.shape[0] + 2

    # Reference: the whole trajectory prefilled on a fresh request must yield
    # the same third token (chunk-vs-step parity of the engine path).
    for rid, prompt in prompts.items():
        ref = f"ref-{rid}"
        h.ingest(ref)
        (t3_ref,) = step("prefill", {ref: torch.cat([prompt, t1[rid], t2[rid]])}).values()
        assert t3_ref.item() == t3[rid].item(), (rid, t3_ref.item(), t3[rid].item())
        h.remove(ref)

    # The serving regime: a request may not cross index_topk (reduced: 64).
    h.ingest("long")
    too_long = torch.zeros(h.cfg.index_topk + 1, dtype=torch.long)
    with pytest.raises(RuntimeError, match="index_topk"):
        sub.prepare_inputs("prefill", SimpleNamespace(request_id="long"), {"text_inputs": [too_long]})
    assert sub.max_batch_size("decode") == h.kda.max_slots

    h.remove("r0", "r1", "long")
    assert h.kda.num_free == h.kda.max_slots
    assert h.kda.tracked_requests() == set()
