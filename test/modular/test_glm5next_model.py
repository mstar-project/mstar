"""GLM-5.3-Flash assembled-model tests on the resource-pool engine — CPU,
reduced config, no GPU deps.
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
    StepRunner,
    resolve_spec_dependencies,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource  # noqa: E402
from mstar.engine.resources.kv import manager as kv_mod  # noqa: E402
from mstar.engine.resources.linear_attn.config import (  # noqa: E402
    LinearAttnBackend,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent.config import RecurrentStateSpec  # noqa: E402
from mstar.engine.resources.recurrent.pool import RecurrentStatePool  # noqa: E402
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
    KDA,
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
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine  # noqa: E402

# Chunked prefill and recurrent decode are mathematically equal but numerically
# distinct (different reduction structure). In fp32 the gap between them is
# BLAS-backend-dependent, and a tolerance wide enough to cover any backend
# would swallow real bugs — gate wiring, state desync or op order move the
# logits far more — so cross-path parity runs in float64 (LOGITS_ATOL_F64).
# LOGITS_ATOL is for the same-path bitwise/greedy checks, which are stable.
#
# The MLA backend's SDPA fallback computes in fp32 whatever the model dtype, so
# the float64 run still isolates the KDA chunk-vs-recurrent structure.
LOGITS_ATOL = 1e-3
LOGITS_ATOL_F64 = 1e-5
_WEIGHT_STD = 0.03

# Small pages so a reduced prefill spans several: the paged latent gather of
# the MLA fallback and the page-boundary bookkeeping of the KV resource are
# on the path the parity tests measure. (The serve YAML tunes these under
# ``resources: kv_cache:`` — this is the same override hook.)
_PAGE_SIZE = 8
_MAX_PAGES = 32


def _slot_of(pool: RecurrentStatePool, rid: str) -> int | None:
    slot = pool._slots.get(rid, {}).get(LABEL)
    return None if slot is None else slot.index


def _holders(pool: RecurrentStatePool) -> set[str]:
    return {rid for rid, labels in pool._slots.items() if labels}


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


class _StubTransfer:
    """No transfer engine in a unit test: the KV resource's transfer manager is
    replaced per test (``KVManager.__init__`` builds one unconditionally)."""

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
def _cpu_flashinfer(monkeypatch):
    """CPU ``flashinfer.norm.rmsnorm`` so RMSNorm-backed forwards run here."""
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
    process_weights_after_loading(model, torch.device("cpu"))
    model.eval()
    model.requires_grad_(False)


def _build_model(
    seed: int, mtp_num_draft_tokens: int = 0, dtype: torch.dtype = torch.float32
) -> tuple[Glm5NextForCausalLM, Glm5NextModelConfig]:
    """Reduced-config model with small random weights, no engine attached."""
    torch.manual_seed(seed)
    cfg = Glm5NextModelConfig.reduced()
    cfg.mtp_num_draft_tokens = mtp_num_draft_tokens
    model = Glm5NextForCausalLM(cfg)
    _randomize(model, dtype)
    return model, cfg


class _GreedySampler:
    """Argmax in place of the Triton sampler for the engine-path smoke test
    (``forward_batched`` samples inside the forward)."""

    def sample(self, request_ids, logits, **kwargs):
        return logits.argmax(-1)


class _Harness:
    """A reduced ``Glm5NextForCausalLM`` behind the REAL engine resources."""

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
        self.pool = self.resources[KDA_STATE]
        self.kda = self.resources[KDA]

    @property
    def usable(self) -> int:
        return self.pool.config.usable_slots

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
        """Rewind a request's KV stream by ``n`` tokens. The KV resource has
        no rewind entry point of its own, and on a paged cache a rewind is
        only the stream's ``stored_len`` — its pages stay leased as a
        high-water mark and heal by overwrite.
        """
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
        step.steps.pop(SAMPLER)  # these tests read logits, never sampled tokens
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
    """Chunked prefill and recurrent decode agree over the same tokens."""
    # float64: isolates the chunk-vs-recurrent algorithmic equivalence from
    # platform fp32 rounding, which the reduced config amplifies layer by layer.
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
        assert h.stored_len(ref_id) == ids.shape[0]
        h.remove(ref_id)
    assert _holders(h.pool) == set()  # no slot leaks
    assert h.pool.num_free_slots == h.usable

    # Subject: joint prefill, then joint stepwise decode.
    request_ids = list(prefill_lens)
    h.ingest(*request_ids)
    prefill_logits = h.prefill(
        request_ids, [tokens[rid][: prefill_lens[rid]] for rid in request_ids])
    slots = {rid: _slot_of(h.pool, rid) for rid in request_ids}
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
            # Same greedy trajectory, not merely close logits.
            assert int(logits[i].argmax()) == int(want.argmax())

    for rid in request_ids:
        assert h.stored_len(rid) == prefill_lens[rid] + num_decode
    h.remove(*request_ids)
    assert h.pool.num_free_slots == h.usable


@torch.no_grad()
def test_decode_replays_bitwise_across_kda_snapshot_restore():
    """snapshot -> decode k -> restore + KV rewind -> decode k again: the
    replay is BIT-exact (same states, same ops), the strongest save/restore
    check there is. The KV plane rewinds for free, the recurrent state does
    not — hence the snapshot, taken from the request's slot in the pool.
    """
    h = _Harness(seed=12)
    cfg = h.cfg
    prefill_len, num_decode = 9, 3
    torch.manual_seed(13)
    ids = torch.randint(0, cfg.vocab_size, (prefill_len + num_decode,))

    h.ingest("r0")
    h.prefill(["r0"], [ids[:prefill_len]])

    slot = _slot_of(h.pool, "r0")
    snap = {name: block[:, slot].clone() for name, block in h.pool._blocks.items()}
    assert snap["state"].shape == (
        len(cfg.kda_layer_indices), cfg.linear_num_heads, cfg.linear_head_dim, cfg.linear_head_dim)

    def run_decode():
        out = []
        for step in range(num_decode):
            pos = prefill_len + step
            out.append(h.decode(["r0"], ids[pos : pos + 1]))
        return torch.cat(out)

    first = run_decode()
    assert h.stored_len("r0") == prefill_len + num_decode

    for name, saved in snap.items():
        h.pool._blocks[name][:, slot].copy_(saved)
    h.rewind_kv("r0", num_decode)  # paged-KV rewind heals by overwrite
    assert h.stored_len("r0") == prefill_len
    replay = run_decode()

    assert torch.equal(first, replay)
    h.remove("r0")


def test_kda_pool_lifecycle_is_explicit_and_loud():
    """The pool's slot lifecycle, driven on the model's own declared pool."""
    h = _Harness(seed=14, max_slots=2)
    pool = h.pool
    ids = torch.zeros(4, dtype=torch.long)
    h.ingest("r0")
    assert _slot_of(pool, "r0") is None  # ingest registers, admit leases

    step, _ = h.open_step("prefill", ["r0"], [ids])
    slot = _slot_of(pool, "r0")
    assert slot is not None and slot != 0  # not the sink
    # a fresh slot reads as zeros until the step that wrote it commits
    assert h.kda.current_plan().has_state.tolist() == [False]
    h.close_step(step)
    assert pool._slots["r0"][LABEL].has_state

    # Chunked continue: the same slot is kept (idempotent lease) and resumed.
    step, _ = h.open_step("prefill", ["r0"], [ids[:2]])
    assert _slot_of(pool, "r0") == slot
    assert h.kda.current_plan().has_state.tolist() == [True]
    h.close_step(step)

    # A planned-but-dropped step (its forward never ran) moves nothing.
    h.open_step("decode", ["r0"], [ids[:1]])
    assert _slot_of(pool, "r0") == slot and pool.num_free_slots == 1

    # reset zeroes the slot and keeps it; a freeing reset hands it back
    pool.block("state", 0)[slot].fill_(1.0)
    pool.reset_request("r0")
    assert _slot_of(pool, "r0") == slot and not pool._slots["r0"][LABEL].has_state
    assert pool._blocks["state"][:, slot].abs().sum() == 0
    pool.reset_request("r0", free=True)
    assert _slot_of(pool, "r0") is None and pool.num_free_slots == 2
    h.remove("r0")
    h.remove("r0")  # idempotent, engine may double-retire
    assert pool.num_free_slots == 2 and _holders(pool) == set()

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
    assert "pool is full" in outcome.reason.message
    h.remove("a", "b", "c")


# --- (2) layer_types schedule honored ---------------------------------------


class _KVSpy:
    """Records which latent plane the MLA layers write and read through the
    KV resource."""

    def __init__(self, kv):
        self.writes: list[int] = []
        self.views: list[int] = []
        write, view = kv.write_kv, kv.layer_view

        def spy_write(latent, v=None, layer_idx=None, label=None):
            self.writes.append(layer_idx)
            return write(latent, v, layer_idx=layer_idx, label=label)

        def spy_view(layer_idx=None):
            self.views.append(layer_idx)
            return view(layer_idx)

        kv.write_kv = spy_write
        kv.layer_view = spy_view


@torch.no_grad()
def test_layer_schedule_honored():
    """idx % 4 == 3 -> MLA (compact planes), else KDA; dense iff idx < 3."""
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
        else:
            assert isinstance(layer.self_attn, Glm5NextKdaAttention)
            assert layer.kda_pos == cfg.kda_layer_indices.index(idx)
            assert layer.self_attn.pool is h.pool and layer.self_attn.kda is h.kda
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
    slot = _slot_of(h.pool, "r0")
    recurrent, conv = h.pool._blocks["state"], h.pool._blocks["conv"]
    for kda_pos in range(len(cfg.kda_layer_indices)):
        assert recurrent[kda_pos, slot].abs().sum() > 0
        assert conv[kda_pos, slot].abs().sum() > 0
    # ... and only that slot: the sink and the free slots stay zero.
    others = [s for s in range(h.pool.config.max_slots) if s != slot]
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


def test_kda_layers_hold_no_per_slot_state():
    """The pool is the only per-slot storage: the KDA layers bind the one pool
    and the one KDA resource, and no layer keeps a staging plane of its own."""
    h = _Harness(seed=15, max_slots=6)  # 7 slots: a size no model dim shares
    slots = h.pool.config.max_slots
    kda_layers = [layer for layer in h.lm.model.layers if layer.is_linear_attention]
    assert len(kda_layers) > 1
    assert len({id(layer.self_attn.pool) for layer in kda_layers}) == 1
    assert len({id(layer.self_attn.kda) for layer in kda_layers}) == 1
    for module in h.lm.modules():
        tensors = [*module.parameters(recurse=False), *module.buffers(recurse=False),
                   *(v for v in vars(module).values() if isinstance(v, torch.Tensor))]
        assert not any(t.dim() >= 2 and t.shape[0] == slots for t in tensors), module


# --- (3) registry + Model construction --------------------------------------


def _specs_by_key(model: Glm5NextModel) -> dict:
    specs = model.get_node_resources()
    assert [type(s) for s in specs] == [
        KVSpec, AttentionSpec, SamplerSpec, RecurrentStateSpec, LinearAttnSpec]
    assert all(s.nodes == {"LLM"} for s in specs)
    return {s.resource_key: s for s in specs}


@pytest.mark.parametrize("name", ["glm5_next_tp8.yaml", "glm5_next_tp8_mtp.yaml"])
def test_serve_yaml_pages_cover_the_slot_pool(name):
    """Every slot can hold a full-length request without the KV pool refusing it (under MTP
    a request's rejected drafts take cache rows too)."""
    import yaml

    serve = yaml.safe_load((Path(__file__).parents[2] / "configs" / name).read_text())
    kv, kda = serve["resources"]["kv_cache"], serve["resources"]["kda_state"]
    cfg = Glm5NextModelConfig(max_seq_len=serve["max_seq_len"])
    cfg.mtp_num_draft_tokens = serve["model_kwargs"]["mtp_num_draft_tokens"]
    # both pools hold a sink out of circulation
    requests = kda["max_slots"] - 1
    assert requests == 64
    assert (kv["max_num_pages"] - 1) * kv["page_size"] >= requests * cfg.kv_rows


@pytest.mark.parametrize("name", ["glm5_next_tp8.yaml", "glm5_next_tp8_mtp.yaml"])
def test_serve_yaml_caps_prefill_steps(name):
    import yaml

    serve = yaml.safe_load((Path(__file__).parents[2] / "configs" / name).read_text())
    assert serve["model_kwargs"]["prefill_max_step_tokens"] == "auto"


def test_glm5next_model_constructs_from_config():
    """The five node resources, full-size and reduced."""
    model = object.__new__(Glm5NextModel)
    model.config = Glm5NextModelConfig()
    model.kda_max_requests = 32
    model.kda_conv_dtype = torch.bfloat16
    specs = _specs_by_key(model)

    kv = specs[KV_CACHE].config
    # 11 compact latent planes (full-attention layers only), one latent head.
    assert kv.layout == KVLayout.MLA
    assert kv.num_layers == 11
    assert kv.num_kv_heads == 1
    # NoPE: no rope slot in the cache unless MTP needs it (checked below)
    assert (kv.kv_lora_rank, kv.qk_rope_head_dim, kv.head_dim) == (512, 0, 512)
    assert kv.num_qo_heads == 64
    assert kv.max_seq_len == model.config.index_topk == 2048
    assert kv.page_size == 128  # the FlashInfer MLA kernel's page size

    attn = specs[ATTN]
    assert attn.config.kv_cache == KV_CACHE and attn.depends_on() == {KV_CACHE}
    assert attn.config.backend == AttnBackend.FLASHINFER_MLA
    assert attn.config.sm_scale == pytest.approx(256 ** -0.5)

    sampler = specs[SAMPLER]
    assert sampler.vocab_size == 154880 and sampler.enable_repetion_penalty

    pool = specs[KDA_STATE].config
    assert pool.num_layers == 34
    assert pool.max_slots == 33 and pool.usable_slots == 32  # + the sink slot
    rec, conv = pool.blocks["state"], pool.blocks["conv"]
    assert rec.shape == (64, 128, 128) and rec.dtype == torch.float32
    assert conv.shape == (3 * 64 * 128, 3) and conv.dtype == torch.bfloat16
    assert rec.shard_dims == conv.shard_dims == (0,)  # heads and channels shard
    specs[KDA_STATE].apply_yaml_overrides(max_slots=8)
    assert pool.max_slots == 8
    kda = specs[KDA]
    assert kda.depends_on() == {KDA_STATE}
    assert kda.config.variant is LinearAttnVariant.KDA
    assert kda.config.backend is LinearAttnBackend.TRITON
    assert kda.config.gate_lower_bound == -5.0

    model.config.mtp_num_draft_tokens = 2
    kv_mtp = _specs_by_key(model)[KV_CACHE].config
    assert kv_mtp.num_layers == 12  # + draft plane 11
    assert kv_mtp.head_dim == 576  # the rope slot masks rejected draft rows

    from mstar.graph.base import Loop

    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "decode"}
    assert isinstance(walks["decode"], Loop)

    # Real constructor path: no tokenizer IO at init, the tokenizer is lazy.
    constructed = Glm5NextModel(
        "zai-org/GLM-5.3-Flash", config_variant="reduced", kda_max_requests=4)
    assert constructed.config.num_hidden_layers == 8
    reduced = _specs_by_key(constructed)
    kv_reduced = reduced[KV_CACHE].config
    # Same absorbed path as full-size — there is no non-absorbed fallback:
    # 2 planes, one latent head of 32.
    assert kv_reduced.layout == KVLayout.MLA
    assert kv_reduced.num_layers == 2
    assert kv_reduced.num_kv_heads == 1
    assert kv_reduced.head_dim == kv_reduced.kv_lora_rank == 32
    assert kv_reduced.num_qo_heads == constructed.config.num_attention_heads == 4
    pool = reduced[KDA_STATE].config
    assert pool.max_slots == 5 and pool.num_layers == 6
    assert pool.blocks["state"].shape == (4, 16, 16)
    assert pool.blocks["conv"].shape == (3 * 4 * 16, 3)
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

    # The registry is lazy: a (module, class) pair resolved on demand.
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

    # Draft-loop call contract: (embeds, prev_hidden) -> the shared-head norm
    # output, which is both the caller-owned lm_head's input and the next
    # draft's prev_hidden; run inside a planned engine step like any other layer.
    spy = _KVSpy(h.kv)
    h.ingest("r0")
    step, _ = h.open_step("prefill", ["r0"], [torch.tensor([5])])
    embeds = model_on.model.embed_tokens(torch.tensor([5]))
    prev_hidden = torch.randn(1, cfg.hidden_size)
    head_input = mtp(embeds, prev_hidden)
    h.close_step(step)
    assert head_input.shape == (1, cfg.hidden_size)
    assert spy.writes == spy.views == [mtp.kv_plane]  # the layer set its own plane
    assert model_on.lm_head(head_input).shape == (1, cfg.vocab_size)
    h.remove("r0")


@torch.no_grad()
def test_mtp_block_row_query_matches_the_causal_block():
    """A draft pass queries one row of the step's block (the other rows attend with zero
    queries and rewrite their latents): its output is that row of the MTP layer run over the
    whole block."""
    h = _Harness(seed=18, mtp_num_draft_tokens=3)
    layer = h.lm.mtp.transformer_layer
    attn = layer.self_attn
    hidden = h.cfg.hidden_size
    h.ingest("r0")
    step, _ = h.open_step("prefill", ["r0"], [torch.arange(2, 7)])
    layer(torch.randn(5, hidden))  # the MTP plane's context
    h.close_step(step)
    block = 4
    x = torch.randn(block, hidden)
    step, _ = h.open_step("prefill", ["r0"], [torch.zeros(block, dtype=torch.long)])
    whole = layer(x)
    latents = x.new_zeros(1, block, attn.kv_lora_rank + attn.mla_cache_kpe)
    for row in range(block):
        got = layer(x[row:row + 1], row=row, latents=latents)
        torch.testing.assert_close(got[0], whole[row], rtol=1e-4, atol=1e-5)
    h.close_step(step)
    h.remove("r0")


# --- (5) loader <-> model seam, executed -----------------------------------


def _raw_checkpoint_stream(model, cfg):
    """Invert the glm5_next weight map: (raw HF checkpoint name, tensor)."""
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
    """Raw HF-named stream -> load_weights -> every parameter lands."""
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


@torch.no_grad()
def test_checkpoint_load_refuses_a_missing_trunk_weight(tmp_path):
    """The production load path, both branches (sharded index and a single file):
    a snapshot without one trunk tensor is refused, not served from garbage."""
    import json

    from safetensors.torch import save_file

    source, cfg = _build_model(seed=23)
    stream = dict(_raw_checkpoint_stream(source, cfg))
    dropped = "model.language_model.layers.1.self_attn.b_proj.weight"
    assert dropped in stream
    model = Glm5NextModel("x", config_variant="reduced")

    def write(root: Path, tensors: dict, sharded: bool) -> None:
        root.mkdir()
        name = "model-00001-of-00001.safetensors" if sharded else "model.safetensors"
        save_file(tensors, str(root / name))
        if sharded:
            index = {"weight_map": {key: name for key in tensors}}
            (root / "model.safetensors.index.json").write_text(json.dumps(index))

    for sharded in (True, False):
        full = tmp_path / f"full-{sharded}"
        write(full, stream, sharded)
        model._load_checkpoint(_build_model(seed=24)[0], str(full), "cpu", None)

        partial = tmp_path / f"partial-{sharded}"
        write(partial, {k: v for k, v in stream.items() if k != dropped}, sharded)
        with pytest.raises(RuntimeError, match="layers.1.self_attn.b_proj.weight"):
            model._load_checkpoint(_build_model(seed=24)[0], str(partial), "cpu", None)


def _safe_open_shards(repo_dir, device="cpu", prefix=None, keys=None, slice_spec=None):
    """The mmap reader the loader iterators replaced: safe_open + get_slice."""
    import json

    from safetensors import safe_open

    weight_map = json.loads((Path(repo_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(Path(repo_dir) / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if keys is not None and key not in keys:
                    continue
                spec = slice_spec(key) if slice_spec is not None else None
                if spec is None:
                    yield key, f.get_tensor(key)
                    continue
                dim, start, stop = spec
                sl = f.get_slice(key)
                index = [slice(None)] * len(sl.get_shape())
                index[dim] = slice(start, stop)
                yield key, sl[tuple(index)]


class _LinearScanMatcher:
    """The first-win rule scan ``load_weights_into`` used before indexing."""

    def __init__(self, rules):
        self.rules = rules

    def __call__(self, name):
        from mstar.model.loader.base import _apply_stacked

        return _apply_stacked(name, self.rules)


@torch.no_grad()
def test_fp8_checkpoint_loads_bit_identical_to_the_mmap_reader(tmp_path, monkeypatch):
    """A sharded fp8 block-scaled checkpoint through the production TP read
    path (sliced reads, dequant, fp8-resident experts), both TP=2 ranks:
    every parameter matches, bit for bit, what the safe_open reader and
    the linear rule scan load."""
    import json

    from safetensors.torch import save_file

    import mstar.model.loader.base as loader_base
    import mstar.model.loader.iterators as loader_iters
    from mstar.distributed.communication import CommGroup
    from mstar.model.glm5_next.quantization import FP8_DTYPE

    source, cfg = _build_model(seed=25)
    torch.manual_seed(25)
    stream = {}
    for key, tensor in _raw_checkpoint_stream(source, cfg):
        # Routed and shared experts ship fp8 with 16x16 block scales here
        # (reduced_fp8); every other tensor stays unquantized.
        if tensor.ndim == 2 and (".experts." in key or ".shared_experts." in key):
            rows, cols = tensor.shape
            stream[key] = (tensor * 64).to(FP8_DTYPE)
            stream[key + "_scale_inv"] = torch.rand(-(-rows // 16), -(-cols // 16)) + 0.5
        else:
            stream[key] = tensor
    names = sorted(stream)
    weight_map = {}
    for i in range(3):
        shard = f"model-{i + 1:05d}-of-00003.safetensors"
        save_file({k: stream[k] for k in names[i::3]}, str(tmp_path / shard))
        weight_map.update({k: shard for k in names[i::3]})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    model = Glm5NextModel("x", config_variant="reduced_fp8")

    def load(rank):
        group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
        with torch.device("meta"):
            lm = Glm5NextForCausalLM(model.config, comm_group=group)
        lm = lm.to(torch.bfloat16)
        lm.to_empty(device="cpu")
        model._load_checkpoint(lm, str(tmp_path), "cpu", group)
        return dict(lm.named_parameters())

    for rank in range(2):
        new = load(rank)
        with monkeypatch.context() as m:
            m.setattr(loader_iters, "iter_safetensors_shards", _safe_open_shards)
            m.setattr(loader_base, "_StackedMatcher", _LinearScanMatcher)
            old = load(rank)
        assert new.keys() == old.keys()
        assert any(p.dtype == torch.uint8 for p in new.values())  # fp8-resident experts
        for name, param in new.items():
            assert param.dtype == old[name].dtype, name
            assert torch.equal(param.view(torch.uint8), old[name].view(torch.uint8)), name


def test_full_geometry_names_resolve_through_pipeline():
    """Representative full-config (45-layer) raw names resolve to targets
    the loader expects — pins the forget_gate remap at the geometry the
    real index has (the reduced roundtrip can't see layer 44)."""
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


def test_moe_quant_kernel_resolution_and_clamp_guard(monkeypatch):
    """'reference' and 'auto' resolve off the device; 'triton' refuses without one."""
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
    with pytest.raises(RuntimeError, match="needs CUDA"):
        blk_triton.process_weights_after_loading("cpu")

    # before sm89 (A100) there are no fp8 dots: 'auto' is the reference, 'triton' refuses
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    blk_a100 = Glm5NextSparseMoeBlock(cfg_auto)
    blk_a100.process_weights_after_loading("cuda")
    assert blk_a100._use_fused is False
    with pytest.raises(RuntimeError, match="sm89"):
        Glm5NextSparseMoeBlock(cfg_triton).process_weights_after_loading("cuda")

    cfg_bf16 = Glm5NextModelConfig.reduced()  # no quantization_config
    cfg_bf16.moe_quant_kernel = "auto"
    blk_bf16 = Glm5NextSparseMoeBlock(cfg_bf16)
    blk_bf16.process_weights_after_loading("cpu")
    assert blk_bf16.fp8_experts is False and blk_bf16._use_fused is False


def test_module_tree_matches_loader_expectations():
    """The assembled tree == the loader's expected target set, name for name — the loader
    <-> model integration seam, pinned without a checkpoint.
    """
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
    """The submodule's own seam (prepare_inputs -> declare_step -> preprocess
    -> forward_batched), with a greedy stand-in for the Triton sampler.
    """
    from types import SimpleNamespace

    h = _Harness(seed=0)
    sub, resources = h.sub, h.resources
    resources[SAMPLER] = _GreedySampler()  # forward_batched samples in-forward

    def step(walk, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rids = list(inputs)
        node_in = [
            sub.prepare_inputs(walk, SimpleNamespace(request_id=f"wire-{rid}", rid_handle=rid),
                               {"text_inputs": [ids]})
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
        rows = out.row_outputs["new_token"]
        return {rid: rows[i : i + 1] for i, rid in enumerate(rids)}

    prompts = {
        "r0": torch.tensor([5, 9, 13, 7, 21], dtype=torch.long),
        "r1": torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=torch.long),
    }
    h.ingest("r0", "r1")
    t1 = step("prefill", prompts)
    t2 = step("decode", t1)
    t3 = step("decode", t2)
    for rid, prompt in prompts.items():
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
        sub.prepare_inputs("prefill", SimpleNamespace(request_id="wire-long", rid_handle="long"),
                           {"text_inputs": [too_long]})
    assert sub.max_batch_size("decode") == h.usable

    h.remove("r0", "r1", "long")
    assert h.pool.num_free_slots == h.usable
    assert _holders(h.pool) == set()


@pytest.mark.parametrize("version,mtp,kpe", [
    ("0.6.18", 0, 0), ("0.7.0", 0, 0), ("0.6.17", 0, 64), ("0.6.18", 3, 64), (None, 0, 0),
])
def test_rope_slot_follows_what_flashinfer_can_plan(monkeypatch, version, mtp, kpe):
    # an older FlashInfer can't plan a cache without the slot; a box without one keeps none
    import sys
    import types

    from mstar.model.glm5_next import config as config_mod

    fake = None if version is None else types.SimpleNamespace(__version__=version)
    monkeypatch.setitem(sys.modules, "flashinfer", fake)
    config_mod._flashinfer_before.cache_clear()
    cfg = Glm5NextModelConfig()
    cfg.mtp_num_draft_tokens = mtp
    try:
        assert cfg.mla_cache_kpe == kpe
    finally:
        config_mod._flashinfer_before.cache_clear()
