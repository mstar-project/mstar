"""Which GDN kernels a linear-attention resource runs, decided once at build.

``AUTO`` (the default) resolves through ``current_platform``: FlashInfer's
kernels everywhere they ran before ``AUTO`` existed, AITER's on AMD GPUs. A
backend named in a spec or a deployment's YAML is built as named, and fails at
build where it cannot run. AITER's chunked prefill cannot be captured, so its
resource refuses prefill buckets past a token bound (the CUDA graph runner
then never captures them; see test_accelerator_graph_capture.py).

Everything here runs on the CPU: ``current_platform`` is pointed at a chosen
``Platform``, builds touch no device, and no kernel launches.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar import platforms
from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig, PackedAcceleratorGraphConfig
from mstar.engine.resources import BucketKey, apply_yaml_overrides
from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo
from mstar.engine.resources.linear_attn import (
    LinearAttnBackend,
    LinearAttnConfig,
    LinearAttnManager,
    LinearAttnSpec,
    LinearAttnVariant,
    gdn_kernels_aiter,
)
from mstar.engine.resources.linear_attn.base import resolve_linear_attn_backend
from mstar.engine.resources.linear_attn.gdn import GDNManager
from mstar.engine.resources.linear_attn.gdn_kernels import (
    FlashInferDecode,
    FlashInferGDNKernels,
    FlashInferPrefill,
)
from mstar.engine.resources.linear_attn.gdn_kernels_aiter import (
    DEFAULT_MAX_CAPTURED_PREFILL_TOKENS,
    AiterChunkedPrefill,
    AiterGDNKernels,
    AiterSequentialDecode,
    AiterSequentialPrefill,
    FlyDSLDecode,
)
from mstar.engine.resources.linear_attn.mamba2 import Mamba2Manager
from mstar.engine.resources.linear_attn.wrappers import GDNMixedWrapper
from mstar.engine.resources.recurrent import (
    DeltaNetGeometry,
    Mamba2Geometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import SlotLease
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, XpuPlatform

CPU = torch.device("cpu")
POOL, BACKEND = "gdn_state", "linear_attn"
QWEN35 = DeltaNetGeometry(num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel_size=4)


class _Mi300x(RocmPlatform):
    """An AMD GPU whose architecture is known without a device call."""

    def arch(self, device=None) -> str:
        return "gfx942"


@pytest.fixture
def on_platform(monkeypatch):
    """Make ``current_platform`` resolve to the given instance."""

    def use(platform):
        monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
        return platform

    return use


@pytest.fixture
def aiter_installed(monkeypatch):
    """The AITER strategy probes for its packages at build; pretend both exist."""
    monkeypatch.setattr(gdn_kernels_aiter, "_installed", lambda name: True)


def pool_spec(
    geometry=QWEN35, state_dtype=torch.float32, disable_sink=False,
) -> RecurrentStateSpec:
    return RecurrentStateSpec(
        POOL, {"llm"},
        RecurrentStateConfig(
            num_layers=2, blocks=geometry.to_blocks(state_dtype=state_dtype),
            max_slots=8, disable_sink_slot=disable_sink,
        ),
    )


def build(device=CPU, pool=None, **config):
    pool = pool or pool_spec()
    config.setdefault("variant", LinearAttnVariant.GDN)
    return LinearAttnManager.build(
        LinearAttnSpec(BACKEND, {"llm"}, LinearAttnConfig(recurrent_state=POOL, **config)),
        EngineResourceInfo(device=device, dependencies={POOL: pool}),
    )


def lease(walk: str, bs: int, num_tokens: int) -> SlotLease:
    return SlotLease(slot=0, bucket=BucketKey(graph_walk=walk, bs=bs, num_tokens=num_tokens))


def packed(bs: int, num_tokens: int) -> CGSlotSpec:
    """A packed prefill bucket, whose one row may hold every token."""
    config = PackedAcceleratorGraphConfig(
        capture_graph_walk="prefill_text", capture_token_lengths=[num_tokens],
        make_node_input=None,
    )
    return CGSlotSpec(BucketKey(graph_walk="prefill_text", bs=bs, num_tokens=num_tokens), 0, config)


def batched(bs: int, tokens_per_row: int = 1) -> CGSlotSpec:
    """A batched bucket: decode at one token a row."""
    config = BatchedAcceleratorGraphConfig(
        capture_graph_walk="decode",
        single_request_inputs=SimpleNamespace(input_seq_len=tokens_per_row),
    )
    return CGSlotSpec(
        BucketKey(graph_walk="decode", bs=bs, num_tokens=bs * tokens_per_row), 0, config,
    )


# --- resolution ----------------------------------------------------------------


def test_the_default_backend_is_auto():
    assert LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN).backend \
        is LinearAttnBackend.AUTO


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        (CudaPlatform(), LinearAttnBackend.FLASHINFER),
        (CpuPlatform(), LinearAttnBackend.FLASHINFER),
        (XpuPlatform(), LinearAttnBackend.FLASHINFER),
        (RocmPlatform(), LinearAttnBackend.AITER),
    ],
    ids=repr,
)
def test_auto_resolves_to_the_platforms_kernels(on_platform, platform, expected):
    on_platform(platform)
    assert resolve_linear_attn_backend(LinearAttnBackend.AUTO, torch.device("cuda")) is expected


@pytest.mark.parametrize(
    "platform", [CudaPlatform(), CpuPlatform(), XpuPlatform(), RocmPlatform()], ids=repr,
)
def test_auto_on_the_cpu_device_is_what_it_was(on_platform, platform):
    """A resource built on the CPU device (dummy mode, CPU tests) runs no
    kernel, so it resolves as it did before AUTO, on any machine."""
    on_platform(platform)
    assert resolve_linear_attn_backend(LinearAttnBackend.AUTO, CPU) is LinearAttnBackend.FLASHINFER


@pytest.mark.parametrize("platform", [CudaPlatform(), RocmPlatform()], ids=repr)
@pytest.mark.parametrize("backend", [LinearAttnBackend.FLASHINFER, LinearAttnBackend.AITER])
def test_an_explicit_backend_is_kept(on_platform, platform, backend):
    on_platform(platform)
    assert resolve_linear_attn_backend(backend, CPU) is backend


# --- builds --------------------------------------------------------------------


@pytest.mark.parametrize("platform", [CudaPlatform(), CpuPlatform()], ids=repr)
def test_auto_builds_flashinfer_where_it_did_before(on_platform, platform):
    on_platform(platform)
    manager = build()
    assert type(manager) is GDNManager
    assert type(manager._kernels) is FlashInferGDNKernels
    prefill = manager._get_wrapper("main", is_decode=False)
    decode = manager._get_wrapper("main", is_decode=True)
    assert type(prefill._kernel) is FlashInferPrefill
    assert type(decode._kernel) is FlashInferDecode


def test_auto_builds_aiter_on_an_amd_gpu(on_platform, aiter_installed):
    on_platform(_Mi300x())
    manager = build(device=torch.device("cuda"))
    kernels = manager._kernels
    assert type(kernels) is AiterGDNKernels
    assert type(manager._get_wrapper("main", is_decode=False)._kernel) is AiterChunkedPrefill
    assert type(manager._get_wrapper("main", is_decode=True)._kernel) is FlyDSLDecode
    captured = manager._get_wrapper("main", is_decode=False, lease=lease("prefill_text", 4, 256))
    assert type(captured._kernel) is AiterSequentialPrefill


def test_flashinfer_named_on_an_amd_gpu_fails_at_build(on_platform):
    on_platform(_Mi300x())
    with pytest.raises(ValueError, match="CUTLASS kernels for NVIDIA GPUs"):
        build(device=torch.device("cuda"), backend=LinearAttnBackend.FLASHINFER)


def test_the_flashinfer_prefill_state_dtype_follows_the_nvidia_arch(on_platform, monkeypatch):
    """SM100 reads a bf16 pool as is; SM90 round-trips it through fp32."""
    on_platform(CudaPlatform())
    monkeypatch.setattr(FlashInferGDNKernels, "_check_state_dtype", staticmethod(lambda *a: None))
    for capability, expected in (((9, 0), torch.float32), ((10, 0), torch.bfloat16)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
        kernels = build(
            device=torch.device("cuda"), pool=pool_spec(state_dtype=torch.bfloat16),
        )._kernels
        assert kernels.prefill_dtype is expected


def test_the_flashinfer_state_dtype_table_ignores_an_amd_capability(on_platform, monkeypatch):
    """An MI300X reports capability (9, 4); the table must not read it as SM90,
    and on a CPU device it never asks at all."""
    on_platform(CpuPlatform())
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: calls.append(1))
    assert build()._kernels.prefill_dtype is torch.float32
    assert calls == []


def test_aiter_needs_the_sink_slot(on_platform, aiter_installed):
    on_platform(_Mi300x())
    with pytest.raises(ValueError, match="need the pool's sink slot"):
        build(device=torch.device("cuda"), pool=pool_spec(disable_sink=True))


def test_aiter_refuses_a_state_dtype_its_kernels_do_not_keep(on_platform, aiter_installed):
    on_platform(_Mi300x())
    with pytest.raises(NotImplementedError, match="fp32 or bf16"):
        build(device=torch.device("cuda"), pool=pool_spec(state_dtype=torch.float16))
    bf16 = build(device=torch.device("cuda"), pool=pool_spec(state_dtype=torch.bfloat16))
    assert bf16._kernels.state_dtype is torch.bfloat16


def test_aiter_without_aiter_fails_at_build(on_platform, monkeypatch):
    on_platform(_Mi300x())
    monkeypatch.setattr(gdn_kernels_aiter, "_installed", lambda name: name != "aiter")
    with pytest.raises(ImportError, match="needs AMD's aiter package"):
        build(device=torch.device("cuda"))


# --- decode kernel choice --------------------------------------------------------


def test_flydsl_decode_is_refused_off_its_fixed_scale(on_platform, aiter_installed):
    """FlyDSL's decode bakes in 1/sqrt(K); any other scale takes the
    sequential kernel, decided at build rather than on the first call."""
    on_platform(_Mi300x())
    cuda = torch.device("cuda")
    assert build(device=cuda, sm_scale=128 ** -0.5)._kernels.flydsl_decode
    manager = build(device=cuda, sm_scale=0.1)
    assert not manager._kernels.flydsl_decode
    assert type(manager._get_wrapper("main", is_decode=True)._kernel) is AiterSequentialDecode


def test_flydsl_decode_needs_flydsl_and_its_architectures(on_platform, monkeypatch):
    cuda = torch.device("cuda")
    monkeypatch.setattr(gdn_kernels_aiter, "_installed", lambda name: name != "flydsl")
    on_platform(_Mi300x())
    assert not build(device=cuda)._kernels.flydsl_decode

    monkeypatch.setattr(gdn_kernels_aiter, "_installed", lambda name: True)

    class _Mi210(RocmPlatform):
        def arch(self, device=None):
            return "gfx90a"

    on_platform(_Mi210())
    assert not build(device=cuda)._kernels.flydsl_decode


# --- captured prefill bound ------------------------------------------------------


def test_aiter_refuses_prefill_buckets_past_its_bound(on_platform, aiter_installed):
    on_platform(_Mi300x())
    manager = build(device=torch.device("cuda"))
    bound = DEFAULT_MAX_CAPTURED_PREFILL_TOKENS
    assert manager.capture_refusal(packed(16, bound)) is None
    assert "max_captured_prefill_tokens" in manager.capture_refusal(packed(1, 2 * bound))
    # a batched bucket's row holds only its share of the tokens
    assert manager.capture_refusal(batched(16, tokens_per_row=bound // 16)) is None
    # decode is never refused, however wide its bucket
    assert manager.capture_refusal(batched(4 * bound)) is None


def test_a_bound_of_zero_leaves_decode_captured(on_platform, aiter_installed):
    """``max_captured_prefill_tokens: 0`` sends every prefill bucket to eager;
    one-token rows take the decode kernel and stay captured."""
    on_platform(_Mi300x())
    manager = build(device=torch.device("cuda"), max_captured_prefill_tokens=0)
    assert manager.capture_refusal(packed(1, 32)) is not None
    assert manager.capture_refusal(batched(128)) is None


def test_the_bound_is_a_deployment_setting(on_platform, aiter_installed):
    on_platform(_Mi300x())
    pool = pool_spec()
    spec = LinearAttnSpec(
        BACKEND, {"llm"},
        LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN),
    )
    apply_yaml_overrides(
        [pool, spec],
        {"resources": {BACKEND: {"backend": "aiter", "max_captured_prefill_tokens": 64}}},
    )
    assert spec.config.backend is LinearAttnBackend.AITER
    manager = LinearAttnManager.build(
        spec, EngineResourceInfo(device=torch.device("cuda"), dependencies={POOL: pool}),
    )
    assert manager._kernels.uncapturable_prefill(64) is None
    assert manager._kernels.uncapturable_prefill(128) is not None
    with pytest.raises(ValueError, match=">= 0"):
        spec.apply_yaml_overrides(max_captured_prefill_tokens=-1)


def test_flashinfer_captures_every_prefill_bucket(on_platform):
    on_platform(CudaPlatform())
    manager = build()
    assert manager.capture_refusal(packed(1, 1 << 20)) is None


def test_yaml_rejects_an_unknown_backend():
    spec = LinearAttnSpec(
        BACKEND, {"llm"},
        LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN),
    )
    with pytest.raises(ValueError, match="'cutlass' is not a valid LinearAttnBackend"):
        spec.apply_yaml_overrides(backend="cutlass")


# --- combined walks: a split walk's mixed step ----------------------------------

MIXED = CombinedSplit("mixed", prefill_walks=("prefill_text",))


def test_a_split_walk_runs_both_halves_on_aiters_kernels(on_platform, aiter_installed):
    """A mixed step's split takes its prefill and decode halves from the
    resource's kernels, eagerly and under capture. The caller normalises q and
    k for both halves, so neither does it again."""
    on_platform(_Mi300x())
    manager = build(device=torch.device("cuda"), splits=(MIXED,))
    eager = manager._get_wrapper("main", False, split=MIXED)
    assert type(eager) is GDNMixedWrapper
    assert type(eager._prefill._kernel) is AiterChunkedPrefill
    assert type(eager._decode._kernel) is FlyDSLDecode
    captured = manager._get_wrapper("main", False, lease=lease("mixed", 8, 64), split=MIXED)
    assert type(captured._prefill._kernel) is AiterSequentialPrefill
    assert type(captured._decode._kernel) is FlyDSLDecode
    for wrapper in (eager, captured):
        assert not wrapper.qk_l2norm_in_kernel
        for half in (wrapper._prefill, wrapper._decode):
            assert not half.qk_l2norm_in_kernel and not half._kernel._kernels.qk_l2norm
    # the resource's own decode still normalises in the kernel
    assert manager._kernels.qk_l2norm
    assert manager._get_wrapper("main", is_decode=True).qk_l2norm_in_kernel


def test_a_split_walk_on_flashinfer_skips_only_the_decode_halfs_norm(on_platform):
    """FlashInfer's chunked prefill never normalises, and the decode half is
    told not to, since the caller already did."""
    on_platform(CudaPlatform())
    manager = build(splits=(MIXED,), sm_scale=0.05)
    mixed = manager._get_wrapper("main", False, split=MIXED)
    assert type(mixed._prefill._kernel) is FlashInferPrefill
    assert type(mixed._decode._kernel) is FlashInferDecode
    twin = mixed._decode._kernel._kernels
    assert (twin.qk_l2norm, twin.sm_scale, twin.has_sink) == (False, 0.05, True)
    assert manager._kernels.qk_l2norm


# --- Mamba-2 ---------------------------------------------------------------------


@pytest.mark.parametrize("platform", [CudaPlatform(), _Mi300x()], ids=repr)
@pytest.mark.parametrize("backend", list(LinearAttnBackend), ids=str)
def test_mamba2_runs_its_own_kernels_under_every_backend(on_platform, platform, backend):
    on_platform(platform)
    geometry = Mamba2Geometry(num_heads=4, head_dim=8, state_size=16, n_groups=2, conv_kernel_size=4)
    pool = RecurrentStateSpec(
        POOL, {"llm"}, RecurrentStateConfig(num_layers=1, blocks=geometry.to_blocks()),
    )
    manager = build(pool=pool, variant=LinearAttnVariant.MAMBA2, backend=backend)
    assert type(manager) is Mamba2Manager


# --- what the AITER prefill kernels stage per step (CPU tensors) ------------------


def _prefill_plan(real_spans, slots, has_state):
    from mstar.engine.resources.linear_attn.gdn_kernels import GDNPrefillPlan

    spans = tuple(s or 1 for s in real_spans)
    cu = [0]
    for s in spans:
        cu.append(cu[-1] + s)
    return GDNPrefillPlan(
        slots=torch.tensor(slots, dtype=torch.int32),
        cu_seqlens=torch.tensor(cu, dtype=torch.int64),
        has_state=torch.tensor(has_state),
        num_rows=len(spans),
        token_mask=torch.ones(cu[-1]),
        real_spans=tuple(real_spans),
        spans=spans,
        real_tokens=sum(real_spans),
        borrowed=sum(1 for s in real_spans if s == 0),
    )


def _aiter_kernels():
    return AiterGDNKernels(
        geometry=QWEN35, state_dtype=torch.float32, sm_scale=None, qk_l2norm=True,
        max_captured_prefill_tokens=DEFAULT_MAX_CAPTURED_PREFILL_TOKENS, flydsl_decode=False,
    )


def test_the_sequential_prefill_slot_map():
    """The kernel reads a row's state at column 0 and stores it after token t
    where column t is not -1: each row's slot sits at column 0 and at its last
    token, and a row on the sink (padding, nothing to extend) is all -1, so the
    kernel skips it."""
    kernel = _aiter_kernels().captured_prefill(bs=4, num_tokens=8)
    kernel.plan(_prefill_plan((3, 1, 0, 0), [5, 3, 0, 0], [True, False, False, False]))
    expected = torch.full((4, 8), -1, dtype=torch.int32)
    expected[0, 0] = expected[0, 2] = 5
    expected[1, 0] = 3
    assert torch.equal(kernel._map, expected)
    # a later, narrower layout leaves no stale cells behind
    kernel.plan(_prefill_plan((8, 0, 0, 0), [7, 0, 0, 0], [False] * 4))
    expected = torch.full((4, 8), -1, dtype=torch.int32)
    expected[0, 0] = expected[0, 7] = 7
    assert torch.equal(kernel._map, expected)


def test_a_bucket_refuses_a_layout_it_cannot_hold():
    kernel = _aiter_kernels().captured_prefill(bs=2, num_tokens=8)
    with pytest.raises(ValueError, match="does not fit"):
        kernel.plan(_prefill_plan((9, 0), [1, 0], [False, False]))


def test_fresh_rows_read_zeros_and_carried_rows_keep_their_state():
    """Fresh rows' slots are zeroed before the kernel reads them; a carried
    row's index lands on the sink instead, leaving its own slot as it was."""
    from mstar.engine.resources.linear_attn.gdn_kernels_aiter import _FreshSlots

    fresh = _FreshSlots()
    fresh.plan(_prefill_plan((4, 2, 0), [5, 3, 0], [True, False, False]))
    state = torch.ones(8, 2, 2, 2)
    fresh.zero(state)
    assert torch.equal(state[5], torch.ones(2, 2, 2)), "a carried row's state was cleared"
    assert not state[3].any() and not state[0].any()
    untouched = [1, 2, 4, 6, 7]
    assert torch.equal(state[untouched], torch.ones(5, 2, 2, 2))
