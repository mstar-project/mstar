"""Unit tests for the weight-absorbed MLA attention backend.

Three things are worth pinning, and all of them run on CPU:

* The kernel predicate. ``flashinfer.mla.BatchMLAPagedAttentionWrapper`` is
  hard-locked to the real Kimi latent dims and an off-dim call corrupts the
  CUDA context rather than raising, so the decision has to be made — correctly
  — before any kernel is built.
* The capture decision it drives. The SDPA fallback loops over requests in
  Python, so a walk that uses it must not be captured; a walk that doesn't
  touch the resource still must be.
* The SDPA path's own numbers, against a naive per-request reference, since
  that path is what serves every config the kernel cannot.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.cuda_graph_runner import capture_blocked_by
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    KVConfig,
    KVLayout,
    KVSpec,
    KVStep,
    Segment,
    StepContext,
    SubmoduleStep,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.attn.mla import MlaAbsorbManager, mla_kernel_available_for
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.convenience import AttentionCallable
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import KVPlanOutput, SequenceView

# Real Kimi latent dims; flashinfer's MLA wrapper is hard-locked to these.
REAL_CKV, REAL_KPE = 512, 64
CUDA = torch.device("cuda:0")  # constructible without a GPU present

PAGE_SIZE = 4
CKV, KPE = 6, 2
HEAD_DIM = CKV + KPE
HEADS = 3
MAX_PAGES = 32
SCALE = 0.125


def _kv_config(
    head_dim: int = HEAD_DIM, layout: KVLayout = KVLayout.MLA, num_layers: int = 1,
) -> KVConfig:
    return KVConfig(
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        max_seq_len=64,
        max_num_pages=MAX_PAGES,
        page_size=PAGE_SIZE,
        num_qo_heads=HEADS,
        layout=layout,
    )


def _manager(
    kv_config: KVConfig | None = None, mla_ckv_dim: int | None = CKV,
) -> MlaAbsorbManager:
    return MlaAbsorbManager(
        kv_cache="kv",
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_config=kv_config or _kv_config(),
        softmax_scale=SCALE,
        mla_ckv_dim=mla_ckv_dim,
    )


@pytest.fixture
def sm90(monkeypatch):
    """Pretend we are on a Hopper GPU, without needing one."""
    import mstar.engine.resources.attn.mla as mla_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (9, 0))
    # _mla_kernel_available is functools.cache'd and imports flashinfer; stub it
    # so the caller's own logic is what is under test.
    monkeypatch.setattr(
        mla_mod, "_mla_kernel_available",
        lambda ckv, kpe, sm: (ckv, kpe, sm) == (REAL_CKV, REAL_KPE, 9),
    )


class TestKernelPredicate:
    def test_true_only_for_real_dims_on_sm90(self, sm90):
        cfg = _kv_config(head_dim=REAL_CKV + REAL_KPE)
        assert mla_kernel_available_for(cfg, REAL_CKV, CUDA) is True

    def test_false_for_reduced_dims(self, sm90):
        # the reduced test config: right backend, wrong latent width
        assert mla_kernel_available_for(_kv_config(head_dim=40), 32, CUDA) is False

    def test_false_without_ckv_dim(self, sm90):
        assert mla_kernel_available_for(_kv_config(head_dim=40), None, CUDA) is False

    def test_is_total_on_cpu(self, sm90):
        """Must return False, not raise: torch.cuda.get_device_capability rejects
        a CPU device, and the absorbed-SDPA fallback has to stay reachable
        there."""
        cfg = _kv_config(head_dim=REAL_CKV + REAL_KPE)
        assert mla_kernel_available_for(cfg, REAL_CKV, torch.device("cpu")) is False


class TestCaptureDecision:
    """No wrapper is planned on the SDPA path, so a captured graph could only
    hold a paged wrapper — capture must be cancelled, not attempted."""

    def _step(self, *resource_keys: str) -> SubmoduleStep:
        steps = {
            key: (KVStep() if key == "kv" else AttentionStep())
            for key in resource_keys
        }
        return SubmoduleStep(segments=[], steps=steps)

    def test_not_blocked_when_kernel_serves_the_dims(self, sm90):
        manager = MlaAbsorbManager(
            kv_cache="kv", device=CUDA, dtype=torch.float32,
            kv_config=_kv_config(head_dim=REAL_CKV + REAL_KPE),
            mla_ckv_dim=REAL_CKV,
        )
        assert manager.mla_kernel_available() is True
        assert manager.capture_blocked is None

    @pytest.mark.parametrize(
        "ckv, head_dim",
        [(32, 40), (None, 40), (REAL_CKV, REAL_CKV + 128)],
    )
    def test_blocked_when_absorbed_falls_back_to_sdpa(self, sm90, ckv, head_dim):
        manager = MlaAbsorbManager(
            kv_cache="kv", device=CUDA, dtype=torch.float32,
            kv_config=_kv_config(head_dim=head_dim), mla_ckv_dim=ckv,
        )
        reason = manager.capture_blocked
        assert reason is not None
        assert "MLA" in reason and "eager" in reason

    def test_blocked_on_cpu_device(self, sm90):
        manager = _manager(_kv_config(head_dim=REAL_CKV + REAL_KPE), REAL_CKV)
        assert manager.capture_blocked is not None

    def test_other_backends_never_block(self):
        manager = FlashInferManager(
            kv_cache="kv", device=torch.device("cpu"),
            dtype=torch.float32, kv_config=_kv_config(layout=KVLayout.NHD),
        )
        assert manager.capture_blocked is None

    def test_a_walk_that_uses_the_resource_is_blocked(self):
        resources = {"kv": None, "attn": _manager()}
        reason = capture_blocked_by(self._step("kv", "attn"), resources)
        assert reason is not None and "attn" in reason

    def test_a_walk_that_does_not_touch_it_still_captures(self):
        """Per step, not per node: a model whose prefill needs the fallback
        keeps its decode graphs."""
        resources = {"kv": None, "attn": _manager()}
        assert capture_blocked_by(self._step("kv"), resources) is None
        assert capture_blocked_by(None, resources) is None

    def test_planning_a_captured_step_refuses(self):
        """The backstop for the two decisions drifting apart."""
        from mstar.engine.resources import SlotLease
        from mstar.engine.resources.step import BucketKey

        manager = _manager()
        ctx = StepContext(
            request_ids=("r0",), graph_walk="decode", slot=0, capture=True,
            slot_lease=SlotLease(
                slot=0, bucket=BucketKey(graph_walk="decode", bs=1, num_tokens=1),
            ),
        )
        with pytest.raises(AssertionError, match="absorbed MLA"):
            manager.plan(AttentionStep(segments=()), ctx)


class TestSdpaPlan:
    """The fallback's layout comes off the KV plan's views and nothing else."""

    @staticmethod
    def _view(rid: str, pages: list[int], length: int, to_compute: int):
        return SequenceView(
            request_id=rid, label="main", page_idxs=pages,
            length=length, to_compute=to_compute,
        )

    def _plan(self, views: list[SequenceView]):
        kv_out = KVPlanOutput(cpu_indptrs=None, views=views)
        return _manager()._sdpa_plan(kv_out)

    def test_slices_pack_in_plan_order(self):
        plan = self._plan([
            self._view("r0", [0, 1], length=5, to_compute=5),
            self._view("r1", [2], length=3, to_compute=3),
        ])
        assert [(r.q_start, r.seq_len) for r in plan.requests] == [(0, 5), (5, 3)]

    def test_total_len_is_the_context_after_this_step(self):
        """What the causal mask needs: cached tokens precede the fresh ones."""
        plan = self._plan([self._view("r0", [0, 1, 2], length=9, to_compute=2)])
        (req,) = plan.requests
        assert (req.seq_len, req.total_len) == (2, 9)
        assert req.page_indices.tolist() == [0, 1, 2]

    def test_a_stream_read_from_a_later_page_is_refused(self):
        view = SequenceView(
            request_id="r0", label="main", page_idxs=[0],
            length=2, to_compute=2, start=4,
        )
        with pytest.raises(AssertionError, match="first page"):
            self._plan([view])


class TestSdpaNumerics:
    """The eager path against a naive per-request reference, over the real
    MLA-layout cache and through the write the layer actually makes."""

    @pytest.fixture
    def stubbed_transfer(self, monkeypatch):
        class _StubTransfer:
            def __init__(self, *args, **kwargs):
                pass

            def get_kv_transfer_info(self):
                return None

            def cleanup(self):
                pass

        monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)

    def test_matches_a_naive_reference_across_steps(self, stubbed_transfer):
        torch.manual_seed(0)
        num_layers = 2
        cfg = _kv_config(num_layers=num_layers)
        kv = KVManager(
            cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert tuple(kv.kv_cache.tensor.shape) == (
            num_layers, MAX_PAGES, PAGE_SIZE, HEAD_DIM
        ), "MLA layout caches one latent per token: no K/V and no KV-head axis"

        attn = _manager(cfg)
        assert not attn.mla_kernel_available(), "expected the SDPA fallback on CPU"
        run = AttentionCallable(kv=kv, attn=attn)

        rids = ["a", "b"]
        for rid in rids:
            kv.ingest_request(rid)
        # every request's full latent history, per layer
        history = {
            (rid, lyr): torch.zeros(0, HEAD_DIM)
            for rid in rids for lyr in range(num_layers)
        }

        # prefill, two decodes, then a mixed step
        for spans in [(5, 3), (1, 1), (1, 1), (2, 6)]:
            segments = tuple(
                Segment(rid, "main", span)
                for rid, span in zip(rids, spans, strict=True)
            )
            kv_step = KVStep(segments=segments)
            ctx = StepContext(
                request_ids=tuple(rids), graph_walk="walk", slot=0, capture=False,
            )
            assert kv.admit(kv_step, ctx).ok
            ctx.plan_results["kv"] = kv.plan(kv_step, ctx)
            attn.plan(AttentionStep(segments=segments, causal=True), ctx)
            run.bind_step("main")

            total = sum(spans)
            for lyr in range(num_layers):
                run.set_layer_idx(lyr)
                q_nope = torch.randn(total, HEADS, CKV)
                q_pe = torch.randn(total, HEADS, KPE)
                kv_c = torch.randn(total, 1, CKV)
                k_pe = torch.randn(total, 1, KPE)

                got = run.run_mla(q_nope, q_pe, kv_c, k_pe)

                latents = torch.cat([kv_c, k_pe], dim=-1).squeeze(1)
                query = torch.cat([q_nope, q_pe], dim=-1)
                offset = 0
                for rid, span in zip(rids, spans, strict=True):
                    key = history[(rid, lyr)] = torch.cat(
                        [history[(rid, lyr)], latents[offset:offset + span]], dim=0
                    )
                    rows = slice(offset, offset + span)
                    torch.testing.assert_close(
                        got[rows],
                        _reference(query[rows], key, ckv=CKV, scale=SCALE),
                        rtol=1e-5, atol=1e-5,
                    )
                    offset += span

                # and the latents landed where the KV plan said they would
                for rid, span in zip(rids, spans, strict=True):
                    stream = kv._streams[rid]["main"]
                    resident = stream.stored_len + span
                    gathered = kv.kv_cache.tensor[lyr][
                        stream.page_indices
                    ].reshape(-1, HEAD_DIM)[:resident]
                    torch.testing.assert_close(gathered, history[(rid, lyr)])

            kv.commit(kv_step, ctx)


def _reference(
    query: torch.Tensor, key: torch.Tensor, ckv: int, scale: float,
) -> torch.Tensor:
    """One request's causal MLA over its whole latent history; the value is the
    ckv slice of the same latent (the absorbed path's ``kv_c``)."""
    span = query.shape[0]
    old_len = key.shape[0] - span
    scores = torch.einsum("hqd,kd->hqk", query.transpose(0, 1), key) * scale
    q_pos = old_len + torch.arange(span)
    mask = torch.arange(key.shape[0])[None, :] <= q_pos[:, None]
    weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    return torch.einsum("hqk,kd->hqd", weights, key[:, :ckv]).transpose(0, 1)


class TestBackendSelection:
    def test_mla_spec_builds_the_absorbed_manager(self):
        spec = AttentionSpec(
            resource_key="attn",
            nodes={"llm"},
            config=AttentionConfig(
                kv_cache="kv",
                backend=AttnBackend.MLA,
                softmax_scale=SCALE,
                mla_ckv_dim=CKV,
            ),
        )
        assert spec.depends_on() == {"kv"}
        manager = AttentionManager.build(
            spec,
            EngineResourceInfo(
                device=torch.device("cpu"),
                kv_dtype=torch.float32,
                dependencies={
                    "kv": KVSpec(
                        resource_key="kv", nodes={"llm"}, config=_kv_config(),
                    ),
                },
            ),
        )
        assert isinstance(manager, MlaAbsorbManager)
        assert manager.depends_on() == {"kv"}

    def test_the_layer_still_writes_through_the_kv_resource(self):
        """One compressed latent per token rather than a K/V pair, but the
        write is the KV resource's either way — see `write_latent`."""
        assert _manager().requires_kv_write is True
