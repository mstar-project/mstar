"""Window and softmax-scale plumbing on CPU, plus single-GPU numerical and replay tests that
check FlashInfer against an explicit FP32 PyTorch attention reference."""

import importlib.util
import itertools
import os
import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import torch

from mstar.engine.resources.attn import wrappers
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    CrossAttentionConfig,
    CrossAttentionSpec,
    derive_kv_groups,
)
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.kv.config import (
    DEFAULT_PAGE_SIZE,
    KVGroup,
    KVLayout,
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    StreamLayout,
)
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs, SequenceView, build_paged_indptrs
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.spec import CombinedSplit, resolve_spec_dependencies
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep

# A captured plan stages its host arrays behind a CUDA event.
CAPTURED = (False, True) if torch.cuda.is_available() else (False,)


def geometry(max_seq_len=64):
    return PagedKVConfig(num_layers=1, num_qo_heads=4, num_kv_heads=2, head_dim=128,
                         max_seq_len=max_seq_len, page_size=16, max_num_pages=6)


WINDOW = 4


def window_pool(sliding=(1, 2), **geometry):
    """By default layer 0 full, layers 1 and 2 keeping a window of 4 as an attention over the
    pool declares them: their pages hold 8 tokens."""
    cfg = PagedKVConfig(**{**dict(num_layers=3, num_qo_heads=4, num_kv_heads=2, head_dim=128,
                                  max_seq_len=64, page_size=16, max_num_pages=6), **geometry})
    resolve_spec_dependencies([
        KVSpec(resource_key="kv", nodes={"LLM"}, config=cfg),
        AttentionSpec(resource_key="local", nodes={"LLM"},
                      config=AttentionConfig("kv", sliding_window=WINDOW, layers=sliding)),
        AttentionSpec(resource_key="full", nodes={"LLM"}, config=AttentionConfig("kv")),
    ])
    return cfg


def build_manager(window, device="cpu", backend=AttnBackend.FLASHINFER, kv_config=None, **config):
    """An attention over ``kv_config``, or over a fresh one-layer pool as long as the window,
    which then keeps every token, so the window only masks."""
    kv = KVSpec(resource_key="kv", nodes={"LLM"}, config=kv_config or geometry(window or 64))
    spec = AttentionSpec(resource_key="attn", nodes={"LLM"},
                         config=AttentionConfig("kv", backend=backend, sliding_window=window, **config))
    if kv_config is None:
        resolve_spec_dependencies([kv, spec])
    return AttentionManager.build(spec, EngineResourceInfo(
        device=torch.device(device), dependencies={"kv": kv}, kv_dtype=torch.bfloat16,
    ))


def context(lengths, query_lengths, captured=False, preplan=False):
    # Deliberately noncontiguous pages, with distinct streams for two requests.
    pages = ([3, 0], [4, 1])
    views = [SequenceView(f"r{i}", "main", list(pages[i][: (n + 15) // 16]), n, q)
             for i, (n, q) in enumerate(zip(lengths, query_lengths, strict=True))]
    plan = KVPlanOutput(build_paged_indptrs(views, 16), views)
    walk = "decode" if all(q == 1 for q in query_lengths) else "prefill"
    lease = SlotLease(0, BucketKey(walk, len(lengths), sum(query_lengths))) if captured else None
    return StepContext(tuple(v.request_id for v in views), walk, 0, False,
                       is_preplan=preplan, slot_lease=lease, plan_results={"kv": {"main": plan}})


class SlidingWindowTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"MSTAR_WORKSPACE_BUFFER_MB": "1"}))
        self.fake = SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=MagicMock(side_effect=lambda *a, **kw: MagicMock()),
            BatchDecodeWithPagedKVCacheWrapper=MagicMock(side_effect=lambda *a, **kw: MagicMock()),
        )
        self.enterContext(patch.dict("sys.modules", {"flashinfer": self.fake}))

    def test_window_validation(self):
        for value in (0, -1, True, False, 4.0, "4"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "sliding_window"):
                AttentionConfig("kv", sliding_window=value)
        for value in (None, 1, 4):
            self.assertEqual(AttentionConfig("kv", sliding_window=value).sliding_window, value)

    def test_unsupported_backend_fails_before_dependency_access(self):
        # MLA reads sm_scale but not the window
        cases = [(AttnBackend.FLASHINFER_MLA, "sliding_window")] + [
            (backend, name)
            for backend in (AttnBackend.DENSE, AttnBackend.FLEX, AttnBackend.XPU_PAGED)
            for name in ("sliding_window", "sm_scale")
        ]
        for backend, name in cases:
            config = AttentionConfig("kv", backend=backend, **{name: 4})
            spec = AttentionSpec(resource_key="attn", nodes={"LLM"}, config=config)
            with self.subTest(backend=backend, config=name), \
                    self.assertRaisesRegex(ValueError, f"{backend.name}.*ignores {name}"):
                AttentionManager.build(spec, EngineResourceInfo(device=torch.device("cpu")))

    def test_window_and_scale_change_the_fingerprint(self):
        plain = build_manager(None)
        keys = {m.fingerprint() for m in (plain, build_manager(4), build_manager(None, sm_scale=0.25))}
        self.assertEqual(len(keys), 3)
        # neither declared: the base's key, which every existing model already has
        self.assertEqual(plain.fingerprint(), AttentionManager.fingerprint(plain))

    def test_mixed_wrapper_hands_the_scale_to_both_halves(self):
        mixed = wrappers.FlashInferMixedWrapper(
            decode_workspace=None, prefill_workspace=None, num_qo_heads=4, num_kv_heads=2,
            head_dim=128, page_size=16, batch_size=2, max_total_tokens=8, max_num_pages=8,
            max_prefill_rows=1, device=torch.device("cpu"), sm_scale=0.5,
        )
        for half in (mixed._decode, mixed._prefill):
            self.assertEqual(half.sm_scale, 0.5)

    def test_unbounded_dense_backend_and_fallback_are_unchanged(self):
        module = "mstar.engine.resources.attn.dense"
        with patch(f"{module}._fa3_unavailable_reason", return_value=None), \
                patch(f"{module}.DenseAttentionManager") as constructor:
            self.assertIs(build_manager(None, backend=AttnBackend.DENSE), constructor.return_value)
        with patch(f"{module}._fa3_unavailable_reason", return_value="test"), \
                patch(f"{module}._warn_dense_fallback"):
            manager = build_manager(None, backend=AttnBackend.DENSE)
        self.assertIsInstance(manager, FlashInferManager)
        self.assertEqual(manager._window_left, -1)

    def test_both_wrappers_receive_window_and_scale_and_reuse_plans(self):
        for (window, expected), scale in itertools.product(((None, -1), (1, 0), (4, 3)), (None, 0.25)):
            for captured in CAPTURED:
                for query_lengths in ([7, 5], [1, 1]):
                    with self.subTest(window=window, scale=scale, captured=captured, queries=query_lengths):
                        manager = build_manager(window, sm_scale=scale)
                        ctx = context([7, 5], query_lengths, captured)
                        manager.plan(AttentionStep(), ctx)
                        wrapper = manager._current_plan_states["main"]
                        kwargs = wrapper.attn_wrapper.plan.call_args.kwargs
                        self.assertEqual(kwargs["window_left"], expected)
                        self.assertEqual(kwargs["sm_scale"], scale)
                        self.assertEqual(wrapper.accelerator_graph, captured)
                        manager.plan(AttentionStep(), ctx)
                        self.assertIs(manager._current_plan_states["main"], wrapper)
                        self.assertEqual(wrapper.attn_wrapper.plan.call_count, 2)

    def test_noncausal_rejected_even_when_promoting_preplan(self):
        manager = build_manager(4)
        ctx = context([7, 5], [1, 1], captured=True, preplan=True)
        manager.plan(AttentionStep(), ctx)
        ctx.is_preplan = False
        with self.assertRaisesRegex(ValueError, "causal"):
            manager.plan(AttentionStep(causal=False), ctx)
        planned = manager._preplan_states["main"]
        manager.plan(AttentionStep(), ctx)
        self.assertIs(manager._current_plan_states["main"], planned)
        self.assertEqual(planned.attn_wrapper.plan.call_count, 1)

    def test_unbounded_noncausal_prefill_still_supported(self):
        manager = build_manager(None)
        manager.plan(AttentionStep(causal=False), context([7, 5], [7, 5]))
        args = manager._current_plan_states["main"].attn_wrapper.plan.call_args.kwargs
        self.assertFalse(args["causal"])
        self.assertEqual(args["window_left"], -1)

    def test_two_attention_resources_plan_and_commit_one_kv_resource(self):
        ctx = context([7, 5], [7, 5])
        kv_output = ctx.plan_results.pop("kv")

        class RecordingKV(Resource):
            plans = 0
            commits = 0

            @classmethod
            def build(cls, spec, info):
                return cls()

            @property
            def supports_preplan(self):
                return True

            def plan(self, step, ctx):
                self.plans += 1
                return kv_output

            def commit(self, step, ctx):
                self.commits += 1

        kv = RecordingKV()
        local, global_ = build_manager(4), build_manager(None)
        runner = StepRunner({"local": local, "global": global_, "kv": kv})
        step = SubmoduleStep(steps={"kv": KVStep(), "local": AttentionStep(), "global": AttentionStep()})
        step.set_ctx(ctx)
        runner.plan(step)
        runner.commit(step)
        self.assertEqual((kv.plans, kv.commits), (1, 1))
        for manager, expected in ((local, 3), (global_, -1)):
            args = manager._current_plan_states["main"].attn_wrapper.plan.call_args.kwargs
            self.assertEqual(args["window_left"], expected)
            self.assertIs(args["paged_kv_indices"], kv_output["main"].cpu_indptrs.paged_kv_indices)

    def test_window_pool_attention_reads_its_group(self):
        pool = window_pool()
        full_ctx, window_ctx = context([7, 5], [7, 5]), context([7, 5], [7, 5])
        plans = KVPlanOutputs(full_ctx.plan_results["kv"], later_groups=(window_ctx.plan_results["kv"],))
        for window, expected in ((None, full_ctx), (WINDOW, window_ctx)):
            with self.subTest(window=window):
                manager = build_manager(window, kv_config=pool, layers=(1, 2) if window else None)
                self.assertEqual(manager._wrapper_kv_kwargs["page_size"], 8 if window else 16)
                self.assertEqual(manager._wrapper_kv_kwargs["max_num_pages"], pool.num_units)
                manager.plan(AttentionStep(), StepContext(
                    ("r0", "r1"), "prefill", 0, False, plan_results={"kv": plans},
                ))
                args = manager._current_plan_states["main"].attn_wrapper.plan.call_args.kwargs
                self.assertIs(
                    args["paged_kv_indices"],
                    expected.plan_results["kv"]["main"].cpu_indptrs.paged_kv_indices,
                )

    def test_capture_bound_holds_a_full_pool_and_padding_rows(self):
        """The full group's capture bound covers a real row holding most of the pool plus
        padding rows naming a sink page per page of their span."""
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.engine.resources.attn.base import captured_page_capacity
        from mstar.engine.resources.kv.manager import KVManager
        from mstar.engine.resources.kv.transfer import TransferEngineInfo

        cfg = window_pool(num_qo_heads=1, num_kv_heads=1, head_dim=4, max_seq_len=1024, max_num_pages=8)
        kv = KVManager(cfg=cfg, name="kv", joint_comm_group=None,
                       transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
                       device=torch.device("cpu"), dtype=torch.float32)
        kv.ingest_request("r")
        while kv._arena.num_free > 3:
            step, ctx = KVStep(segments=(Segment("r", "main", 16),)), StepContext(("r",), "walk", 0, False)
            self.assertTrue(kv.admit(step, ctx).ok)
            kv.plan(step, ctx)
            kv.commit(step, ctx)
        pads = ("p0", "p1", "p2")
        step = KVStep(segments=(Segment("r", "main", 1),) + tuple(Segment(p, "main", 4 * 16) for p in pads))
        ctx = StepContext(("r",), "walk", 0, False)
        ctx.set_padded_rids(("r", *pads))
        self.assertTrue(kv.admit(step, ctx).ok)
        named = sum(len(v.page_idxs) for v in kv._sequence_views(list(step.segments), ctx=ctx))
        num_tokens = sum(s.span for s in step.segments)
        self.assertLessEqual(named, captured_page_capacity(len(step.segments), num_tokens, cfg, cfg.groups[0]))

    def test_a_window_pool_allots_every_group(self):
        """A request is allotted its prompt's pages in every group, so one that fits only if the
        window's pages went uncounted waits; once the prompt is written, the window's share drops
        to the pages it holds, and the waiting request fits."""
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.engine.resources.kv import manager as manager_mod
        from mstar.engine.resources.kv.manager import KVManager, RoomMark
        from mstar.engine.resources.kv.transfer import TransferEngineInfo

        # 17 arena pages for requests; a 48-token prompt takes 3 of the full layer and 6 of the window
        cfg = window_pool(num_qo_heads=1, num_kv_heads=1, head_dim=4)
        layout = StreamLayout(spans=(PrefixSpan(48, 48, "walk"),), growth=(("loop", 1),))
        self.assertEqual(cfg.request_pages, 17)
        self.assertEqual(KVReqConfig(kv_layout={"main": layout}).pages(None, {"loop": 0}, cfg), 9)
        # the window's pages peak at its budget plus the prompt's step, while the full layer's grow
        self.assertEqual(KVReqConfig(kv_layout={"main": layout}).pages(None, {"loop": 32}, cfg), 5 + 8)
        self.enterContext(patch.object(manager_mod, "_DEBUG_ASSERTS", True))
        kv = KVManager(cfg=cfg, name="kv", joint_comm_group=None,
                       transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
                       device=torch.device("cpu"), dtype=torch.float32)
        for rid in ("a", "b"):
            kv.ingest_request(rid, KVReqConfig(kv_layout={"main": layout}, loop_caps={"loop": 16}))
            kv.ask(rid, "LLM", "walk")
        self.assertIsNone(kv.reserve("a", "LLM", "walk", None))
        self.assertIsInstance(kv.reserve("b", "LLM", "walk", None), RoomMark)
        for span in (32, 16, *[1] * 16):
            step, ctx = KVStep(segments=(Segment("a", "main", span),)), StepContext(("a",), "walk", 0, False)
            self.assertTrue(kv.admit(step, ctx).ok)
            kv.plan(step, ctx)
            kv.commit(step, ctx)
            if span == 32:
                # the prompt's second chunk still needs the pages the window released
                self.assertIsInstance(kv.reserve("b", "LLM", "walk", None), RoomMark)
            elif span == 16:
                # a holds 3 pages of the full layer and 1 of the window
                self.assertEqual(kv._allotments["a"].pages, 4)
                self.assertIsNone(kv.reserve("b", "LLM", "walk", None))
        kv.remove_request("a")
        # an admit unwound gives back the pages it took in each group
        free = kv._arena.num_free
        step, ctx = KVStep(segments=(Segment("b", "main", 48),)), StepContext(("b",), "walk", 0, False)
        self.assertTrue(kv.admit(step, ctx).ok)
        self.assertEqual(free - kv._arena.num_free, 9)
        kv.rollback_admit(step, ctx)
        self.assertEqual(kv._arena.num_free, free)

    def test_capacity_benchmark_counts_what_admission_lets_in(self):
        """At one rank `benchmark/kv_capacity.py` stops at the request that waits for its allotment,
        as above, though the pages left free hold its prompt; above one rank nothing allots."""
        from benchmark import kv_capacity

        def requests(shards):
            cfg = window_pool(num_qo_heads=1, num_kv_heads=1, head_dim=4)
            return kv_capacity.fill(cfg, 64, shards=shards, decode=16, device="cpu")["requests"]

        self.assertEqual(requests(1), 1)
        self.assertGreater(requests(2), 1)

    def test_window_pool_refusals(self):
        # what depends on the device is refused as the attention builds: FlashInfer's fa3 is SM90's only
        with self.assertRaises(ValueError):
            build_manager(WINDOW, kv_config=window_pool(), layers=(1, 2), flashinfer_backend="trtllm-gen")
        ampere = SimpleNamespace(cuda_capability=lambda device=None: (8, 0))
        with patch("mstar.engine.resources.attn.base.current_platform", ampere), self.assertRaises(ValueError):
            build_manager(WINDOW, kv_config=window_pool(), layers=(1, 2), flashinfer_backend="fa3")
        # the backend as it resolves, whatever was declared when the pool resolved
        with self.assertRaisesRegex(ValueError, "flashinfer"):
            build_manager(None, kv_config=window_pool(), backend=AttnBackend.DENSE)
        # a window over a pool its specs never resolved would plan as if it kept every page
        with self.assertRaisesRegex(ValueError, "resolve"):
            build_manager(WINDOW, kv_config=geometry(), layers=(0,))
        # a window turns the prefix cache off, unless the serving config asked for it
        cfg = PagedKVConfig(num_layers=2, num_qo_heads=4, num_kv_heads=2, head_dim=128,
                            max_seq_len=64, page_size=16, max_num_pages=6)
        cfg.apply_yaml_overrides(prefix_cache=True)
        with self.assertRaisesRegex(ValueError, "prefix_cache"):
            resolve_spec_dependencies([
                KVSpec(resource_key="kv", nodes={"LLM"}, config=cfg),
                AttentionSpec(resource_key="local", nodes={"LLM"},
                              config=AttentionConfig("kv", sliding_window=WINDOW, layers=(1,))),
                AttentionSpec(resource_key="full", nodes={"LLM"}, config=AttentionConfig("kv")),
            ])
        for layers in ((), (1, 1)):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, "layers"):
                AttentionConfig("kv", sliding_window=WINDOW, layers=layers)
        with self.assertRaisesRegex(ValueError, "layers"):
            AttentionConfig("kv", layers=(1,))

        def attn(key, **config):
            return AttentionSpec(resource_key=key, nodes={"LLM"}, config=AttentionConfig("kv", **config))

        local, full = attn("local", sliding_window=WINDOW, layers=(1, 2)), attn("full")
        # the rest, before anything builds
        for name, readers, shape in (
            ("a layer declared twice", [local, full, attn("other", sliding_window=WINDOW, layers=(2,))], {}),
            ("a layer past the pool", [attn("local", sliding_window=WINDOW, layers=(1, 3)), full], {}),
            ("two windows", [local, full, attn("other", sliding_window=16, layers=(0,))], {}),
            ("a window of every layer", [attn("local", sliding_window=WINDOW, layers=(0, 1, 2)), full], {}),
            ("a window without layers", [attn("local", sliding_window=WINDOW), full], {}),
            ("full layers no attention reads", [local], {}),
            ("another backend", [local, attn("full", backend=AttnBackend.DENSE)], {}),
            ("cross attention", [local, full, CrossAttentionSpec(resource_key="cross", nodes={"LLM"},
                                                                 config=CrossAttentionConfig("kv", "kv"))], {}),
            ("an MLA pool", [local, full], dict(layout=KVLayout.MLA, kv_lora_rank=64, qk_rope_head_dim=64)),
            ("a window alone over an MLA pool", [attn("local", sliding_window=WINDOW)],
             dict(layout=KVLayout.MLA, kv_lora_rank=64, qk_rope_head_dim=64)),
        ):
            cfg = PagedKVConfig(**{**dict(num_layers=3, num_kv_heads=2, head_dim=128, max_seq_len=64,
                                          page_size=16), **shape})
            specs = {"kv": KVSpec(resource_key="kv", nodes={"LLM"}, config=cfg)} | {r.resource_key: r for r in readers}
            with self.subTest(name):
                self.assertTrue(derive_kv_groups("kv", cfg, specs))
                self.assertEqual(len(cfg.groups), 1)

    def test_a_window_as_long_as_the_context_keeps_every_token(self):
        cfg = PagedKVConfig(num_layers=3, num_kv_heads=2, head_dim=128, max_seq_len=1024)
        self.assertEqual([], derive_kv_groups("kv", cfg, {
            "kv": KVSpec(resource_key="kv", nodes={"LLM"}, config=cfg),
            "local": AttentionSpec(resource_key="local", nodes={"LLM"},
                                   config=AttentionConfig("kv", sliding_window=1024)),
        }))
        self.assertEqual(cfg.groups, (KVGroup((0, 1, 2), DEFAULT_PAGE_SIZE),))

    def test_default_auto_plans_fa2_from_sm100(self):
        """From SM100 FlashInfer's auto can send short rows to cuDNN, and its fa3 is SM90's only."""
        blackwell = SimpleNamespace(cuda_capability=lambda device=None: (10, 0))
        with patch("mstar.engine.resources.attn.base.current_platform", blackwell):
            ctx = context([7, 5], [7, 5])
            plans = KVPlanOutputs(ctx.plan_results["kv"], later_groups=(ctx.plan_results["kv"],))
            for window in (None, WINDOW):
                with self.subTest(window=window):
                    manager = build_manager(window, kv_config=window_pool(), layers=(1, 2) if window else None)
                    manager.plan(AttentionStep(), StepContext(
                        ("r0", "r1"), "prefill", 0, False, plan_results={"kv": plans},
                    ))
                    self.assertEqual(self.fake.BatchPrefillWithPagedKVCacheWrapper.call_args.kwargs["backend"], "fa2")
            with self.assertRaises(ValueError):
                build_manager(WINDOW, kv_config=window_pool(), layers=(1, 2), flashinfer_backend="fa3")

    def test_a_layer_must_attend_through_its_group(self):
        """Over a pool of two groups an attention runs the layers it declares, or without layers
        the first group's: through the cursor, or `run(layer_idx=)`, which takes precedence;
        with neither it can't tell its layer."""
        q = torch.zeros(1, 4, 128)
        for window, refused in ((None, (1, 2)), (WINDOW, (0,))):
            manager = build_manager(window, kv_config=window_pool(), layers=(1, 2) if window else None)
            manager._current_plan_states["main"] = MagicMock()
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "layer it runs"):
                manager.run(q, "main", None)
            for layer in range(3):
                with self.subTest(window=window, layer=layer):
                    if layer in refused:
                        with self.assertRaises(ValueError):
                            manager.set_default_layer_idx(layer)
                        manager.set_default_layer_idx(1 if window else 0)
                        with self.assertRaises(ValueError):
                            manager.run(q, "main", None, layer_idx=layer)
                    else:
                        manager.set_default_layer_idx(layer)
                        manager.run(q, "main", None, layer_idx=layer)
        # an explicit layer is checked without moving the stack's cursor
        manager.set_default_layer_idx(1)
        manager.run(q, "main", None, layer_idx=2)
        self.assertEqual(manager._default_layer_idx, 1)


class SlidingWindowGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or importlib.util.find_spec("flashinfer") is None:
            raise unittest.SkipTest("CUDA and FlashInfer are required")

    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"MSTAR_WORKSPACE_BUFFER_MB": "32"}))
        generator = torch.Generator(device="cuda").manual_seed(7)
        self.q = torch.randn(2, 24, 4, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
        self.k = torch.randn(2, 24, 2, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
        self.v = torch.randn(2, 24, 2, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
        self.cache = torch.zeros(6, 2, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
        for request, pages in enumerate(([3, 0], [4, 1])):
            for pos in range(24):
                self.cache[pages[pos // 16], 0, pos % 16] = self.k[request, pos]
                self.cache[pages[pos // 16], 1, pos % 16] = self.v[request, pos]

    def reference(self, lengths, query_lengths, window, scale=None):
        scale = 128 ** -0.5 if scale is None else scale
        results = []
        for request, (n, nq) in enumerate(zip(lengths, query_lengths, strict=True)):
            q = self.q[request, n - nq:n].float().transpose(0, 1)
            k = self.k[request, :n].float().repeat_interleave(2, dim=1).transpose(0, 1)
            v = self.v[request, :n].float().repeat_interleave(2, dim=1).transpose(0, 1)
            scores = q @ k.transpose(-1, -2) * scale
            qpos = torch.arange(n - nq, n, device="cuda")[:, None]
            kpos = torch.arange(n, device="cuda")[None, :]
            allowed = kpos <= qpos
            if window is not None:
                allowed &= kpos > qpos - window
            results.append((scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v).transpose(0, 1))
        return torch.cat(results)

    def packed_queries(self, lengths, query_lengths):
        return torch.cat([self.q[r, n - nq:n] for r, (n, nq) in enumerate(zip(lengths, query_lengths, strict=True))])

    def test_prefill_and_four_decode_steps(self):
        # The second case prompts over a cached prefix, so a window starts mid-page, under a
        # scale; its short row decodes from exactly the window's width to past it.
        cases = (([19, 9], [19, 9], None), ([19, 3], [5, 2], 3.87 / 128 ** 0.5))
        for prompt, prompt_queries, scale in cases:
            managers = {window: build_manager(window, "cuda", sm_scale=scale) for window in (None, 1, 4)}
            for increment in range(5):
                lengths = [n + increment for n in prompt]
                queries = prompt_queries if increment == 0 else [1, 1]
                ctx = context(lengths, queries)
                q = self.packed_queries(lengths, queries)
                for window, manager in managers.items():
                    with self.subTest(scale=scale, window=window, step=increment):
                        manager.plan(AttentionStep(), ctx)
                        actual = manager.run(q, "main", self.cache)
                        torch.testing.assert_close(
                            actual.float(), self.reference(lengths, queries, window, scale),
                            atol=0.015, rtol=0.015,
                        )
            # At the final decode positions, key zero is outside either local window.
            q = self.packed_queries(lengths, [1, 1])
            before = {w: m.run(q, "main", self.cache).clone() for w, m in managers.items()}
            poisoned = self.cache.clone()
            poisoned[[3, 4], :, 0] = 100
            for window in (1, 4):
                torch.testing.assert_close(managers[window].run(q, "main", poisoned), before[window], atol=0, rtol=0)
            self.assertFalse(torch.equal(managers[None].run(q, "main", poisoned), before[None]))

    def test_mixed_step_with_window_and_scale(self):
        # a decode row beside a prompt over a cached prefix, padded to a bucket's tokens
        lengths, queries, scale = [20, 13], [1, 9], 3.87 / 128 ** 0.5
        indptrs = context(lengths, queries).plan_results["kv"]["main"].cpu_indptrs
        real = sum(queries)
        workspace = lambda: torch.empty(32 << 20, dtype=torch.uint8, device="cuda")  # noqa: E731
        mixed = wrappers.FlashInferMixedWrapper(
            decode_workspace=workspace(), prefill_workspace=workspace(), num_qo_heads=4,
            num_kv_heads=2, head_dim=128, page_size=16, batch_size=2, max_total_tokens=real + 7,
            max_num_pages=16, max_prefill_rows=1, device=torch.device("cuda"), sm_scale=scale,
        )
        mixed.plan(
            causal=True, dtype=torch.bfloat16, window_left=3, qo_indptr=indptrs.qo_indptr,
            paged_kv_indptr=indptrs.paged_kv_indptr, paged_kv_indices=indptrs.paged_kv_indices,
            paged_kv_last_page_len=indptrs.paged_kv_last_page_len, kv_lens=indptrs.kv_lens,
        )
        q = torch.zeros(real + 7, 4, 128, device="cuda", dtype=torch.bfloat16)
        q[:real] = self.packed_queries(lengths, queries)
        torch.testing.assert_close(
            mixed.run(q, self.cache)[:real].float(), self.reference(lengths, queries, 4, scale),
            atol=0.015, rtol=0.015,
        )

    def test_cuda_graph_prefill_and_decode_replanning(self):
        manager = build_manager(4, "cuda")
        # Capture and replay both wrapper kinds; decode replans with a longer cache.
        for lengths, queries in (([19, 9], [19, 9]), ([20, 10], [1, 1])):
            ctx = context(lengths, queries, captured=True)
            manager.plan(AttentionStep(), ctx)
            q = self.packed_queries(lengths, queries)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    manager.run(q, "main", self.cache)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = manager.run(q, "main", self.cache)
            graph.replay()
            torch.testing.assert_close(output.float(), self.reference(lengths, queries, 4), atol=0.015, rtol=0.015)
            if queries == [1, 1]:
                lengths = [21, 11]
                manager.plan(AttentionStep(), context(lengths, queries, captured=True))
                q.copy_(self.packed_queries(lengths, queries))
                graph.replay()
                torch.testing.assert_close(output.float(), self.reference(lengths, queries, 4), atol=0.015, rtol=0.015)


class Pool(NamedTuple):
    num_layers: int
    page_size: int
    sliding: tuple[int, ...]
    max_num_pages: int = 16
    head_dim: int = 128
    window: int = 100

    def specs(self, grouped, backend="auto", splits=()):
        """The pool and its attentions, resolved. With `grouped` the windowed attention declares
        its window and layers, so the pool keeps only their window; without, the attentions
        declare no window, so the pool keeps every token (`_Stack` masks to the window)."""
        cfg = PagedKVConfig(num_layers=self.num_layers, num_qo_heads=4, num_kv_heads=2,
                            head_dim=self.head_dim, max_seq_len=1024, page_size=self.page_size,
                            max_num_pages=self.max_num_pages)
        return resolve_spec_dependencies([KVSpec(resource_key="kv", nodes={"LLM"}, config=cfg)] + [
            AttentionSpec(resource_key=f"attn{window}", nodes={"LLM"}, config=AttentionConfig(
                "kv", sliding_window=window if grouped else None, flashinfer_backend=backend, splits=splits,
                layers=self.sliding if grouped and window else None,
            ))
            for window in self.windows()
        ])

    def windows(self):
        return {self.window_of(layer) for layer in range(self.num_layers)}

    def window_of(self, layer):
        return self.window if layer in self.sliding else None


POOLS = {
    # layer 0 full, the rest sliding: 192-token full pages, 64-token sliding ones
    "192/64": Pool(num_layers=4, page_size=192, sliding=(1, 2, 3)),
    # 8 full and 23 sliding layers, 184/64
    "184/64": Pool(num_layers=31, page_size=184, sliding=tuple(range(8, 31))),
    # Muse Spark's shape at head_dim 64, with a window of whole 64-token sliding pages
    "184/64 hd64 w128": Pool(num_layers=31, page_size=184, sliding=tuple(range(8, 31)), head_dim=64, window=128),
}


class _Stack:
    """A KV pool and its attentions, driven through admit/plan/write/attend/commit
    the way a decoder layer stack drives them, for two requests."""

    def __init__(self, pool, grouped=True, backend="auto", splits=()):
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.engine.resources.kv.manager import KVManager
        from mstar.engine.resources.kv.transfer import TransferEngineInfo

        specs = pool.specs(grouped, backend, splits)
        cfg = specs["kv"].config
        self.kv = KVManager(
            cfg=cfg, name="kv", joint_comm_group=None,
            transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
            device=torch.device("cuda"), dtype=torch.bfloat16,
        )
        info = EngineResourceInfo(
            device=torch.device("cuda"), kv_dtype=torch.bfloat16, dependencies={"kv": specs["kv"]},
        )
        self.pool = pool
        if grouped:
            self.attn = {
                spec.config.sliding_window: AttentionManager.build(spec, info)
                for spec in specs.values() if isinstance(spec, AttentionSpec)
            }
        else:
            # the window as a mask alone, over a pool that keeps every token
            self.attn = {
                window: FlashInferManager(
                    kv_cache="kv", device=info.device, dtype=info.kv_dtype, kv_config=cfg,
                    backend=backend, splits=splits, sliding_window=window,
                )
                for window in pool.windows()
            }
        self.rids = ("a", "b")
        for rid in self.rids:
            self.kv.ingest_request(rid)

    def plan(self, spans, lease=None):
        step = KVStep(segments=tuple(Segment(r, "main", n) for r, n in zip(self.rids, spans, strict=True)))
        ctx = StepContext(self.rids, lease.bucket.graph_walk if lease else "walk", 0, False, slot_lease=lease)
        assert self.kv.admit(step, ctx).ok
        ctx.plan_results["kv"] = self.kv.plan(step, ctx)
        for attn in self.attn.values():
            attn.plan(AttentionStep(), ctx)
        return step, ctx

    def forward(self, q, k, v):
        """q/k/v: [layers, tokens, heads, dim]. Returns [layers, tokens, heads, dim]."""
        out = []
        for layer in range(q.shape[0]):
            self.kv.write_kv(k[layer], v[layer], layer_idx=layer, label="main")
            attn = self.attn[self.pool.window_of(layer)]
            out.append(attn.run(q[layer], "main", self.kv.layer_view(layer), layer_idx=layer))
        return torch.stack(out)

    def released(self):
        """Per request, the tokens its window has released so far."""
        return [self.kv._streams[rid]["main"].groups[1].released for rid in self.rids]


class WindowPoolGPU(unittest.TestCase):
    """A pool that keeps only the window for its sliding layers, end to end on FlashInfer."""

    PREFILL = (300, 520)
    DECODES = 70

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or importlib.util.find_spec("flashinfer") is None:
            raise unittest.SkipTest("CUDA and FlashInfer are required")

    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"MSTAR_WORKSPACE_BUFFER_MB": "128"}))

    def tokens(self, pool, total=None):
        generator = torch.Generator(device="cuda").manual_seed(11)
        total = total or max(self.PREFILL) + self.DECODES

        def randn(heads):
            return torch.randn(2, pool.num_layers, total, heads, pool.head_dim, generator=generator,
                               device="cuda", dtype=torch.bfloat16)
        return randn(4), randn(2), randn(2)

    @staticmethod
    def packed(t, starts, spans):
        return torch.cat([t[r, :, s:s + n] for r, (s, n) in enumerate(zip(starts, spans, strict=True))], dim=1)

    @staticmethod
    def reference(q, k, v, starts, spans, layer, window):
        out = []
        for r, (s, n) in enumerate(zip(starts, spans, strict=True)):
            qr = q[r, layer, s:s + n].float().transpose(0, 1)
            kr = k[r, layer, :s + n].float().repeat_interleave(2, dim=1).transpose(0, 1)
            vr = v[r, layer, :s + n].float().repeat_interleave(2, dim=1).transpose(0, 1)
            scores = qr @ kr.transpose(-1, -2) / qr.shape[-1] ** 0.5
            qpos = torch.arange(s, s + n, device="cuda")[:, None]
            kpos = torch.arange(s + n, device="cuda")[None, :]
            allowed = kpos <= qpos
            if window is not None:
                allowed &= kpos > qpos - window
            out.append((scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ vr).transpose(0, 1))
        return torch.cat(out)

    def run_eager(self, pool, grouped, backend):
        stack = _Stack(pool, grouped, backend)
        q, k, v = self.tokens(pool)
        starts, outputs, spans = [0, 0], [], list(self.PREFILL)
        for _ in range(self.DECODES + 1):
            step, ctx = stack.plan(spans)
            out = stack.forward(*(self.packed(t, starts, spans) for t in (q, k, v)))
            stack.kv.commit(step, ctx)
            outputs.append((list(starts), list(spans), out))
            starts = [s + n for s, n in zip(starts, spans, strict=True)]
            spans = [1, 1]
        return stack, (q, k, v), outputs

    def test_windowed_pool_matches_the_unwindowed_one_and_the_reference(self):
        from flashinfer.utils import determine_attention_backend

        for (name, pool), backend in itertools.product(POOLS.items(), ("auto", "fa2")):
            with self.subTest(pool=name, backend=backend):
                grouped, tensors, got = self.run_eager(pool, True, backend)
                _, _, want = self.run_eager(pool, False, backend)
                # the sliding layers hold about the window, the full ones every token
                kept = grouped.kv.config.groups[-1]
                for rid in grouped.rids:
                    stream = grouped.kv._streams[rid]["main"]
                    held = stream.groups[1]
                    self.assertEqual(stream.stored_len + stream.released, held.stored_len + held.released)
                    self.assertGreater(held.released, 0)
                    self.assertLess(held.stored_len, kept.retention.context_budget + kept.page_size)
                grouped.kv.assert_pages_conserved()
                # The window's pages hold the keys the mask lets through: bit for bit under
                # FA3, within a bf16 rounding under FA2, whose prefill splits the KV in chunks
                # of pages, so the sliding layers' smaller pages sum in another order.
                fa3 = backend == "auto" and determine_attention_backend(
                    torch.device("cuda"), 0, False, False, torch.bfloat16, torch.bfloat16,
                ) == "fa3"
                tol = 0 if fa3 else 2**-7
                for (starts, spans, a), (_, _, b) in zip(got, want, strict=True):
                    torch.testing.assert_close(a, b, atol=tol, rtol=tol)
                    for layer in (0, pool.sliding[0], pool.sliding[-1]):
                        torch.testing.assert_close(
                            a[layer].float(), self.reference(*tensors, starts, spans, layer, pool.window_of(layer)),
                            atol=2e-2, rtol=2e-2,
                        )

    def test_captured_decode_replays_over_the_window(self):
        for (name, pool), backend in itertools.product(POOLS.items(), ("auto", "fa2")):
            with self.subTest(pool=name, backend=backend):
                self.capture_and_replay(pool, backend)

    def capture_and_replay(self, pool, backend):
        stack = _Stack(pool, backend=backend)
        stack.kv.build_accelerator_graph_buffers([], 2, 64)
        q, k, v = self.tokens(pool)
        # prefill eagerly, then capture one decode step and replay it for the rest
        step, ctx = stack.plan(list(self.PREFILL))
        stack.forward(*(self.packed(t, [0, 0], self.PREFILL) for t in (q, k, v)))
        stack.kv.commit(step, ctx)
        prefilled = stack.released()
        starts = list(self.PREFILL)
        lease = SlotLease(0, BucketKey("decode", 2, 2))
        buffers = [self.packed(t, starts, [1, 1]).clone() for t in (q, k, v)]
        graph = None
        for _ in range(self.DECODES):
            step, ctx = stack.plan([1, 1], lease)
            for buf, t in zip(buffers, (q, k, v), strict=True):
                buf.copy_(self.packed(t, starts, [1, 1]))
            if graph is None:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    stack.forward(*buffers)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = stack.forward(*buffers)
            graph.replay()
            for layer in (0, pool.sliding[-1]):
                torch.testing.assert_close(
                    output[layer].float(), self.reference(q, k, v, starts, [1, 1], layer, pool.window_of(layer)),
                    atol=2e-2, rtol=2e-2,
                )
            stack.kv.commit(step, ctx)
            starts = [s + 1 for s in starts]
        # the replayed steps released pages of their own, not only the prefill's
        for after, before in zip(stack.released(), prefilled, strict=True):
            self.assertGreater(after, before)

    def test_captured_mixed_steps_match_eager_past_the_window(self):
        """Steps of a decode row and a prompt chunk, replayed on one captured mixed graph
        (`FlashInferMixedWrapper`) over the window's plans, match the same steps run eagerly."""
        self.check_captured_steps("mixed", (self.CHUNK, 1))

    def test_captured_prefill_chunks_match_eager_past_the_window(self):
        """Steps where both rows take a chunk of their prompt, replayed on one captured
        prefill graph (`FlashInferPrefillWrapper`) over the window's plans, match the same
        steps run eagerly."""
        self.check_captured_steps("prefill", (self.CHUNK // 2, 3 * self.CHUNK // 2))

    STEPS, CHUNK = 12, 16

    def check_captured_steps(self, walk, spans):
        for (name, pool), backend in itertools.product(POOLS.items(), ("auto", "fa2")):
            with self.subTest(pool=name, backend=backend):
                eager, tensors = self.run_steps(pool, backend, walk, spans, captured=False)
                captured, _ = self.run_steps(pool, backend, walk, spans, captured=True)
                for (starts, step_spans, a), (_, _, b) in zip(captured, eager, strict=True):
                    torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2)
                    for layer in (0, pool.sliding[-1]):
                        torch.testing.assert_close(
                            a[layer].float(),
                            self.reference(*tensors, starts, step_spans, layer, pool.window_of(layer)),
                            atol=2e-2, rtol=2e-2,
                        )

    def run_steps(self, pool, backend, walk, spans, captured):
        """Prefill eagerly, then STEPS steps where the rows take `spans` tokens, trading
        places each step: one bucket of sum(spans) tokens in `walk`."""
        stack = _Stack(pool, backend=backend, splits=(CombinedSplit("mixed", prefill_walks=("prefill",)),))
        stack.kv.build_accelerator_graph_buffers([], 2, 64)
        tensors = self.tokens(pool, max(self.PREFILL) + self.STEPS * max(spans))
        step, ctx = stack.plan(list(self.PREFILL))
        stack.forward(*(self.packed(t, [0, 0], self.PREFILL) for t in tensors))
        stack.kv.commit(step, ctx)
        prefilled = stack.released()
        starts, outputs, graph = list(self.PREFILL), [], None
        lease = SlotLease(0, BucketKey(walk, 2, sum(spans))) if captured else None
        for i in range(self.STEPS):
            step_spans = list(spans[::-1] if i % 2 else spans)
            # The engine replans a slot only after the step that last held it has run.
            # Replanning sooner restages host buffers that step's copies may still read.
            torch.cuda.synchronize()
            step, ctx = stack.plan(step_spans, lease)
            inputs = [self.packed(t, starts, step_spans) for t in tensors]
            if not captured:
                output = stack.forward(*inputs)
            elif graph is None:
                buffers = [x.clone() for x in inputs]
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    stack.forward(*buffers)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = stack.forward(*buffers)
            if captured:
                for buf, x in zip(buffers, inputs, strict=True):
                    buf.copy_(x)
                graph.replay()
            outputs.append((list(starts), step_spans, output.clone()))
            stack.kv.commit(step, ctx)
            starts = [s + n for s, n in zip(starts, step_spans, strict=True)]
        # the steps past the window released pages of their own, not only the prefill's
        for after, before in zip(stack.released(), prefilled, strict=True):
            self.assertGreater(after, before)
        return outputs, tensors


if __name__ == "__main__":
    unittest.main()
