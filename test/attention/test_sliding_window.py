"""Window plumbing on CPU, plus opt-in single-GPU numerical and replay tests.

Run with unittest discover -s test/attention. Set MSTAR_TEST_CUDA=1 to also
exercise FlashInfer against an explicit FP32 PyTorch attention reference.
"""

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep, AttnBackend
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.kv.config import KVConfig, KVSpec, KVStep
from mstar.engine.resources.kv.plan import KVPlanOutput, SequenceView, build_paged_indptrs
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext, SubmoduleStep


def geometry():
    return KVConfig(num_layers=1, num_qo_heads=4, num_kv_heads=2, head_dim=128,
                    max_seq_len=64, page_size=16, max_num_pages=6)


def build_manager(window, device="cpu", backend=AttnBackend.FLASHINFER):
    kv = KVSpec(resource_key="kv", nodes={"LLM"}, config=geometry())
    spec = AttentionSpec(resource_key="attn", nodes={"LLM"},
                         config=AttentionConfig("kv", backend=backend, sliding_window=window))
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
        for backend in (AttnBackend.DENSE, AttnBackend.XPU_PAGED):
            spec = AttentionSpec(resource_key="attn", nodes={"LLM"},
                                 config=AttentionConfig("kv", backend=backend, sliding_window=4))
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, backend.name):
                AttentionManager.build(spec, EngineResourceInfo(device=torch.device("cpu")))

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

    def test_both_wrappers_receive_window_and_reuse_plans(self):
        for window, expected in ((None, -1), (1, 0), (4, 3)):
            for captured in (False, True):
                for query_lengths in ([7, 5], [1, 1]):
                    with self.subTest(window=window, captured=captured, queries=query_lengths):
                        manager = build_manager(window)
                        ctx = context([7, 5], query_lengths, captured)
                        manager.plan(AttentionStep(), ctx)
                        wrapper = manager._current_plan_states["main"]
                        self.assertEqual(wrapper.attn_wrapper.plan.call_args.kwargs["window_left"], expected)
                        self.assertEqual(wrapper.use_cuda_graph, captured)
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


@unittest.skipUnless(os.environ.get("MSTAR_TEST_CUDA") == "1", "set MSTAR_TEST_CUDA=1 for GPU tests")
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

    def reference(self, lengths, query_lengths, window):
        results = []
        for request, (n, nq) in enumerate(zip(lengths, query_lengths, strict=True)):
            q = self.q[request, n - nq:n].float().transpose(0, 1)
            k = self.k[request, :n].float().repeat_interleave(2, dim=1).transpose(0, 1)
            v = self.v[request, :n].float().repeat_interleave(2, dim=1).transpose(0, 1)
            scores = q @ k.transpose(-1, -2) / (128 ** 0.5)
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
        managers = {window: build_manager(window, "cuda") for window in (None, 1, 4)}
        for increment in range(5):
            lengths = [19 + increment, 9 + increment]
            queries = lengths if increment == 0 else [1, 1]
            ctx = context(lengths, queries)
            q = self.packed_queries(lengths, queries)
            for window, manager in managers.items():
                with self.subTest(window=window, step=increment):
                    manager.plan(AttentionStep(), ctx)
                    actual = manager.run(q, "main", self.cache)
                    torch.testing.assert_close(actual.float(), self.reference(lengths, queries, window), atol=0.015, rtol=0.015)
        # At the final decode positions, key zero is outside either local window.
        q = self.packed_queries([23, 13], [1, 1])
        before = {w: m.run(q, "main", self.cache).clone() for w, m in managers.items()}
        for page in (3, 4):
            self.cache[page, :, 0] = 100
        for window in (1, 4):
            torch.testing.assert_close(managers[window].run(q, "main", self.cache), before[window], atol=0, rtol=0)
        self.assertFalse(torch.equal(managers[None].run(q, "main", self.cache), before[None]))

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


if __name__ == "__main__":
    unittest.main()
