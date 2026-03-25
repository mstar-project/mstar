"""
GraphWalk 1 tests: Engine execution + Mooncake integration.
Tests engines in dummy mode (model=None, no GPU required) and verifies
the interleaved LLM<->flow loop fires nodes in the correct order.
"""
import sys

sys.path.insert(0, ".")


from mminf.engine.kv_store import PageAllocator
import pytest

from mminf.engine.ar_engine import AREngine
from mminf.engine.base import EngineType, NodeBatch, NodeOutput
from mminf.engine.enc_dec_engine import EncoderDecoderEngine
from mminf.engine.flow_engine import FlowEngine
from mminf.graph.base import GraphEdge
from mminf.graph.request_queues import PerRequestNodeQueues
from mminf.model.base import WorkerGraph
from mminf.model.dummy_model import DummyModel
from mminf.worker.engine_manager import EngineManager
from mminf.worker.micro_scheduler import MicroScheduler
from mminf.worker.node_manager_utils import (
    WorkerGraphQueues,
    WorkerGraphsManager,
)

# ======================================================================
# PageAllocator tests
# ======================================================================


class TestPageAllocator:
    def test_allocate_and_free(self):
        alloc = PageAllocator(max_num_pages=10)
        assert alloc.num_free == 10

        pages = alloc.allocate(3)
        assert len(pages) == 3
        assert alloc.num_free == 7
        # Pages should be unique
        assert len(set(pages)) == 3

        alloc.free(pages)
        assert alloc.num_free == 10

    def test_allocate_all_pages(self):
        alloc = PageAllocator(max_num_pages=5)
        pages = alloc.allocate(5)
        assert len(pages) == 5
        assert alloc.num_free == 0

    def test_exhaustion_raises(self):
        alloc = PageAllocator(max_num_pages=3)
        alloc.allocate(3)
        with pytest.raises(RuntimeError, match="Not enough free pages"):
            alloc.allocate(1)

    def test_free_then_reallocate(self):
        alloc = PageAllocator(max_num_pages=4)
        p1 = alloc.allocate(4)
        alloc.free(p1[:2])
        assert alloc.num_free == 2
        p2 = alloc.allocate(2)
        assert len(p2) == 2
        assert alloc.num_free == 0

    def test_allocate_zero(self):
        alloc = PageAllocator(max_num_pages=5)
        pages = alloc.allocate(0)
        assert pages == []
        assert alloc.num_free == 5


# ======================================================================
# Engine tests (dummy mode — model=None)
# ======================================================================


class TestEngines:
    def _make_batch(self, node_name: str, request_ids: list[str]) -> NodeBatch:
        return NodeBatch(
            node_name=node_name,
            graph_walk="test",
            request_ids=request_ids,
            per_request_input_tensors={rid: {} for rid in request_ids},
        )

    def test_ar_engine_type(self):
        engine = AREngine()
        assert engine.engine_type() == EngineType.AR

    def test_ar_engine_dummy_execute(self):
        engine = AREngine()
        batch = self._make_batch("LLM", ["req1", "req2"])
        output = engine.execute_batch(batch)
        assert isinstance(output, NodeOutput)
        assert "req1" in output.per_request_output_tensors
        assert "req2" in output.per_request_output_tensors

    def test_ar_engine_add_remove_request(self):
        engine = AREngine()
        engine.add_request("req1")
        assert ("req1", "main") in engine.request_states
        assert engine.request_states[("req1", "main")].seq_len == 0
        assert engine.request_states[("req1", "main")].page_indices == []

        engine.remove_request("req1")
        assert ("req1", "main") not in engine.request_states

    def test_ar_engine_pause_resume(self):
        engine = AREngine()
        engine.add_request("req1")
        assert not engine.request_states[("req1", "main")].is_paused

        engine.pause_request("req1")
        assert engine.request_states[("req1", "main")].is_paused

        engine.resume_request("req1")
        assert not engine.request_states[("req1", "main")].is_paused

    def test_ar_engine_remove_nonexistent(self):
        engine = AREngine()
        # Should not raise
        engine.remove_request("nonexistent")

    def test_flow_engine_type(self):
        engine = FlowEngine()
        assert engine.engine_type() == EngineType.FLOW

    def test_flow_engine_dummy_execute(self):
        engine = FlowEngine()
        batch = self._make_batch("flow", ["req1"])
        output = engine.execute_batch(batch)
        assert isinstance(output, NodeOutput)
        assert "req1" in output.per_request_output_tensors

    def test_enc_dec_engine_type(self):
        engine = EncoderDecoderEngine()
        assert engine.engine_type() == EngineType.ENC_DEC

    def test_enc_dec_engine_dummy_execute(self):
        engine = EncoderDecoderEngine()
        batch = self._make_batch("text_emb", ["req1", "req2", "req3"])
        output = engine.execute_batch(batch)
        assert isinstance(output, NodeOutput)
        assert len(output.per_request_output_tensors) == 3


# ======================================================================
# EngineManager tests
# ======================================================================


class TestEngineManager:
    def test_from_config_dummy(self):
        configs = [
            {"engine_type": "ar", "node_names": ["LLM"], "model_config": {}},
            {"engine_type": "flow", "node_names": ["flow"], "model_config": {}},
            {
                "engine_type": "enc_dec",
                "node_names": ["text_emb", "image_emb", "VAE_dec"],
                "model_config": {},
            },
        ]
        mgr = EngineManager.from_config(configs, device="cpu")
        assert mgr.get_engine("LLM").engine_type() == EngineType.AR
        assert mgr.get_engine("flow").engine_type() == EngineType.FLOW
        assert mgr.get_engine("text_emb").engine_type() == EngineType.ENC_DEC
        assert mgr.get_engine("VAE_dec").engine_type() == EngineType.ENC_DEC
        # text_emb and image_emb share the same engine instance
        assert mgr.get_engine("text_emb") is mgr.get_engine("image_emb")

    def test_add_remove_request_propagation(self):
        configs = [
            {"engine_type": "ar", "node_names": ["LLM"], "model_config": {}},
            {"engine_type": "flow", "node_names": ["flow"], "model_config": {}},
        ]
        mgr = EngineManager.from_config(configs, device="cpu")
        mgr.add_request("req1")
        ar_engine = mgr.get_engine("LLM")
        assert isinstance(ar_engine, AREngine)
        assert ("req1", "main") in ar_engine.request_states

        mgr.remove_request("req1")
        assert ("req1", "main") not in ar_engine.request_states


# ======================================================================
# Image generation loop integration test
# ======================================================================


class TestImageGenLoop:
    """
    Uses DummyModel's image_gen graph walk to verify the interleaved
    LLM<->flow loop fires nodes in the correct order.

    The image_gen graph structure:
    Sequential([
        Parallel([text_emb_section, img_emb_section]),
        Loop(
            Sequential([LLM, flow]),
            n_iters=10,
            outputs=[latents -> VAE_dec]
        ),
        VAE_dec
    ])
    """

    def _build_image_gen_graph(self):
        model = DummyModel()
        graphs = model.get_graph_walk_graphs()
        return graphs["image_gen"]

    def test_loop_node_order(self):
        """Verify nodes fire in the correct order through the loop."""
        graph = self._build_image_gen_graph()
        queues = PerRequestNodeQueues(waiting=graph)

        # Provide all initial external inputs
        initial_inputs = [
            GraphEdge(name="text", next_node="text_emb"),
            GraphEdge(name="images", next_node="image_emb"),
            GraphEdge(name="existing_text_emb", next_node="concat_text"),
            GraphEdge(name="existing_image_emb", next_node="concat_img"),
            GraphEdge(name="latents", next_node="LLM"),
        ]
        queues.process_new_inputs(initial_inputs)

        fired_nodes = []
        max_iterations = 100  # safety bound

        for _ in range(max_iterations):
            if not queues.ready and queues.waiting is None:
                break
            assert queues.ready, "Deadlock: no ready nodes but waiting nodes remain"

            # Pop and process one node at a time
            node = queues.ready.pop(0)
            fired_nodes.append(node.name)
            queues.process_new_inputs(node.outputs)

        # Expected order:
        # 1. text_emb and image_emb fire (parallel, order may vary)
        # 2. concat_text and concat_img fire (parallel, order may vary)
        # 3. 10 iterations of LLM then flow
        # 4. VAE_dec fires last
        assert fired_nodes[-1] == "VAE_dec"

        # Count LLM and flow occurrences — should be 10 each
        llm_count = fired_nodes.count("LLM")
        flow_count = fired_nodes.count("flow")
        assert llm_count == 10, f"Expected 10 LLM nodes, got {llm_count}"
        assert flow_count == 10, f"Expected 10 flow nodes, got {flow_count}"

        # Verify LLM always fires before flow within each iteration
        llm_indices = [i for i, s in enumerate(fired_nodes) if s == "LLM"]
        flow_indices = [i for i, s in enumerate(fired_nodes) if s == "flow"]
        for llm_idx, flow_idx in zip(llm_indices, flow_indices, strict=True):
            assert llm_idx < flow_idx, (
                f"LLM (index {llm_idx}) should fire before flow (index {flow_idx})"
            )

        # Verify VAE_dec fires only once
        assert fired_nodes.count("VAE_dec") == 1

    def test_loop_with_worker_graph_manager(self):
        """
        Test the full loop using WorkerGraphsManager (single-worker scenario).
        All nodes on one worker, verifying queue management works end-to-end.
        """
        graph = self._build_image_gen_graph()

        worker_graph_id = "sg_image_gen"
        worker_graph = WorkerGraph(
            section=graph,
            graph_walks={"image_gen"},
            worker_graph_id=worker_graph_id,
        )

        manager = WorkerGraphsManager(
            queues={
                worker_graph_id: WorkerGraphQueues(
                    worker_graph_id=worker_graph_id,
                    graph_walks={"image_gen"},
                    worker_graph=worker_graph,
                    per_request_queues={},
                )
            },
            per_request_info={},
            all_worker_graph_ids_to_graph_walks={worker_graph_id: {"image_gen"}},
            all_worker_graph_ids_to_nodes={
                worker_graph_id: graph.get_node_names()
            },
        )

        # Add a request
        request_id = "test_req_1"
        manager.add_request(
            request_id=request_id,
            worker_graph_ids=[worker_graph_id],
            worker_graph_to_worker={worker_graph_id: "worker_0"},
        )
        manager.update_graph_walk(request_id, "image_gen")

        # Provide initial inputs
        initial_inputs = [
            GraphEdge(name="text", next_node="text_emb"),
            GraphEdge(name="images", next_node="image_emb"),
            GraphEdge(name="existing_text_emb", next_node="concat_text"),
            GraphEdge(name="existing_image_emb", next_node="concat_img"),
            GraphEdge(name="latents", next_node="LLM"),
        ]
        manager.process_new_inputs(request_id, initial_inputs)

        fired_nodes = []
        max_iterations = 100

        for _ in range(max_iterations):
            queue = manager.queues[worker_graph_id]
            ready_map = queue.get_ready_node_names()

            if request_id not in ready_map or not ready_map[request_id]:
                # Check if done
                if queue.is_done(request_id):
                    break
                break  # no more ready nodes

            # Pop one node at a time
            node_name = ready_map[request_id][0]
            nodes = queue.pop_ready_nodes(request_id, [node_name])
            assert nodes, f"Expected to pop node {node_name}"

            node = nodes[0]
            fired_nodes.append(node.name)

            # Process outputs through the manager
            routing = manager.process_node_outputs(request_id, node.outputs)

            # In single-worker scenario, all routing should be internal
            # (to_workers should be empty since all nodes are on this worker)

        assert "VAE_dec" in fired_nodes
        assert fired_nodes.count("LLM") == 10
        assert fired_nodes.count("flow") == 10

    def test_micro_scheduler_picks_nodes(self):
        """Test that MicroScheduler correctly selects and batches ready nodes."""
        graph = self._build_image_gen_graph()
        worker_graph_id = "sg_test"
        worker_graph = WorkerGraph(
            section=graph,
            graph_walks={"image_gen"},
            worker_graph_id=worker_graph_id,
        )

        engine_configs = [
            {"engine_type": "ar", "node_names": ["LLM"], "model_config": {}},
            {"engine_type": "flow", "node_names": ["flow"], "model_config": {}},
            {
                "engine_type": "enc_dec",
                "node_names": [
                    "text_emb", "image_emb", "concat_text", "concat_img", "VAE_dec"
                ],
                "model_config": {},
            },
        ]
        engine_mgr = EngineManager.from_config(engine_configs, device="cpu")
        scheduler = MicroScheduler(engine_mgr)

        manager = WorkerGraphsManager(
            queues={
                worker_graph_id: WorkerGraphQueues(
                    worker_graph_id=worker_graph_id,
                    graph_walks={"image_gen"},
                    worker_graph=worker_graph,
                    per_request_queues={},
                )
            },
            per_request_info={},
            all_worker_graph_ids_to_graph_walks={worker_graph_id: {"image_gen"}},
            all_worker_graph_ids_to_nodes={
                worker_graph_id: graph.get_node_names()
            },
        )

        request_id = "req_sched"
        manager.add_request(
            request_id=request_id,
            worker_graph_ids=[worker_graph_id],
            worker_graph_to_worker={worker_graph_id: "w0"},
        )
        manager.update_graph_walk(request_id, "image_gen")

        # Initially no nodes ready
        batch = scheduler.get_next_batch(manager)
        assert batch is None

        # Feed inputs
        initial_inputs = [
            GraphEdge(name="text", next_node="text_emb"),
            GraphEdge(name="images", next_node="image_emb"),
            GraphEdge(name="existing_text_emb", next_node="concat_text"),
            GraphEdge(name="existing_image_emb", next_node="concat_img"),
            GraphEdge(name="latents", next_node="LLM"),
        ]
        manager.process_new_inputs(request_id, initial_inputs)

        # Now scheduler should find ready nodes (text_emb and image_emb)
        batch = scheduler.get_next_batch(manager)
        assert batch is not None
        # enc_dec has lowest priority, but these are the only ready nodes
        assert batch.node_name in ("text_emb", "image_emb")


# ======================================================================
# Prefill/decode graph test
# ======================================================================


class TestPrefillDecodeGraph:
    def test_prefill_graph(self):
        """Verify prefill graph fires text_emb + image_emb -> concat -> LLM."""
        model = DummyModel()
        graphs = model.get_graph_walk_graphs()
        prefill = graphs["prefill"]

        queues = PerRequestNodeQueues(waiting=prefill)
        inputs = [
            GraphEdge(name="text", next_node="text_emb"),
            GraphEdge(name="images", next_node="image_emb"),
            GraphEdge(name="existing_text_emb", next_node="concat_text"),
            GraphEdge(name="existing_image_emb", next_node="concat_img"),
        ]
        queues.process_new_inputs(inputs)

        fired = []
        for _ in range(20):
            if not queues.ready and queues.waiting is None:
                break
            assert queues.ready
            node = queues.ready.pop(0)
            fired.append(node.name)
            queues.process_new_inputs(node.outputs)

        assert fired[-1] == "LLM"
        assert "text_emb" in fired
        assert "image_emb" in fired
        assert "concat_text" in fired
        assert "concat_img" in fired


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
