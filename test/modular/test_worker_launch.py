"""
Tests for conductor worker derivation and launching.
"""
import sys

sys.path.insert(0, ".")

import os
import tempfile
import time

import pytest

from mminf.model.dummy_model import DummyModel

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "examples", "dummy.yaml")


class TestGetNodeEngineTypes:
    def test_returns_all_nodes(self):
        model = DummyModel()
        mapping = model.get_node_engine_types()
        expected_nodes = {"text_emb", "concat_text", "image_emb", "concat_img", "LLM", "flow", "VAE_dec"}
        assert set(mapping.keys()) == expected_nodes

    def test_engine_types_are_valid(self):
        model = DummyModel()
        mapping = model.get_node_engine_types()
        valid_types = {"ar", "flow", "enc_dec"}
        for node, etype in mapping.items():
            assert etype in valid_types, f"Node {node} has invalid engine type {etype}"

    def test_specific_mappings(self):
        model = DummyModel()
        mapping = model.get_node_engine_types()
        assert mapping["LLM"] == "ar"
        assert mapping["flow"] == "flow"
        assert mapping["text_emb"] == "enc_dec"
        assert mapping["VAE_dec"] == "enc_dec"


class TestDeriveWorkerInfo:
    def test_worker_ids(self):
        """Verify worker IDs are derived from unique ranks in the config."""
        from mminf.conductor.conductor import Conductor

        model = DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            conductor = Conductor(
                model=model,
                model_config_file=CONFIG_PATH,
                socket_path_prefix=os.path.join(tmpdir, "ipc_derive"),
            )
            # dummy.yaml has ranks 0, 1, 2, 3, 4
            assert sorted(conductor.worker_ids) == [
                "worker_0", "worker_1", "worker_2", "worker_3", "worker_4"
            ]
            conductor.shutdown()

    def test_per_worker_worker_graphs(self):
        """Verify each worker gets the correct worker graphs assigned to it."""
        from mminf.conductor.conductor import Conductor

        model = DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            conductor = Conductor(
                model=model,
                model_config_file=CONFIG_PATH,
                socket_path_prefix=os.path.join(tmpdir, "ipc_worker_graphs"),
            )
            # Every worker should have at least one worker graph
            for worker_id in conductor.worker_ids:
                assert len(conductor._per_worker_graphs[worker_id]) > 0, (
                    f"{worker_id} has no worker graphs"
                )
            conductor.shutdown()

    def test_per_worker_engine_configs(self):
        """Verify engine configs are built correctly per worker."""
        from mminf.conductor.conductor import Conductor

        model = DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            conductor = Conductor(
                model=model,
                model_config_file=CONFIG_PATH,
                socket_path_prefix=os.path.join(tmpdir, "ipc_engines"),
            )
            for worker_id in conductor.worker_ids:
                configs = conductor._per_worker_engine_configs[worker_id]
                assert len(configs) > 0, f"{worker_id} has no engine configs"
                for cfg in configs:
                    assert "engine_type" in cfg
                    assert "node_names" in cfg
                    assert cfg["engine_type"] in {"ar", "flow", "enc_dec"}
            conductor.shutdown()

    def test_global_worker_graph_maps(self):
        """Verify all_worker_graph_ids_to_graph_walks and all_worker_graph_ids_to_nodes are populated."""
        from mminf.conductor.conductor import Conductor

        model = DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            conductor = Conductor(
                model=model,
                model_config_file=CONFIG_PATH,
                socket_path_prefix=os.path.join(tmpdir, "ipc_worker_graphs_global"),
            )
            assert len(conductor._all_worker_graph_ids_to_graph_walks) == len(conductor.worker_graphs)
            assert len(conductor._all_worker_graph_ids_to_nodes) == len(conductor.worker_graphs)

            for worker_graph_id in conductor.worker_graphs:
                assert worker_graph_id in conductor._all_worker_graph_ids_to_graph_walks
                assert worker_graph_id in conductor._all_worker_graph_ids_to_nodes
                assert len(conductor._all_worker_graph_ids_to_graph_walks[worker_graph_id]) > 0
                assert len(conductor._all_worker_graph_ids_to_nodes[worker_graph_id]) > 0
            conductor.shutdown()


class TestWorkerSpawning:
    def test_workers_spawn_and_are_alive(self):
        """Integration test: spawn Worker processes, verify alive, shutdown, verify dead."""
        from mminf.conductor.conductor import Conductor

        model = DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            conductor = Conductor(
                model=model,
                model_config_file=CONFIG_PATH,
                socket_path_prefix=os.path.join(tmpdir, "ipc_spawn"),
            )

            # Give processes a moment to start
            time.sleep(1)

            # All processes should be alive
            assert len(conductor._worker_processes) == len(conductor.worker_ids)
            for p in conductor._worker_processes:
                assert p.is_alive(), f"Process {p.name} is not alive"

            # Shutdown
            conductor.shutdown()

            # All processes should be dead
            for p in conductor._worker_processes:
                assert not p.is_alive(), f"Process {p.name} is still alive after shutdown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
