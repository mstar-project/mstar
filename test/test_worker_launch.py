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


class TestGetStageEngineTypes:
    def test_returns_all_stages(self):
        model = DummyModel()
        mapping = model.get_stage_engine_types()
        expected_stages = {"text_emb", "concat_text", "image_emb", "concat_img", "LLM", "flow", "VAE_dec"}
        assert set(mapping.keys()) == expected_stages

    def test_engine_types_are_valid(self):
        model = DummyModel()
        mapping = model.get_stage_engine_types()
        valid_types = {"ar", "flow", "enc_dec"}
        for stage, etype in mapping.items():
            assert etype in valid_types, f"Stage {stage} has invalid engine type {etype}"

    def test_specific_mappings(self):
        model = DummyModel()
        mapping = model.get_stage_engine_types()
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

    def test_per_worker_subgraphs(self):
        """Verify each worker gets the correct subgraphs assigned to it."""
        from mminf.conductor.conductor import Conductor

        model = DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            conductor = Conductor(
                model=model,
                model_config_file=CONFIG_PATH,
                socket_path_prefix=os.path.join(tmpdir, "ipc_subgraphs"),
            )
            # Every worker should have at least one subgraph
            for worker_id in conductor.worker_ids:
                assert len(conductor._per_worker_subgraphs[worker_id]) > 0, (
                    f"{worker_id} has no subgraphs"
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
                    assert "stage_names" in cfg
                    assert cfg["engine_type"] in {"ar", "flow", "enc_dec"}
            conductor.shutdown()

    def test_global_subgraph_maps(self):
        """Verify all_subgraph_ids_to_phases and all_subgraph_ids_to_stages are populated."""
        from mminf.conductor.conductor import Conductor

        model = DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            conductor = Conductor(
                model=model,
                model_config_file=CONFIG_PATH,
                socket_path_prefix=os.path.join(tmpdir, "ipc_global"),
            )
            assert len(conductor._all_subgraph_ids_to_phases) == len(conductor.subgraphs)
            assert len(conductor._all_subgraph_ids_to_stages) == len(conductor.subgraphs)

            for sg_id in conductor.subgraphs:
                assert sg_id in conductor._all_subgraph_ids_to_phases
                assert sg_id in conductor._all_subgraph_ids_to_stages
                assert len(conductor._all_subgraph_ids_to_phases[sg_id]) > 0
                assert len(conductor._all_subgraph_ids_to_stages[sg_id]) > 0
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
