"""Models may specialize their walk graphs from the deployment config.

Whoever constructs a model for workers (the conductor process, or a node
agent on another host) must run the config-parsing calls before spawning, or
the two sides disagree about which walks and nodes exist. BAGEL's
CFG-parallel walks are the concrete case: they only register when the
placement names the CFG nodes.
"""

import os

import pytest

pytest.importorskip("transformers")

CFG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "bagel_cfg_parallel.yaml")


def _fresh_bagel():
    from mstar.model.bagel.bagel_model import BagelModel
    from mstar.model.registry import HF_MODELS

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return BagelModel(
            model_path_hf=HF_MODELS["bagel"]["model_path_hf"], cache_dir=None
        )
    except Exception as e:  # tokenizer assets not cached on this machine
        pytest.skip(f"BAGEL tokenizer assets unavailable: {e}")


class TestConfigWarming:
    def test_fresh_model_lacks_cfg_walks(self):
        model = _fresh_bagel()
        walks = model.get_graph_walk_graphs()
        assert "image_gen_cfg" not in walks

    def test_config_call_registers_cfg_walks(self):
        model = _fresh_bagel()
        model.get_worker_graphs(CFG)
        walks = model.get_graph_walk_graphs()
        assert "image_gen_cfg" in walks
        cfg_nodes = set(walks["image_gen_cfg"].get_nodes())
        assert {"LLM_cfg_text", "LLM_cfg_img"} <= cfg_nodes

    def test_node_agent_warms_like_the_conductor(self):
        # The mapping workers build from the model (node -> partition) must
        # cover the CFG nodes after the agent's warming calls; a miss there
        # surfaces as scheduler KeyErrors on the remote host.
        model = _fresh_bagel()
        model.get_worker_graphs(CFG)
        model.get_sharding_config(CFG)
        node_to_partition = {}
        for pdef in model.get_partitions():
            walks = model.get_graph_walk_graphs()
            for walk_name in pdef.graph_walks:
                section = walks.get(walk_name)
                if section:
                    for node_name in section.get_nodes():
                        node_to_partition[node_name] = pdef.name
        assert "LLM_cfg_text" in node_to_partition
        assert "LLM_cfg_img" in node_to_partition


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
