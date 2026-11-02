import sys

sys.path.insert(0, ".")


from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule


def _make_model() -> OrpheusModel:
    model = object.__new__(OrpheusModel)
    model.config = OrpheusModelConfig()
    return model


def test_the_llm_captures_every_walk_it_runs():
    """A walk no capture replays runs eagerly, and nothing reports it."""
    model = _make_model()
    llm = OrpheusLLMSubmodule(None, model.config)

    captured = {walk for config in llm.get_accelerator_graph_configs("cpu") for walk in config.replay_graph_walks}
    runs = {walk for walk, graph in model.get_graph_walk_graphs().items() if "LLM" in graph.get_nodes()}

    assert runs | set(model.get_combined_graph_walks()["LLM"]) <= captured
