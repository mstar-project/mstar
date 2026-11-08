"""``StubModel`` and ``make_model`` stand in for a real ``Model``."""
import pytest

from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import Model, names_checkpoint
from mstar.model.submodule_base import ChunkingPolicy, chunk_loop_name
from mstar.testing_utils.model import StubModel, make_model


def test_a_stub_declares_nothing_and_inherits_the_base_defaults():
    model = make_model()
    assert isinstance(model, Model), "code under test may rely on any method of Model"
    assert model.get_node_resources() == [] and model.get_graph_walk_graphs() == {}
    assert model.get_submodule("LLM") is None, "None is how a model asks for dummy mode"
    assert model.kv_labels() == {}


@pytest.mark.parametrize("method", [
    "get_initial_forward_pass_args", "get_partition_forward_pass_args", "process_prompt",
])
def test_a_stub_refuses_to_invent_a_request_path(method):
    with pytest.raises(NotImplementedError, match=method):
        getattr(make_model(), method)()


def test_an_override_replaces_a_method_and_takes_the_call_arguments():
    model = make_model(get_max_output_tokens=lambda **kw: kw["cap"] + 1, get_node_resources=list)
    assert model.get_max_output_tokens(cap=7) == 8
    assert model.get_node_resources() == []


def test_an_override_replaces_a_plain_attribute():
    assert make_model(LENGTH_KWARGS=("n",)).LENGTH_KWARGS == ("n",)


def test_an_override_replaces_a_property():
    assert make_model(nodes=["LLM"]).nodes == ["LLM"]


def test_a_name_model_does_not_have_is_refused():
    with pytest.raises(TypeError, match="get_node_resource"):
        make_model(get_node_resource=list)


def test_overrides_stay_on_their_own_stub():
    make_model(get_node_resources=lambda: ["kv"])
    assert make_model().get_node_resources() == []


def test_an_override_counts_as_the_class_overriding_the_method():
    """``names_checkpoint`` reads the class, so as not to call a path that may download."""
    assert not names_checkpoint(make_model())
    assert names_checkpoint(make_model(checkpoint_path=lambda: "/a/checkpoint"))


def test_overridden_walks_are_chunked_like_a_real_models():
    node = GraphNode(name="LLM", input_names=["ids"], outputs=[
        GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
    ])
    policy = ChunkingPolicy(walks=("prefill",), chunkable=True, max_batch_tokens=8)
    model = make_model(
        get_graph_walk_graphs=lambda: {"prefill": node}, get_chunking_policies=lambda: {"LLM": (policy,)},
    )
    assert chunk_loop_name("LLM", "prefill") in model.get_graph_walk_graphs()["prefill"].get_loops()


def test_a_subclass_keeps_its_own_state_and_the_defaults():
    class _Counting(StubModel):
        def __init__(self):
            self.calls = 0

        def process_prompt(self, *args, **kwargs):
            self.calls += 1
            return {}

    model = _Counting()
    assert model.process_prompt("hi") == {} and model.calls == 1
    assert model.get_max_output_tokens(max_output_tokens=3) == 3
