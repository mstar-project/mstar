"""The Praxis action head receives the expert's settled action, not its input."""

import pytest
import torch

from mstar.graph.base import TensorPointerInfo
from mstar.graph.graph_io import WorkerGraphIO
from mstar.model.praxis1.praxis1_model import DENOISE_LOOP_NAME, Praxis1Model


def _pointer(uuid: int) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=[1],
        uuid=uuid, source_session_id="test:0", source_entity="worker_0",
    )


@pytest.mark.parametrize("walk", [Praxis1Model.CHUNK_WALK, Praxis1Model.CHUNK_WARM_WALK])
@pytest.mark.parametrize("num_steps", [1, 2])
def test_action_head_gets_only_final_expert_output(walk: str, num_steps: int) -> None:
    model = Praxis1Model(
        "runwayml/praxis-1-droid", skip_weight_loading=True,
        num_steps=num_steps, max_context_chunks=1, max_text_tokens=8,
    )
    io = WorkerGraphIO(model.get_graph_walk_graphs()[walk])
    loop = io.loops[DENOISE_LOOP_NAME]
    loop.max_iters = num_steps  # the request's loop cap, applied by the runtime
    backbone = io.nodes["backbone"]
    expert = io.nodes["action_expert"]
    assert {edge.name for edge in backbone.outputs if edge.next_node == "action_expert"} >= {
        "expert_video_latents", "expert_action_latents",
    }
    assert {edge.name for edge in expert.outputs if edge.next_node == "backbone"} == {
        "video_latents", "action_latents",
    }

    for step in range(num_steps):
        # The backbone forwards its input unchanged to the expert. The expert
        # emits a distinct tensor after the Euler update.
        next(edge for edge in backbone.outputs if edge.next_node == "action_expert" and
             edge.name in {"action_latents", "expert_action_latents"}).tensor_info = [
                 _pointer(100 + step)
             ]
        io.mark_node_complete("backbone")
        assert "video_latents" not in loop._cached_outputs
        next(edge for edge in expert.outputs if edge.name == "action_latents").tensor_info = [
            _pointer(200 + step)
        ]
        completion = io.mark_node_complete("action_expert")

    to_head = [edge for edge in completion.output_edges if edge.next_node == "action_head" and
               edge.name == "action_latents"]
    assert loop.is_done
    assert len(to_head) == 1
    assert [info.uuid for info in to_head[0].tensor_info] == [199 + num_steps]
