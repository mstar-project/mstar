"""A Cosmos3 request with no prompt must still feed the DiT prefill its
``text_inputs``, or that node never becomes ready and the request hangs with
the worker idle. /generate sends an empty ``text`` field as None, so an empty
task ends up here too."""

from __future__ import annotations

import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.graph.base import TensorPointerInfo
from mstar.graph.graph_io import WorkerGraphIO
from mstar.model.base import ProcessPromptOutput
from mstar.model.cosmos3.cosmos3_model import DIT_NODE, VAE_ENCODER_NODE, Cosmos3Model

POLICY = {
    "action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10,
    "action_chunk_size": 32, "num_frames": 33,
}


def _signals(tensors: dict[str, list[torch.Tensor]]) -> dict[str, list[TensorPointerInfo]]:
    return {
        name: [
            TensorPointerInfo(
                dims=list(t.shape), dtype=t.dtype, nbytes=t.nbytes, address=0,
                stride=list(t.stride()), uuid=i, source_session_id="s", source_entity="api_server",
            )
            for i, t in enumerate(ts)
        ]
        for name, ts in tensors.items()
    }


@pytest.mark.parametrize("prompt", [None, "", "pick up the mug"])
@pytest.mark.parametrize("out_mod,kwargs", [("action", POLICY), ("image", {})], ids=["action", "image"])
def test_prefill_nodes_ready_without_prompt(prompt, out_mod, kwargs) -> None:
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    in_mods = ["image", "text"] if prompt else ["image"]
    tensors = {"image_inputs": [torch.zeros(3, 8, 8)]}

    # What the preprocess worker does before handing the request to the conductor.
    out = model.process_prompt(prompt, in_mods, [out_mod], tensors=dict(tensors), **kwargs)
    if isinstance(out, ProcessPromptOutput):
        out = out.new_input_tensors
    tensors.update(out)
    fpa = model.get_initial_forward_pass_args(
        "default", in_mods, [out_mod], _signals(tensors), model_kwargs=kwargs,
    )

    io = WorkerGraphIO(model.get_graph_walk_graphs()[fpa.full_metadata.graph_walk])
    for edge in fpa.inputs:
        assert io.ingest_input(edge), edge.name
    assert io.ready_node_names == {DIT_NODE, VAE_ENCODER_NODE}


def test_generate_form_turns_empty_text_into_none(monkeypatch) -> None:
    seen = {}

    class Server:
        async def wait_for_request_id(self, request_id):
            return None

        def submit_request(self, **kwargs):
            seen.update(kwargs)
            raise HTTPException(status_code=418)

    monkeypatch.setattr(entrypoint, "api_server", Server())
    response = TestClient(entrypoint.app).post("/generate", data={
        "text": "", "input_modalities": "image,text", "output_modalities": "action",
    })
    assert response.status_code == 418
    assert seen["text"] is None and seen["input_modalities"] == ["image"]
