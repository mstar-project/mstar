"""A node's session store lives in that node's process, and nothing routes a request to the replica that holds its
session. On a deployment that runs the node on two replicas, a resumed session lands on the other one about half the
time and finds nothing, so a request that names a session is refused when it is submitted. The count comes from the
worker graphs at start, and a deployment whose replicas can't be counted fails there instead of serving with the
refusal off.
"""

from types import SimpleNamespace

import pytest
from api_server_helpers import make_test_api_server
from fastapi import HTTPException

from mstar.api_server.entrypoint import _node_replicas


class _Sessions:
    """Keeps sessions on node ``dit`` for a request that names ``session_id``."""

    def session_node(self, model_kwargs):
        return "dit" if model_kwargs.get("session_id") else None

    def request_kwargs(self):
        return None


def _worker_graph(nodes, replicas):
    return SimpleNamespace(
        section=SimpleNamespace(get_nodes=lambda: dict.fromkeys(nodes)), replica_ranks=[[r] for r in range(replicas)],
    )


def test_a_session_on_two_replicas_is_refused_at_submission():
    server = make_test_api_server(model=_Sessions(), node_replicas={"dit": 2, "decoder": 1})
    with pytest.raises(HTTPException) as refused:
        server.submit_request(
            text="a prompt", input_modalities=["text"], output_modalities=["text"], model_kwargs={"session_id": "s"},
        )
    assert refused.value.status_code == 400 and "node 'dit'" in refused.value.detail
    assert not server.pending_requests, "a refused request should leave nothing pending"


def test_a_node_counts_the_most_replicas_any_worker_graph_gives_it():
    model = SimpleNamespace(get_worker_graphs=lambda path: [
        _worker_graph({"dit", "decoder"}, 1), _worker_graph({"dit"}, 2),
    ])
    assert _node_replicas(model, "deployment.yaml") == {"dit": 2, "decoder": 1}


def test_an_uncounted_deployment_fails_at_start():
    def placement(path):
        raise RuntimeError("no placement")

    with pytest.raises(RuntimeError, match="no placement"):
        _node_replicas(SimpleNamespace(get_worker_graphs=placement), "deployment.yaml")
