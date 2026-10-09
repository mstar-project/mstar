import json

from fastapi.testclient import TestClient

from mstar.api_server import entrypoint


def test_generate_rejects_malformed_model_kwargs(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", object())

    response = TestClient(entrypoint.app).post(
        "/generate",
        data={"model_kwargs": "{"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "model_kwargs must be valid JSON"}


def test_generate_rejects_non_object_model_kwargs(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", object())

    client = TestClient(entrypoint.app)
    for model_kwargs in ("[1, 2]", '"hello"', "42", "true"):
        response = client.post(
            "/generate",
            data={"model_kwargs": model_kwargs},
        )

        assert response.status_code == 400
        assert response.json() == {"detail": "model_kwargs must be a JSON object"}


def test_generate_does_not_mislabel_downstream_json_errors(monkeypatch):
    class FailingServer:
        def submit_request(self, **kwargs):
            raise json.JSONDecodeError("downstream failure", "{}", 0)

    monkeypatch.setattr(entrypoint, "api_server", FailingServer())

    response = TestClient(entrypoint.app).post("/generate")

    assert response.status_code == 500
    assert response.json()["detail"].startswith("downstream failure")


def test_generate_returns_400_for_a_refused_request(monkeypatch):
    """submit_request refuses bad input with a ValueError (a bad seed, an
    unsupported modality); that is the client's fault, not a 500."""
    class RefusingServer:
        def submit_request(self, **kwargs):
            raise ValueError("seed must fit in a signed 64-bit integer, got 18446744073709551616.")

    monkeypatch.setattr(entrypoint, "api_server", RefusingServer())

    response = TestClient(entrypoint.app).post("/generate")

    assert response.status_code == 400
    assert "64-bit" in response.json()["detail"]
