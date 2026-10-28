"""The Dynamo compose file gives the frontend the worker's model cache.

The worker registers its model by its ``--model-path`` (a snapshot under the
``/models`` mount), and the Dynamo frontend loads the tokenizer and config
from that same path, so a frontend without the mount never lists the model.
"""

from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "mstar/integrations/dynamo/docker/compose.yaml"


def _services():
    return yaml.safe_load(COMPOSE.read_text())["services"]


def _env(service):
    return dict(item.split("=", 1) for item in service.get("environment", []))


def _model_mount(service):
    """The ``/models`` volume entry (``source:target[:mode]``), or None."""
    # the source is a ``${VAR:-default}`` expansion, so split from the right
    return next((v for v in service.get("volumes", []) if "/models" in v.rsplit(":", 2)[1:]), None)


def test_worker_serves_from_the_models_mount():
    worker = _services()["worker"]
    command = worker["command"]
    assert "/models" in command[command.index("--model-path") + 1]
    assert _model_mount(worker) is not None


def test_frontend_mounts_the_worker_model_cache_read_only():
    services = _services()
    mount = _model_mount(services["frontend"])
    assert mount == _model_mount(services["worker"])
    assert mount.endswith(":ro")


def test_frontend_uses_the_worker_hf_home():
    services = _services()
    assert _env(services["frontend"]).get("HF_HOME") == _env(services["worker"])["HF_HOME"]
