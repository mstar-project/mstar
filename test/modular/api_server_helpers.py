"""An ``APIServer`` for tests, built without its ``__init__``.

``APIServer.__init__`` creates the upload dir, starts the ``PreprocessWorker``
and opens the ZMQ communicator. ``make_test_api_server`` skips those and calls
``APIServer._init_request_state``, the method ``__init__`` uses for the
in-memory request bookkeeping, so a field added there reaches every test that
builds its server here. ``test_api_server_helpers.py`` fails if the object
lacks any field a real ``APIServer`` has.

Import it by module name (``from api_server_helpers import ...``): on CI the
stdlib ``test`` package shadows the repo's ``test/`` directory.
"""
import tempfile
from pathlib import Path

from mstar.api_server.entrypoint import APIServer
from mstar.api_server.lora_admin import LoraAdmin
from mstar.api_server.served_models import ServedModels


def make_test_api_server(**overrides) -> APIServer:
    """An ``APIServer`` with every field ``__init__`` sets and no processes,
    sockets or files. ``preprocess_worker``, ``communicator`` and
    ``_msg_thread`` are None unless passed in. ``overrides`` replace fields
    by name; a name ``APIServer`` does not have raises ``TypeError``.
    """
    server = object.__new__(APIServer)
    server._init_request_state()
    server.upload_dir = Path(tempfile.gettempdir())
    server.timeout_seconds = 5.0
    server.enable_nvtx = False
    server.log_stats = False
    server.log_stats_file = None
    server.metrics = None
    server.model = None
    server.model_name = "dummy"
    server.served_models = ServedModels.of("dummy", None)
    # no LoRA block: no adapters to digest, as APIServer.__init__ leaves it
    server.lora_adapter_digests = {}
    server.lora_admin = LoraAdmin(None, server.served_models, None, lambda message: None)
    server.preprocess_worker = None
    server.communicator = None
    server._msg_thread = None
    for name, value in overrides.items():
        if not hasattr(server, name):
            raise TypeError(f"APIServer has no field {name!r}")
        setattr(server, name, value)
    return server
