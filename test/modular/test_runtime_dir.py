"""The IPC socket directory and the upload directory default to predictable
names under the shared ``/tmp``. Startup must refuse one that already exists
as a symlink, belongs to another uid, or is writable by group or others; a
fresh one is created 0700. Uploaded media is written to new 0600 files, never
through an existing file at the name.

Another uid's directory is simulated by reporting a different euid, since a
test cannot chown without root.
"""

import base64
import os
import stat

import pytest
import yaml

from mstar.api_server import media_io
from mstar.communication.communicator import (
    DEPLOYMENT_ANCHOR_ENTITY,
    CommProtocol,
    ZMQCommunicator,
)
from mstar.utils.runtime_dir import ensure_private_dir, write_private_file


def _mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.fixture(params=["other_uid", "symlink", "mode_777"])
def bad_dir(request, tmp_path, monkeypatch):
    """A directory startup must refuse, as a path with the trailing slash the
    ``mstar serve`` defaults carry."""
    path = tmp_path / "run"
    if request.param == "symlink":
        target = tmp_path / "elsewhere"
        target.mkdir(mode=0o700)
        path.symlink_to(target)
    else:
        path.mkdir(mode=0o700)
        if request.param == "mode_777":
            path.chmod(0o777)
        else:
            uid = os.geteuid()
            monkeypatch.setattr(os, "geteuid", lambda: uid + 1)
    return f"{path}/"


@pytest.fixture(params=[0o022, 0o002], ids=["umask022", "umask002"])
def umask(request):
    old = os.umask(request.param)
    yield request.param
    os.umask(old)


class _ReachedError(Exception):
    """Startup got past the directory check."""


def test_helper_refuses(bad_dir):
    with pytest.raises(PermissionError, match=r"refusing .*/run: .*--socket-path-prefix / --upload-dir"):
        ensure_private_dir(bad_dir)


def test_helper_names_the_owner(tmp_path, monkeypatch):
    path = tmp_path / "run"
    path.mkdir(mode=0o700)
    real = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real + 1)
    with pytest.raises(PermissionError, match=f"owned by uid {real}"):
        ensure_private_dir(path)


def test_zmq_communicator_refuses_the_socket_dir(bad_dir):
    with pytest.raises(PermissionError, match="refusing"):
        ZMQCommunicator(DEPLOYMENT_ANCHOR_ENTITY, push_ids=[], protocol=CommProtocol.IPC,
                        ipc_socket_path_prefix=bad_dir)


def test_rust_communicator_refuses_the_socket_dir(bad_dir):
    pytest.importorskip("mstar_rust")
    from mstar.communication.rust_communicator import RustZMQCommunicator

    with pytest.raises(PermissionError, match="refusing"):
        RustZMQCommunicator(DEPLOYMENT_ANCHOR_ENTITY, push_ids=[], protocol=CommProtocol.IPC,
                            ipc_socket_path_prefix=bad_dir)


def test_conductor_refuses_the_socket_dir_before_launching_workers(bad_dir, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import mstar.conductor.conductor as conductor_mod
    from mstar.graph.base import GraphEdge, GraphNode
    from mstar.graph.special_destinations import EMIT_TO_CLIENT
    from mstar.model.base import Model

    class _OneNode(Model):
        def get_node_resources(self):
            return []

        def get_graph_walk_graphs(self):
            return {"prefill": GraphNode(name="llm", input_names=["text_inputs"], outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
            ])}

        def get_submodule(self, *a, **k):
            return None

        def get_initial_forward_pass_args(self, *a, **k):
            raise NotImplementedError

        def get_partition_forward_pass_args(self, *a, **k):
            raise NotImplementedError

        def process_prompt(self, *a, **k):
            return {}

        def postprocess(self, *a, **k):
            return b""

    def _launch(self):
        raise _ReachedError

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"max_seq_len": 64, "node_groups": [{"node_names": ["llm"], "ranks": [0]}]}))
    monkeypatch.setattr(conductor_mod.Conductor, "_launch_workers", _launch)
    with pytest.raises(PermissionError, match="refusing"):
        conductor_mod.Conductor(_OneNode(), str(cfg), socket_path_prefix=bad_dir)


def test_api_server_refuses_the_upload_dir(bad_dir, monkeypatch):
    from mstar.api_server import entrypoint

    def _preprocess_worker(**_):
        raise _ReachedError

    monkeypatch.setattr(entrypoint, "PreprocessWorker", _preprocess_worker)
    with pytest.raises(PermissionError, match="refusing"):
        entrypoint.APIServer(upload_dir=bad_dir)


def test_a_fresh_start_creates_0700_dirs(umask, tmp_path):
    made = ensure_private_dir(f"{tmp_path}/parent/uploads/")
    assert made == tmp_path / "parent" / "uploads"
    assert _mode(made) == 0o700
    assert ensure_private_dir(made) == made

    sockets = tmp_path / "sockets"
    comm = ZMQCommunicator(DEPLOYMENT_ANCHOR_ENTITY, push_ids=[], protocol=CommProtocol.IPC,
                           ipc_socket_path_prefix=f"{sockets}/")
    try:
        assert _mode(sockets) == 0o700
    finally:
        comm.pull_socket.close()


def test_a_parent_created_for_a_nested_dir_passes_its_own_check(umask, tmp_path):
    """The upload dir inside the socket dir, created first: the socket dir it
    made on the way passes the socket dir's check."""
    runtime = tmp_path / "runtime"
    ensure_private_dir(runtime / "uploads")
    assert _mode(runtime) == 0o700
    assert ensure_private_dir(runtime) == runtime


def test_a_subdir_of_a_shared_mount_is_accepted(umask, tmp_path):
    """A volume mounted group-writable and setgid (e.g. Kubernetes ``fsGroup``)
    is refused as the directory itself, but works as the parent of one."""
    mount = tmp_path / "mount"
    mount.mkdir()
    mount.chmod(0o2777)
    with pytest.raises(PermissionError, match="refusing"):
        ensure_private_dir(mount)
    made = ensure_private_dir(mount / "uploads")
    assert _mode(made) & 0o077 == 0


def test_upload_files_are_0600(umask, tmp_path):
    from mstar.api_server.entrypoint import _decode_ws_files
    from mstar.api_server.openai.serving_transcriptions import save_upload

    up = tmp_path / "up"
    payload = base64.b64encode(b"media").decode()
    paths = [
        media_io.save_data_url(f"data:image/png;base64,{payload}", up)[1],
        media_io.save_base64(payload, "wav", "audio", up)[1],
        save_upload(b"audio", "clip.wav", up),
        media_io.write_wav([0.0, 0.5], str(up / "window.wav")),
        *_decode_ws_files([{"name": "a.png", "data": b"img"}], up)[0]["image"],
    ]
    assert {p: _mode(p) for p in paths} == dict.fromkeys(paths, 0o600)


def _tensor(name, data_type, shape, values):
    return {"metadata": {"name": name, "data_type": data_type, "shape": shape},
            "data": {"data_type": data_type, "values": values}}


def _dynamo_tensor_uploads(spec_name, upload_dir):
    """Run a Dynamo tensor spec's ``submit_args`` and return the files it saved."""
    pytest.importorskip("PIL")
    from mstar.integrations.dynamo.bridges import TENSOR_SPECS

    if spec_name == "pi05":
        tensors = {
            "images": _tensor("images", "Uint8", [2, 2, 2, 3], list(range(24))),
            "prompt": _tensor("prompt", "Bytes", [1], [list(b"pick")]),
        }
        args = TENSOR_SPECS["pi05"].submit_args(tensors, upload_dir)
        return args.file_paths["image"]
    trajectory = _tensor("t", "Float32", [2, 3], [0.0] * 6)
    tensors = {
        "video": _tensor("video", "Bytes", [1], [list(b"mp4")]),
        "actions": trajectory,
        "states": trajectory,
    }
    args = TENSOR_SPECS["vjepa2_ac"].submit_args(tensors, upload_dir)
    return args.file_paths["video"]


@pytest.mark.parametrize("spec_name", ["pi05", "vjepa2_ac"])
def test_dynamo_tensor_uploads_check_the_dir_and_are_0600(spec_name, umask, tmp_path):
    paths = _dynamo_tensor_uploads(spec_name, tmp_path / "up")
    assert _mode(tmp_path / "up") == 0o700
    assert paths and {p: _mode(p) for p in paths} == dict.fromkeys(paths, 0o600)


@pytest.mark.parametrize("spec_name", ["pi05", "vjepa2_ac"])
def test_dynamo_tensor_uploads_refuse_the_upload_dir(spec_name, bad_dir):
    with pytest.raises(PermissionError, match="refusing"):
        _dynamo_tensor_uploads(spec_name, bad_dir)


def test_an_existing_file_at_the_upload_name_is_not_overwritten(tmp_path, monkeypatch):
    up = ensure_private_dir(tmp_path / "up")
    taken = up / "00000000-0000-0000-0000-000000000000.png"
    taken.write_bytes(b"keep")
    monkeypatch.setattr(media_io, "uuid4", lambda: "00000000-0000-0000-0000-000000000000")
    payload = base64.b64encode(b"new").decode()
    with pytest.raises(FileExistsError):
        media_io.save_data_url(f"data:image/png;base64,{payload}", up)
    assert taken.read_bytes() == b"keep"

    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    (up / "planted.wav").symlink_to(outside)
    with pytest.raises(FileExistsError):
        write_private_file(up / "planted.wav", b"new")
    assert outside.read_bytes() == b"keep"


class _StopError(Exception):
    pass


def _upload_dirs(monkeypatch, tmp_path) -> dict:
    """The upload dir each entry point uses when none is given, without starting a server."""
    import sys
    import types

    from mstar.api_server import entrypoint
    from mstar.cli import main as cli
    from mstar.integrations.dynamo import worker
    from mstar.model import registry

    seen = {}

    def fake_api_server(**kwargs):
        seen["mstar-serve"] = kwargs["upload_dir"]
        raise _StopError

    class _Model:
        PREPROCESS_TORCH_THREADS = None

        def __init__(self, **kwargs):
            pass

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: fake\n")
    with monkeypatch.context() as m:
        m.setattr(registry, "get_model_class", lambda name: _Model)
        m.setattr(entrypoint, "model_init_kwargs", lambda name: {})
        m.setattr(entrypoint, "APIServer", fake_api_server)
        with pytest.raises(_StopError):
            entrypoint.main(["--config", str(cfg)])

    def fake_ensure(path):
        seen["APIServer"] = path
        raise _StopError

    with monkeypatch.context() as m:
        m.setattr(entrypoint, "ensure_private_dir", fake_ensure)
        with pytest.raises(_StopError):
            entrypoint.APIServer()

    def fake_build_server(args):
        seen["dynamo worker"] = args.upload_dir
        raise _StopError

    with monkeypatch.context() as m:
        m.setattr(worker, "build_server", fake_build_server)
        with pytest.raises(_StopError):
            worker.main(["--config", "c.yaml", "--model-path", "/m", "--socket-path-prefix", str(tmp_path)])

    argvs = []
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "mstar.api_server.entrypoint", types.SimpleNamespace(main=argvs.append))
        cli.main(["serve", "kokoro"])
    seen["mstar serve"] = argvs[0][argvs[0].index("--upload-dir") + 1]
    return seen


def test_every_entry_point_defaults_to_a_per_user_upload_dir(monkeypatch, tmp_path):
    # a shared default would be refused for every user but the one who created it
    from mstar.utils import socket_prefix

    monkeypatch.setattr(socket_prefix.getpass, "getuser", lambda: "alice")
    assert _upload_dirs(monkeypatch, tmp_path) == dict.fromkeys(
        ("mstar-serve", "APIServer", "dynamo worker", "mstar serve"), "/tmp/mstar_uploads_alice/")
    monkeypatch.setattr(socket_prefix.getpass, "getuser", lambda: "bob")
    assert set(_upload_dirs(monkeypatch, tmp_path).values()) == {"/tmp/mstar_uploads_bob/"}


@pytest.mark.parametrize("name, kept", [("alice", True), ("a" * 32, True), ("a" * 33, False),
                                        ("\u00e9" * 30, False), ("a b", False), ("", False)])
def test_the_upload_dir_default_holds_the_user_name_only_when_short_and_ascii(monkeypatch, name, kept):
    # the same rule as the socket prefix; the Rust twin is runtime_dir::upload_dir_for
    from mstar.utils import socket_prefix

    monkeypatch.setattr(socket_prefix.getpass, "getuser", lambda: name)
    assert socket_prefix.default_upload_dir() == f"/tmp/mstar_uploads_{name if kept else os.getuid()}/"


@pytest.fixture(params=["other_uid", "symlink", "mode_777"])
def swapped_upload_dir(request, tmp_path, monkeypatch):
    """An upload dir that passed the startup check, then was removed (a /tmp
    cleaner) and recreated by someone else; returns (path, where a symlink
    points, or None)."""
    path = ensure_private_dir(tmp_path / "up")
    path.rmdir()
    target = None
    if request.param == "symlink":
        target = tmp_path / "theirs"
        target.mkdir(mode=0o700)
        path.symlink_to(target)
    else:
        path.mkdir(mode=0o700)
        if request.param == "mode_777":
            path.chmod(0o777)
        else:
            uid = os.geteuid()
            monkeypatch.setattr(os, "geteuid", lambda: uid + 1)
    return path, target


def _written(path, target) -> list:
    return sorted(os.listdir(target if target is not None else path))


def test_generate_rechecks_the_upload_dir_per_request(swapped_upload_dir, monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from mstar.api_server import entrypoint

    path, target = swapped_upload_dir
    monkeypatch.setattr(entrypoint, "api_server", SimpleNamespace(upload_dir=path))
    client = TestClient(entrypoint.app)
    with pytest.raises(PermissionError, match="refusing"):
        client.post("/generate", data={"text": "hi"}, files=[("files", ("a.png", b"img", "image/png"))])
    assert _written(path, target) == []


def test_generate_ws_rechecks_the_upload_dir_per_message(swapped_upload_dir, monkeypatch):
    import json
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from mstar.api_server import entrypoint
    from mstar.api_server.entrypoint import _decode_ws_files

    path, target = swapped_upload_dir
    with pytest.raises(PermissionError, match="refusing"):
        _decode_ws_files([{"name": "a.png", "data": b"img"}], path)

    monkeypatch.setattr(entrypoint, "api_server", SimpleNamespace(upload_dir=path))
    payload = base64.b64encode(b"img").decode()
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "r1", "files": [{"name": "a.png", "data": payload}]}))
        reply = json.loads(ws.receive_text())
    assert reply["request_id"] == "r1" and reply["error"]
    assert _written(path, target) == []


def test_a_realtime_step_rechecks_the_upload_dir(swapped_upload_dir):
    """Realtime steps write each step's audio with ``write_wav`` and nothing
    else checks the upload dir on that path."""
    import asyncio
    from types import SimpleNamespace

    import numpy as np

    from mstar.api_server.openai.serving_realtime import _run_step

    path, target = swapped_upload_dir
    submitted = []
    api = SimpleNamespace(upload_dir=path, submit_request=lambda **kw: submitted.append(kw))
    session = SimpleNamespace(session_id="s1", audio_so_far=lambda: np.zeros(160, dtype=np.float32), request=None)
    with pytest.raises(PermissionError, match="refusing"):
        asyncio.run(_run_step(api, None, session, ""))
    assert submitted == [] and _written(path, target) == []
