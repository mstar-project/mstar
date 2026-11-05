"""The IPC socket directory and the upload directory default to predictable
names under the shared ``/tmp``. Startup must refuse one that already exists
as a symlink, as a non-directory, or under another uid; a fresh one is created
0700, and an existing one of our own is tightened to 0700 rather than refused,
because earlier releases made these directories with ``makedirs`` and left
them 0775 under Ubuntu's default umask 002. Uploaded media is written to new
0600 files, never through an existing file at the name.

Another uid's directory is simulated by reporting a different euid, since a
test cannot chown without root.
"""

import base64
import logging
import os
import re
import stat

import pytest
import yaml

from mstar.api_server import media_io
from mstar.communication.communicator import DEPLOYMENT_ANCHOR_ENTITY, CommProtocol
from mstar.utils.runtime_dir import ensure_private_dir, write_private_file


def _mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.fixture(params=["other_uid", "symlink", "not_a_dir"])
def bad_dir(request, tmp_path, monkeypatch):
    """A directory startup must refuse, as a path with the trailing slash the
    ``mstar serve`` defaults carry. The symlink points at a directory of our
    own, which would be accepted under its own name: it is the name that is
    refused, so the target is never opened or tightened."""
    path = tmp_path / "run"
    if request.param == "symlink":
        target = tmp_path / "elsewhere"
        target.mkdir(mode=0o700)
        path.symlink_to(target)
    elif request.param == "not_a_dir":
        path.write_bytes(b"not a dir")
    else:
        path.mkdir()
        path.chmod(0o775)  # as an earlier release left it; chmod, so not the umask's
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


def test_the_communicator_creates_its_socket_dir_0700(umask, tmp_path):
    pytest.importorskip("mstar_rust")
    from mstar.communication.rust_communicator import RustZMQCommunicator

    sockets = tmp_path / "sockets"
    comm = RustZMQCommunicator(DEPLOYMENT_ANCHOR_ENTITY, push_ids=[], protocol=CommProtocol.IPC,
                               ipc_socket_path_prefix=f"{sockets}/")
    try:
        assert _mode(sockets) == 0o700
    finally:
        # the anchor entity holds an flock beside its socket; the kernel would
        # only release it at exit, and other tests reuse this process
        os.close(comm._deployment_lock_fd)


def test_a_parent_created_for_a_nested_dir_passes_its_own_check(umask, tmp_path):
    """The upload dir inside the socket dir, created first: the socket dir it
    made on the way passes the socket dir's check."""
    runtime = tmp_path / "runtime"
    ensure_private_dir(runtime / "uploads")
    assert _mode(runtime) == 0o700
    assert ensure_private_dir(runtime) == runtime


def test_a_subdir_of_a_shared_mount_is_accepted(umask, tmp_path):
    """A volume mounted group-writable and setgid (e.g. Kubernetes ``fsGroup``)
    works as the parent of the directory. Passed as the directory itself it is
    tightened to 0700, which is why the docs say to pass a subdirectory: the
    mount would stop being shared. A restart finds the subdirectory already
    there and keeps it 0700, setgid included."""
    mount = tmp_path / "mount"
    mount.mkdir()
    mount.chmod(0o2777)
    made = ensure_private_dir(mount / "uploads")
    assert _mode(made) == 0o700
    assert ensure_private_dir(mount / "uploads") == made and _mode(made) == 0o700
    assert ensure_private_dir(mount) == mount and _mode(mount) == 0o700


def test_a_dir_we_own_is_tightened_to_0700_whatever_its_mode(umask, tmp_path):
    """The upgrade case: before this, main made these directories with
    ``makedirs``/``create_dir_all``, which Ubuntu's default umask 002 leaves at
    0775. Refusing those would stop the first start after this lands, so a real
    directory of our own is chmoded 0700 instead."""
    for mode in (0o775, 0o777, 0o770, 0o755, 0o702, 0o2775):
        path = tmp_path / f"m{mode:o}"
        path.mkdir()
        path.chmod(mode)
        assert ensure_private_dir(path) == path, f"{mode:o}"
        assert _mode(path) == 0o700, f"{mode:o}"


def test_a_dir_another_uid_owns_is_refused_without_being_chmoded(tmp_path, monkeypatch):
    """The owner is read from the descriptor the chmod would use, so the chmod
    can never reach a directory that is not ours: no stat-then-chmod window."""
    path = tmp_path / "theirs"
    path.mkdir()
    # chmod, not mkdir's mode argument: that one is masked by the ambient umask,
    # so the mode asserted below would be the test host's, not a chosen one
    path.chmod(0o775)
    before = _mode(path)
    uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: uid + 1)
    with pytest.raises(PermissionError, match=f"owned by uid {uid}"):
        ensure_private_dir(path)
    assert _mode(path) == before


def test_a_symlink_to_a_dir_we_own_is_refused_without_tightening_its_target(tmp_path):
    """``O_DIRECTORY | O_NOFOLLOW`` never opens through the name, so the target
    is not chmoded even though it would be accepted under its own name."""
    target = tmp_path / "target"
    target.mkdir()
    # chmod, not mkdir's mode argument, for the reason above
    target.chmod(0o775)
    before = _mode(target)
    link = tmp_path / "link"
    link.symlink_to(target)
    for path in (str(link), f"{link}/"):
        with pytest.raises(PermissionError, match="it is a symlink"):
            ensure_private_dir(path)
    assert _mode(target) == before
    # and it is only the name that was refused: the target passes on its own
    assert ensure_private_dir(target) == target and _mode(target) == 0o700


def test_the_entry_points_start_on_dirs_an_earlier_release_left_0775(tmp_path, monkeypatch):
    """#617's upgrade gap, on the three paths that refused: the API server, the
    communicator and the conductor, on directories made exactly as main made
    them (``makedirs(exist_ok=True)``) under umask 002. The Rust frontend's own
    case is ``mstar-server`` in ``test/rust/test_rust_frontend.py``."""
    pytest.importorskip("mstar_rust")
    from mstar.api_server import entrypoint
    from mstar.communication.rust_communicator import RustZMQCommunicator

    old = os.umask(0o002)
    try:
        sockets = tmp_path / "mstar_user_8000"
        uploads = tmp_path / "mstar_uploads_user"
        os.makedirs(sockets, exist_ok=True)
        uploads.mkdir(parents=True, exist_ok=True)
        assert _mode(sockets) == _mode(uploads) == 0o775

        monkeypatch.setattr(entrypoint, "PreprocessWorker", lambda **_: (_ for _ in ()).throw(_ReachedError))
        with pytest.raises(_ReachedError):
            entrypoint.APIServer(upload_dir=f"{uploads}/")
        assert _mode(uploads) == 0o700

        comm = RustZMQCommunicator(DEPLOYMENT_ANCHOR_ENTITY, push_ids=[], protocol=CommProtocol.IPC,
                                   ipc_socket_path_prefix=f"{sockets}/")
        try:
            assert _mode(sockets) == 0o700
            # the reason to tighten rather than accept: libzmq makes its socket
            # files under the umask, so in a 0775 dir the group could write them
            socks = [sockets / n for n in os.listdir(sockets) if n.endswith(".ipc")]
            assert socks and all(_mode(s) & 0o022 for s in socks)
        finally:
            os.close(comm._deployment_lock_fd)
    finally:
        os.umask(old)


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


def test_a_dynamo_tensor_request_tells_the_client_only_an_error_id(bad_dir):
    """Dynamo forwards the exception's message to the client, so the upload dir
    refusal must not be in it: it names the path and both uids."""
    import asyncio
    from types import SimpleNamespace

    pytest.importorskip("PIL")
    from mstar.integrations.dynamo.bridges import TENSOR_SPECS, TensorBridge

    bridge = TensorBridge(SimpleNamespace(upload_dir=bad_dir), TENSOR_SPECS["pi05"], "pi05")
    request = {"tensors": [
        _tensor("images", "Uint8", [2, 2, 2, 3], list(range(24))),
        _tensor("prompt", "Bytes", [1], [list(b"pick")]),
    ]}

    async def drive():
        async for _ in bridge.generate(request):
            pass

    with pytest.raises(Exception) as caught:  # noqa: PT011 — the type is the point below
        asyncio.run(drive())
    assert not isinstance(caught.value, PermissionError), caught.value
    message = str(caught.value)
    assert re.fullmatch(r"internal error \(error id [0-9a-f]{12}\)", message), message
    assert "refusing" not in message and "uid" not in message


def test_a_dynamo_realtime_commit_tells_the_client_only_an_error_id(bad_dir):
    """The same for a committed realtime utterance, whose audio goes through
    the upload dir: the error event reaches the client."""
    import asyncio
    from types import SimpleNamespace

    from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
    from mstar.integrations.dynamo.bridges import RealtimeBridge

    pcm = base64.b64encode(b"\x00\x01" * 80).decode()
    events = [
        {"type": "input_audio_buffer.append", "audio": pcm},
        {"type": "input_audio_buffer.commit"},
    ]

    async def client_events():
        for event in events:
            yield event

    bridge = RealtimeBridge(SimpleNamespace(upload_dir=bad_dir), ADAPTER_REGISTRY["qwen3_omni"], "q3o")

    async def collect():
        return [e async for e in bridge.generate(client_events(), SimpleNamespace(
            is_stopped=lambda: False, is_killed=lambda: False, stop_generating=lambda: None))]

    errors = [e for e in asyncio.run(collect()) if e.get("type") == "error"]
    assert errors, "the commit was accepted"
    message = errors[0]["error"]["message"]
    assert re.fullmatch(r"internal error \(error id [0-9a-f]{12}\)", message), message
    assert errors[0]["error"]["type"] == "server_error"
    assert "refusing" not in message and "uid" not in message


def test_a_streamed_transcription_sends_an_error_event_for_a_refused_upload_dir(bad_dir):
    """After the 200 the refusal has to go in band, with an id and no path:
    each window's WAV is written through the upload dir."""
    from mstar.api_server.openai.serving_transcriptions import _cutting_error_event

    try:
        ensure_private_dir(bad_dir)
    except PermissionError as exc:
        event = _cutting_error_event(exc)
    assert "refusing" not in event and str(bad_dir).rstrip("/") not in event
    assert re.search(r"internal error \(error id [0-9a-f]{12}\)", event), event
    assert '"status": 500' in event or "'status': 500" in event or "500" in event


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
        # sizing torch's pool would outlive the test in this process
        m.setattr(entrypoint, "_set_preprocess_threads", lambda model, config: None)
        # and so would handing pytest's log handlers to a writer thread, which repeats every later record
        m.setattr(entrypoint, "log_off_thread", lambda: None)
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
        # main() would hand pytest's log handlers to a writer thread, which repeats every later record
        m.setattr("mstar.utils.logging_config.log_off_thread", lambda: None)
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


def test_reading_the_entry_points_defaults_leaves_no_log_writer_behind(monkeypatch, tmp_path):
    """``_upload_dirs`` runs the real ``main()`` of two entry points. Without
    patching ``log_off_thread`` out, each takes the root logger's handlers (so
    pytest's) off onto a writer thread that outlives the test and repeats every
    later record, which made ``test_startup_warmup`` see each warmup line more
    than once on the same worker."""
    import threading

    before = {t for t in threading.enumerate()}
    root = logging.getLogger()
    handlers = root.handlers[:]
    _upload_dirs(monkeypatch, tmp_path)
    left = [t.name for t in threading.enumerate() if t not in before]
    assert left == [], left
    assert root.handlers == handlers


@pytest.mark.parametrize("name, kept", [("alice", True), ("a" * 32, True), ("a" * 33, False),
                                        ("\u00e9" * 30, False), ("a b", False), ("", False)])
def test_the_upload_dir_default_holds_the_user_name_only_when_short_and_ascii(monkeypatch, name, kept):
    # the same rule as the socket prefix; the Rust twin is runtime_dir::upload_dir_for
    from mstar.utils import socket_prefix

    monkeypatch.setattr(socket_prefix.getpass, "getuser", lambda: name)
    assert socket_prefix.default_upload_dir() == f"/tmp/mstar_uploads_{name if kept else os.getuid()}/"


@pytest.fixture(params=["other_uid", "symlink", "not_a_dir"])
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
    elif request.param == "not_a_dir":
        path.write_bytes(b"not a dir")
    else:
        path.mkdir()
        path.chmod(0o775)  # chmod, so not the umask's
        uid = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: uid + 1)
    return path, target


def _written(path, target) -> list:
    """What the refused upload left at the directory's name: its entries, or
    nothing for the non-directory planted there, which keeps its own bytes."""
    where = target if target is not None else path
    if not where.is_dir():
        return [] if where.read_bytes() == b"not a dir" else ["written through"]
    return sorted(os.listdir(where))


def test_generate_rechecks_the_upload_dir_per_request(swapped_upload_dir, monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from mstar.api_server import entrypoint

    path, target = swapped_upload_dir
    monkeypatch.setattr(entrypoint, "api_server", SimpleNamespace(upload_dir=path))
    client = TestClient(entrypoint.app, raise_server_exceptions=False)
    r = client.post("/generate", data={"text": "hi"}, files=[("files", ("a.png", b"img", "image/png"))])
    # the refusal names the dir and both uids, so the client gets an id only
    assert r.status_code == 500
    assert re.fullmatch(r"internal error \(error id [0-9a-f]{12}\)", r.json()["detail"]), r.text
    assert str(path) not in r.text and "uid" not in r.text
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

    monkeypatch.setattr(entrypoint, "api_server", SimpleNamespace(upload_dir=path, takes_stream_input=False))
    payload = base64.b64encode(b"img").decode()
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "r1", "files": [{"name": "a.png", "data": payload}]}))
        reply = json.loads(ws.receive_text())
    assert reply["request_id"] == "r1" and reply["status"] == 500
    assert re.fullmatch(r"internal error \(error id [0-9a-f]{12}\)", reply["error"]), reply
    assert str(path) not in json.dumps(reply)
    assert _written(path, target) == []


def test_a_refused_upload_dir_is_logged_in_full_and_sent_as_an_id(swapped_upload_dir, monkeypatch, caplog):
    """The operator needs the path and the owner uid; the client must not have
    them. The id in the reply matches the one in the log line."""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from mstar.api_server import entrypoint

    path, _target = swapped_upload_dir
    monkeypatch.setattr(entrypoint, "api_server", SimpleNamespace(upload_dir=path))
    client = TestClient(entrypoint.app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger=entrypoint.logger.name):
        r = client.post("/generate", data={"text": "hi"},
                        files=[("files", ("a.png", b"img", "image/png"))])
    error_id = re.search(r"error id ([0-9a-f]{12})", r.json()["detail"]).group(1)
    assert f"error id {error_id}" in caplog.text
    assert "refusing" in caplog.text and str(path) in caplog.text


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
