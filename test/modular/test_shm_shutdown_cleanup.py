"""A worker stopped with requests in flight unlinks the tensor files it owns.

A sent tensor's file stays in the shm dir until the consumer's TENSOR_RECEIVED
or the conductor's REMOVE_REQUEST. A server stopped with requests in flight
sends neither, so each worker's in-flight files stayed in ``/dev/shm`` (RAM)
until reboot. The worker processes here run the real
``_worker_process_target`` (its SIGTERM handling and orphan watchdog) around a
stand-in ``Worker``. Files are counted under the test deployment's namespace
only, as the e2e stress leak check does.

Every served deployment runs under a launch token (the API server mints it),
so a server restarted on the same prefix never writes the names an old worker
still tracks, and an orphaned worker can unlink its own files too (#767).
"""

import contextlib
import importlib.util
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import mstar.conductor.conductor as conductor_mod
from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    SharedMemoryCommunicationManager,
    _deployment_namespace,
    create_tensor_communication_manager,
    ensure_deployment_token,
)
from mstar.conductor.conductor import Conductor, _run_worker
from mstar.distributed.base import ShardingConfig

N_IN_FLIGHT = 9
ARENA_SPILLS = 3  # of 6 tensors of 300 KB, one 1 MiB segment holds three
TOKEN_ENV = "MSTAR_DEPLOYMENT_TOKEN"
MINTED_ENV = "MSTAR_DEPLOYMENT_TOKEN_MINTED"


@pytest.fixture(autouse=True)
def _launch_token(monkeypatch):
    """Each test is one launch under an API server, which mints the token (an
    APIServer built earlier in this process may have left another)."""
    monkeypatch.setenv(TOKEN_ENV, "a1" * 8)


class _DeploymentCommunicator(BaseCommunicator):
    """No mesh traffic; the protocol and prefix give the file namespace."""

    protocol = CommProtocol.IPC

    def __init__(self, prefix: str):
        self.ipc_socket_path_prefix = prefix

    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []


def _namespace_files(shm_dir, prefix) -> list[str]:
    ns = _deployment_namespace(_DeploymentCommunicator(prefix))
    return sorted(n for n in os.listdir(shm_dir) if n.startswith(f"mstar_{ns}_"))


def _send_unacked(mgr, n: int, make=lambda i: torch.tensor([i])) -> list[str]:
    """Send one tensor per request and never ack it, as a worker mid-decode
    holds them; returns the files' paths."""
    cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
    cfg.setup({})
    paths = []
    for i in range(n):
        rid = f"req{i}"
        mgr.register_request(rid, cfg)
        [info] = mgr.store_and_return_tensor_info(rid, {"out": [make(i)]})["out"]
        mgr.increment_ref(info.uuid, 1)  # the consumer's pending TENSOR_RECEIVED
        mgr.register_for_send(rid, [info])
        paths.append(mgr._shm_path(mgr.my_entity_id, info.uuid))
    return paths


class _InFlightWorker:
    """Stands in for ``Worker``: a file-transport manager holding unacked
    per-request tensors, and a ``run()`` that serves until stopped."""

    shm_dir = arena = ready = None

    def __init__(self, *, worker_id, socket_path_prefix, tensor_comm_protocol, **_):
        cls = type(self)
        if cls.arena:
            os.environ.update(MSTAR_SHM_ARENA="1", MSTAR_SHM_ARENA_SEGMENT_MB="1",
                              MSTAR_SHM_ARENA_MAX_SEGMENTS="1", MSTAR_SHM_ARENA_SPILL_AFTER_S="0.01")
        self.tensor_manager = create_tensor_communication_manager(
            tensor_comm_protocol, my_entity_id=worker_id, hostname="localhost", device="cpu",
            communicator=_DeploymentCommunicator(socket_path_prefix), shm_dir=cls.shm_dir,
        )
        if cls.arena:
            _send_unacked(self.tensor_manager, 6, make=lambda i: torch.full((300_000,), i, dtype=torch.uint8))
        else:
            _send_unacked(self.tensor_manager, N_IN_FLIGHT)

    def run(self):
        type(self).ready.set()
        while True:
            time.sleep(0.05)


class _StuckWorker(_InFlightWorker):
    """A main thread no stop signal reaches, as one stuck in a C call: only the
    watchdog's hard exit ends it."""

    def run(self):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        type(self).ready.set()
        time.sleep(600)


def _spawned_worker(shm_dir: str, prefix: str, arena: bool, ready,
                    worker_cls=_InFlightWorker, freeze_before_unlink=False) -> None:
    import mstar.worker.worker
    from mstar.conductor.conductor import _worker_process_target

    _InFlightWorker.shm_dir, _InFlightWorker.arena, _InFlightWorker.ready = shm_dir, arena, ready
    mstar.worker.worker.Worker = worker_cls
    if freeze_before_unlink:
        # A stall right after the worker decided to unlink (SIGSTOP freezes
        # its watchdog thread too), until the test sends SIGCONT. Then every
        # unlink runs, whatever the watchdog sends meanwhile.
        unlink = SharedMemoryCommunicationManager.unlink_owned_files

        def _frozen_then_unlink(self):
            os.kill(os.getpid(), signal.SIGSTOP)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            unlink(self)

        SharedMemoryCommunicationManager.unlink_owned_files = _frozen_then_unlink
    _worker_process_target(
        worker_id="worker_0", worker_ids=["worker_0"], my_worker_graphs=[], model_config={},
        all_worker_graph_ids_to_graph_walks={}, all_worker_graph_ids_to_nodes={},
        all_worker_graph_ids_to_dyn_loops={}, sharding_config=None, parallel_groups=None,
        hostname="localhost", socket_path_prefix=prefix, dist_init_method="", device="cpu",
        tensor_comm_protocol=CommProtocol.SHM,
    )


def _conductor_stand_in(shm_dir: str, prefix: str, ready, worker_pid, *worker_args) -> None:
    """The worker's parent, so the test can kill it out from under the worker."""
    p = mp.get_context("spawn").Process(target=_spawned_worker, args=(shm_dir, prefix, False, ready, *worker_args))
    p.start()
    worker_pid.put(p.pid)
    p.join()


def _restarted_worker(shm_dir: str, prefix: str, paths, done) -> None:
    """``worker_0`` of a server restarted on the same prefix: its uuid counter
    starts over, so it writes the uuids the old ``worker_0`` wrote."""
    mgr = create_tensor_communication_manager(
        CommProtocol.SHM, my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=_DeploymentCommunicator(prefix), shm_dir=shm_dir,
    )
    paths.put(_send_unacked(mgr, N_IN_FLIGHT))
    done.wait(120)


def _dirs(tmp_path) -> tuple[str, str]:
    shm_dir, prefix = str(tmp_path / "shm"), str(tmp_path / "ipc")
    os.makedirs(shm_dir)
    return shm_dir, prefix


def _state(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return None


def _gone(pid: int) -> bool:
    return _state(pid) in (None, "Z")


def _wait_for(done, what: str, timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while not done():
        assert time.monotonic() < deadline, what
        time.sleep(0.05)


@pytest.mark.parametrize("arena", [
    False,
    pytest.param(True, id="arena-spill", marks=pytest.mark.skipif(
        importlib.util.find_spec("mstar_rust") is None, reason="needs mstar_rust")),
])
@pytest.mark.parametrize("how", ["shutdown", "ctrl-c"])
def test_a_worker_stopped_with_requests_in_flight_leaves_no_tensor_files(tmp_path, how, arena):
    shm_dir, prefix = _dirs(tmp_path)
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_spawned_worker, args=(shm_dir, prefix, arena, ready))
    p.start()
    try:
        assert ready.wait(120), "worker did not start"
        held = _namespace_files(shm_dir, prefix)
        assert len(held) == (ARENA_SPILLS if arena else N_IN_FLIGHT), held
        if how == "ctrl-c":
            # The whole process group gets the SIGINT, so the conductor's
            # SIGTERM follows at once.
            os.kill(p.pid, signal.SIGINT)
    finally:
        Conductor.shutdown(SimpleNamespace(_worker_processes=[p]))
        if arena:
            # its segments' namespace lock, which outlives the deployment until a later startup sweeps it
            ns = _deployment_namespace(_DeploymentCommunicator(prefix))
            with contextlib.suppress(FileNotFoundError):
                os.unlink(f"/dev/shm/mstar_{ns}.lock")
    assert _namespace_files(shm_dir, prefix) == []


def test_without_a_launch_token_a_stopped_worker_leaves_its_files(tmp_path, monkeypatch):
    """A Conductor built directly, not under an API server: the names repeat
    across launches, and no check of the conductor covers the unlinks after
    it (#767)."""
    monkeypatch.delenv(TOKEN_ENV)
    shm_dir, prefix = _dirs(tmp_path)
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_spawned_worker, args=(shm_dir, prefix, False, ready))
    p.start()
    try:
        assert ready.wait(120), "worker did not start"
        held = _namespace_files(shm_dir, prefix)
        assert len(held) == N_IN_FLIGHT
    finally:
        Conductor.shutdown(SimpleNamespace(_worker_processes=[p]))
    assert _namespace_files(shm_dir, prefix) == held


def test_each_launch_mints_its_own_token_unless_the_launcher_set_one(monkeypatch):
    """A server built again in this process, or started by a process that
    inherited the token, would otherwise write the old launch's names."""
    monkeypatch.delenv(TOKEN_ENV)
    monkeypatch.setenv(MINTED_ENV, "")
    first = ensure_deployment_token()
    assert os.environ[TOKEN_ENV] == first
    second = ensure_deployment_token()
    assert second != first and os.environ[TOKEN_ENV] == second
    monkeypatch.setenv(TOKEN_ENV, "c3" * 8)
    assert ensure_deployment_token() == "c3" * 8


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
@pytest.mark.parametrize("token", [True, False], ids=["launch-token", "no-token"])
def test_an_orphaned_worker_unlinks_its_files_only_when_no_restart_can_reuse_the_names(tmp_path, monkeypatch, token):
    """With its conductor gone the deployment's lock and ports are free. Under a
    launch token a restart writes other names, so the worker unlinks its own;
    without one (a Conductor built directly) a restart on the same prefix may
    already be writing these names, so it leaves them, as it does on any stop."""
    if not token:
        monkeypatch.delenv(TOKEN_ENV)
    shm_dir, prefix = _dirs(tmp_path)
    ctx = mp.get_context("spawn")
    ready, worker_pid = ctx.Event(), ctx.Queue()
    conductor = ctx.Process(target=_conductor_stand_in, args=(shm_dir, prefix, ready, worker_pid))
    conductor.start()
    try:
        assert ready.wait(120), "worker did not start"
        pid = worker_pid.get(timeout=10)
        held = _namespace_files(shm_dir, prefix)
        assert len(held) == N_IN_FLIGHT
    finally:
        conductor.kill()
        conductor.join(timeout=10)
    # the worker's watchdog stops it once the conductor is gone
    _wait_for(lambda: _gone(pid), "orphaned worker did not exit")
    assert _namespace_files(shm_dir, prefix) == ([] if token else held)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_worker_stalled_mid_cleanup_leaves_a_restarted_servers_files_alone(tmp_path, monkeypatch):
    """#767: the stopping worker decides to unlink and stalls; its conductor
    dies, freeing the prefix; a server restarted there writes ``worker_0``'s
    uuids again; the old worker resumes and unlinks what it tracks."""
    shm_dir, prefix = _dirs(tmp_path)
    ctx = mp.get_context("spawn")
    ready, worker_pid, paths, done = ctx.Event(), ctx.Queue(), ctx.Queue(), ctx.Event()
    conductor = ctx.Process(target=_conductor_stand_in,
                            args=(shm_dir, prefix, ready, worker_pid, _InFlightWorker, True))
    conductor.start()
    restarted = pid = None
    try:
        assert ready.wait(120), "worker did not start"
        pid = worker_pid.get(timeout=10)
        old = [os.path.join(shm_dir, n) for n in _namespace_files(shm_dir, prefix)]
        assert len(old) == N_IN_FLIGHT
        os.kill(pid, signal.SIGTERM)  # the conductor's shutdown
        _wait_for(lambda: _state(pid) == "T", "the worker did not stop before its unlinks")
        conductor.kill()  # e.g. the API server's SIGTERM after its 5 s join
        conductor.join(timeout=10)

        monkeypatch.setenv(TOKEN_ENV, "b2" * 8)  # the restarted API server mints its own
        restarted = ctx.Process(target=_restarted_worker, args=(shm_dir, prefix, paths, done))
        restarted.start()
        new = paths.get(timeout=120)
        os.kill(pid, signal.SIGCONT)
        _wait_for(lambda: _gone(pid), "the old worker did not exit")

        assert [p for p in new if not os.path.exists(p)] == []
        assert [p for p in old if os.path.exists(p)] == []
    finally:
        if pid is not None and not _gone(pid):
            os.kill(pid, signal.SIGKILL)
        conductor.kill()
        done.set()
        if restarted is not None:
            restarted.join(timeout=30)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_an_orphaned_worker_the_watchdog_exits_hard_still_unlinks_its_files(tmp_path):
    """A main thread that never serves the watchdog's SIGTERM ends in its
    ``os._exit`` after the 5 s grace, which skips every ``finally``."""
    shm_dir, prefix = _dirs(tmp_path)
    ctx = mp.get_context("spawn")
    ready, worker_pid = ctx.Event(), ctx.Queue()
    conductor = ctx.Process(target=_conductor_stand_in, args=(shm_dir, prefix, ready, worker_pid, _StuckWorker))
    conductor.start()
    try:
        assert ready.wait(120), "worker did not start"
        pid = worker_pid.get(timeout=10)
        assert len(_namespace_files(shm_dir, prefix)) == N_IN_FLIGHT
    finally:
        conductor.kill()
        conductor.join(timeout=10)
    _wait_for(lambda: _gone(pid), "the watchdog did not exit the worker")
    assert _namespace_files(shm_dir, prefix) == []


def test_unlink_owned_files_skips_what_is_gone_and_leaves_peers_alone(tmp_path):
    shm_dir, prefix = _dirs(tmp_path)

    def manager(entity_id):
        return SharedMemoryCommunicationManager(
            my_entity_id=entity_id, hostname="localhost", device="cpu",
            communicator=_DeploymentCommunicator(prefix), shm_dir=shm_dir,
        )

    mine, peer = manager("worker_0"), manager("worker_1")
    already_gone, *_ = _send_unacked(mine, 3)
    peer_files = sorted(os.path.basename(p) for p in _send_unacked(peer, 2))
    os.unlink(already_gone)

    mine.unlink_owned_files()
    mine.unlink_owned_files()
    assert _namespace_files(shm_dir, prefix) == peer_files


def test_a_stop_before_an_unlink_leaves_the_file_for_the_retry(tmp_path, monkeypatch):
    shm_dir, prefix = _dirs(tmp_path)
    mine = SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=_DeploymentCommunicator(prefix), shm_dir=shm_dir,
    )
    _send_unacked(mine, 3)
    unlink, stops = os.unlink, []

    def _stopped_once(path):
        if not stops:
            stops.append(path)
            raise SystemExit(0)  # the SIGTERM handler, firing before the unlink
        unlink(path)

    monkeypatch.setattr(os, "unlink", _stopped_once)
    with pytest.raises(SystemExit):
        mine.unlink_owned_files()
    mine.unlink_owned_files()
    assert _namespace_files(shm_dir, prefix) == []


def test_the_watchdog_and_the_stopping_worker_each_unlink_a_file_once(tmp_path, monkeypatch):
    """The orphan watchdog's thread drains the same files while the worker's
    main thread is inside an unlink."""
    shm_dir, prefix = _dirs(tmp_path)
    mine = SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=_DeploymentCommunicator(prefix), shm_dir=shm_dir,
    )
    _send_unacked(mine, 3)
    unlink, unlinked = os.unlink, []

    def _watchdog_drains_during_the_first(path):
        unlinked.append(path)
        if len(unlinked) == 1:
            watchdog = threading.Thread(target=mine.unlink_owned_files)
            watchdog.start()
            watchdog.join()
        unlink(path)

    monkeypatch.setattr(os, "unlink", _watchdog_drains_during_the_first)
    mine.unlink_owned_files()
    assert len(unlinked) == len(set(unlinked)) == 3
    assert _namespace_files(shm_dir, prefix) == []


def test_a_stop_during_cleanup_neither_cuts_the_unlinks_nor_hides_the_workers_error(monkeypatch):
    """A worker under a live conductor and a launch token whose ``run()``
    raised, hit by the conductor's SIGTERM while it unlinks."""
    monkeypatch.setattr(signal, "signal", lambda *a: None)  # keep this process's handlers
    monkeypatch.setattr(conductor_mod.mp, "parent_process", lambda: SimpleNamespace(is_alive=lambda: True))
    calls = []

    class _Manager:
        names_unique_per_launch = True

        def unlink_owned_files(self):
            calls.append("unlink")
            if len(calls) == 1:
                raise SystemExit(0)  # the conductor's SIGTERM

    class _Crashed:
        tensor_manager = _Manager()

        def run(self):
            raise RuntimeError("step failed")

    with pytest.raises(RuntimeError, match="step failed"):
        _run_worker(_Crashed())
    assert calls == ["unlink", "unlink"]


def test_a_launcher_set_token_is_kept_with_a_warning(monkeypatch, caplog):
    """A fixed value (a systemd ``Environment=`` line, a pod template shared by
    replicas) comes back on every restart, and with it #767's race; it is kept,
    since separately started processes must agree on it, but said out loud."""
    monkeypatch.setenv(MINTED_ENV, "")
    with caplog.at_level("WARNING", logger="mstar.communication.tensors"):
        assert ensure_deployment_token() == "a1" * 8
    assert [r for r in caplog.records if TOKEN_ENV in r.getMessage() and "new for every launch" in r.getMessage()]
    caplog.clear()
    monkeypatch.delenv(TOKEN_ENV)
    with caplog.at_level("WARNING", logger="mstar.communication.tensors"):
        minted = ensure_deployment_token()
        assert ensure_deployment_token() != minted  # an inherited minted one is replaced
    assert not caplog.records


def test_a_stopped_worker_gets_its_signal_handlers_back_for_teardown():
    """The unlinks run with SIGTERM and SIGINT ignored; teardown after them
    does not, so a second stop still ends a teardown that hangs."""
    calls = []

    def _term(_signum, _frame):
        raise SystemExit(0)

    class _Manager:
        names_unique_per_launch = True

        def unlink_owned_files(self):
            calls.append((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)))

    class _Stopped:
        tensor_manager = _Manager()

        def run(self):
            raise SystemExit(0)

    old = signal.signal(signal.SIGTERM, _term), signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(SystemExit):
            _run_worker(_Stopped())
        assert calls == [(signal.SIG_IGN, signal.SIG_IGN)]
        assert signal.getsignal(signal.SIGTERM) is _term
        assert signal.getsignal(signal.SIGINT) is old[1]
    finally:
        signal.signal(signal.SIGTERM, old[0])
        signal.signal(signal.SIGINT, old[1])


class _HungTeardownWorker(_InFlightWorker):
    """Stops on SIGTERM, then hangs in teardown: ``threading._shutdown`` joins a
    thread that never ends, so the conductor's SIGKILL ends the process and the
    arena's exit-time finalizer never runs."""

    def run(self):
        threading.Thread(target=time.sleep, args=(600,), daemon=False).start()
        super().run()


@pytest.mark.skipif(importlib.util.find_spec("mstar_rust") is None, reason="needs mstar_rust")
def test_a_worker_killed_in_a_hung_teardown_leaves_no_arena_segments(tmp_path):
    shm_dir, prefix = _dirs(tmp_path)
    ns = _deployment_namespace(_DeploymentCommunicator(prefix))

    def _segments():
        return sorted(n for n in os.listdir("/dev/shm") if n.startswith(f"mstar_{ns}_arena_"))

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_spawned_worker, args=(shm_dir, prefix, True, ready, _HungTeardownWorker))
    p.start()
    try:
        assert ready.wait(120), "worker did not start"
        assert _segments(), "the arena grew no segment"
        Conductor.shutdown(SimpleNamespace(_worker_processes=[p]))
        assert p.exitcode == -signal.SIGKILL, "the teardown did not hang"
        left = _segments()
    finally:
        if p.is_alive():
            p.kill()
        for n in _segments():
            os.unlink(f"/dev/shm/{n}")
        with contextlib.suppress(FileNotFoundError):
            os.unlink(f"/dev/shm/mstar_{ns}.lock")
    assert left == []
    assert _namespace_files(shm_dir, prefix) == []


def test_a_stop_while_the_handlers_are_saved_still_unlinks_and_keeps_the_error(monkeypatch):
    """The conductor's SIGTERM right after a Ctrl-C can land while ``_run_worker``
    reads the handlers it will put back; the unlinks still run, and so does a
    stop right after the handlers are back, without hiding ``run()``'s error."""
    getsignal, setsignal = signal.getsignal, signal.signal
    stops = {"get": 1, "restore": 1}
    calls = []

    def _get(s):
        if stops["get"]:
            stops["get"] -= 1
            raise SystemExit(0)    # the SIGTERM, mid-snapshot
        return getsignal(s)

    def _set(s, h):
        setsignal(s, h)
        if h is not signal.SIG_IGN and stops["restore"]:
            stops["restore"] -= 1
            raise SystemExit(0)    # another, right after the first handler is back

    class _Manager:
        names_unique_per_launch = True

        def unlink_owned_files(self):
            calls.append("unlink")

    class _Crashed:
        tensor_manager = _Manager()

        def run(self):
            raise RuntimeError("step failed")

    old = getsignal(signal.SIGTERM), getsignal(signal.SIGINT)
    monkeypatch.setattr(signal, "getsignal", _get)
    monkeypatch.setattr(signal, "signal", _set)
    try:
        with pytest.raises(RuntimeError, match="step failed"):
            _run_worker(_Crashed())
        assert calls == ["unlink"] and stops == {"get": 0, "restore": 0}
    finally:
        setsignal(signal.SIGTERM, old[0])
        setsignal(signal.SIGINT, old[1])
