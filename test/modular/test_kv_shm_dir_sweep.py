"""KV snapshot directories when deployments share a tmpfs (#608).

Containers that share ``/dev/shm`` (``--ipc=host``, k8s hostIPC) do not see
each other's pids, and a container's conductor often has the same low pid as
its neighbour's. A directory named and swept by conductor pid was therefore
removed while in use, or shared by two deployments. The directory now lives
under the deployment namespace, so the namespace lock decides.
"""

import os
import signal
import subprocess
import sys

import pytest

from mstar.communication import tensors
from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.engine.resources.kv.transfer import make_deployment_kv_shm_dir

# A worker of a deployment: joins the namespace's KV directory, writes a
# snapshot, reports the path and stays alive until stdin closes or it is killed.
_WORKER = """
import os, sys
os.environ["MSTAR_KV_SHM_DIR"] = sys.argv[1]
from mstar.engine.resources.kv.transfer import make_deployment_kv_shm_dir
path = make_deployment_kv_shm_dir(sys.argv[2])
open(os.path.join(path, "snapshot.pt"), "wb").close()
print(path, flush=True)
sys.stdin.read()
"""


@pytest.fixture
def kv_base(tmp_path, monkeypatch):
    monkeypatch.setenv("MSTAR_KV_SHM_DIR", str(tmp_path))
    held: dict[str, int] = {}
    monkeypatch.setattr(tensors, "_held_namespace_locks", held)
    yield tmp_path
    for fd in held.values():
        os.close(fd)


@pytest.fixture
def start_worker():
    procs = []

    def start(base, namespace):
        proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(base), namespace],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        procs.append(proc)
        path = proc.stdout.readline().strip()
        assert path, f"worker for {namespace} failed to start"
        return proc, path

    yield start
    for proc in procs:
        proc.kill()
        proc.wait()


def test_a_live_deployments_directory_survives_and_a_killed_ones_is_swept(
    kv_base, start_worker,
):
    """Nothing here depends on pids, so it holds across PID namespaces."""
    live, live_dir = start_worker(kv_base, "a" * 12)
    killed, killed_dir = start_worker(kv_base, "c" * 12)
    killed.send_signal(signal.SIGKILL)
    killed.wait()

    own = make_deployment_kv_shm_dir("b" * 12)

    assert own not in (live_dir, killed_dir)
    assert os.path.exists(os.path.join(live_dir, "snapshot.pt")), (
        "another deployment's start removed a live deployment's KV snapshots")
    assert not os.path.exists(killed_dir)
    assert not (kv_base / f"mstar_{'c' * 12}.lock").exists()
    assert sorted(os.listdir(kv_base)) == sorted([
        f"mstar_{'a' * 12}.lock", f"mstar_{'a' * 12}_kv",
        f"mstar_{'b' * 12}.lock", f"mstar_{'b' * 12}_kv",
    ])


def test_every_process_of_one_deployment_joins_one_directory(
    kv_base, start_worker,
):
    """And the directory stays while any one of them runs."""
    other, path = start_worker(kv_base, "a" * 12)
    assert make_deployment_kv_shm_dir("a" * 12) == path
    assert os.stat(path).st_mode & 0o777 == 0o700
    other.send_signal(signal.SIGKILL)
    other.wait()

    _, elsewhere = start_worker(kv_base, "e" * 12)
    assert os.path.exists(os.path.join(path, "snapshot.pt"))
    assert elsewhere != path


def test_an_older_builds_pid_named_directory_is_left_alone(kv_base):
    """Seen from another PID namespace, a live older build's pid looks dead."""
    older = kv_base / f"mstar_kv_{os.getuid()}_999999999_{'1' * 16}"
    older.mkdir()
    (older / "snapshot.pt").touch()
    make_deployment_kv_shm_dir("b" * 12)
    assert (older / "snapshot.pt").exists()


class _Communicator(BaseCommunicator):
    protocol = CommProtocol.IPC
    ipc_socket_path_prefix = "/tmp/mstar/"

    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []

    def register_event_for_poll(self, event):
        pass

    def poll_for_messages(self, timeout_ms=20):
        return False


def _start_transport(shm_dir):
    return SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=_Communicator(), shm_dir=str(shm_dir),
    )


def test_the_tensor_transport_sweep_removes_a_dead_kv_directory(kv_base):
    """CUDA workers never make a KV directory but share the base with the
    tensor files, so their startup sweep is the one that finds it. A lock
    file no process holds is what a killed deployment leaves."""
    dead = kv_base / f"mstar_{'d' * 12}_kv"
    dead.mkdir()
    (dead / "snapshot.pt").touch()
    (kv_base / f"mstar_{'d' * 12}.lock").touch()

    _start_transport(kv_base)

    assert not dead.exists()
    assert not (kv_base / f"mstar_{'d' * 12}.lock").exists()


def test_a_link_under_a_dead_namespace_is_removed_not_followed(
    kv_base, tmp_path_factory,
):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "keep").touch()
    (kv_base / f"mstar_{'d' * 12}.lock").touch()
    (kv_base / f"mstar_{'d' * 12}_kv").symlink_to(outside)

    _start_transport(kv_base)

    assert not os.path.lexists(kv_base / f"mstar_{'d' * 12}_kv")
    assert (outside / "keep").exists()


def test_without_a_namespace_the_directory_is_named_and_swept_by_conductor_pid(kv_base):
    """A control channel other than IPC or TCP derives no namespace. The worker
    then falls back to the pid-named directory instead of failing to start."""
    uid = os.getuid()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = kv_base / f"mstar_kv_{uid}_{dead.pid}_{'0' * 16}"
    stale.mkdir()
    (stale / "orphan.pt").touch()
    live = kv_base / f"mstar_kv_{uid}_{os.getpid()}_{'1' * 16}"
    live.mkdir()

    path = make_deployment_kv_shm_dir("", socket_path_prefix="/tmp/mstar_x", dist_init_method="tcp://h:1",
                                      owner_pid=os.getpid())
    assert os.path.dirname(path) == str(kv_base)
    assert os.path.basename(path).startswith(f"mstar_kv_{uid}_{os.getpid()}_")
    assert os.stat(path).st_mode & 0o777 == 0o700
    # another worker of the same deployment joins it; another deployment does not
    assert make_deployment_kv_shm_dir("", socket_path_prefix="/tmp/mstar_x", dist_init_method="tcp://h:1",
                                      owner_pid=os.getpid()) == path
    assert make_deployment_kv_shm_dir("", socket_path_prefix="/tmp/mstar_y", dist_init_method="tcp://h:1",
                                      owner_pid=os.getpid()) != path
    assert not stale.exists() and live.exists()
    assert not [n for n in os.listdir(kv_base) if n.endswith(".lock")]   # no namespace lock taken
