"""Compile-cache bundles: the Inductor, Triton and MIOpen caches a warmed-up
server leaves behind, packed per software stack so that a new host or
container with the same stack starts warm instead of compiling every capture
bucket again.

A bundle is a directory named by its stack key, under a root that may hold
bundles for several stacks::

    <root>/<product>/torch-<torch>-<hip|cuda>-<version>-triton-<triton>/
        manifest.json        the key, what each tar holds, and who added to it
        inductor.tar         TORCHINDUCTOR_CACHE_DIR, less a Triton dir inside it
        triton.tar           TRITON_CACHE_DIR
        miopen.tar           MIOPEN_USER_DB_PATH (ROCm)
        miopen-kernels.tar   MIOPEN_CUSTOM_CACHE_DIR, when it is not the same dir

Each tar holds its directory's files as ``./<path>`` and is uncompressed, so
packing into an existing bundle appends only the files it lacks or holds an
older copy of, and unpacking keeps the last copy of each file. The layout and
the matching rules are in docs/compile_cache.rst.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import importlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
# written into the Inductor dir by unpack and read at server startup
MARKER = "mstar-bundle.json"
LOCK = ".append.lock"
KINDS = ("inductor", "triton", "miopen", "miopen-kernels")
_KEY_DIR = re.compile(r"torch-(?P<torch>.+)-(?P<runtime>(?:hip|cuda)-[^-]+)-triton-(?P<triton>[^/]+)")


class BundleMismatchError(RuntimeError):
    """No bundle under the given path was built for this host's stack."""


@dataclass(frozen=True)
class StackKey:
    """The stack a bundle's entries were compiled on.

    Inductor and Triton key their own entries on the GPU arch and the torch,
    HIP or CUDA, and Triton builds, so a bundle from another stack unpacks
    harmlessly but is never read: those fields must match. The product is in
    the key because autotuning picks are timings taken on it. An MI300X's
    picks load on an MI325X (both gfx942) but were not tuned there.
    """

    product: str
    arch: str | None
    torch: str
    runtime: str  # "hip-7.2.53211" or "cuda-12.8"
    triton: str

    @property
    def path(self) -> str:
        return f"{self.product}/torch-{self.torch}-{self.runtime}-triton-{self.triton}"

    @classmethod
    def from_path(cls, product: str, rest: str) -> StackKey | None:
        """The key of a bundle directory ``<product>/<rest>`` that has no
        manifest; such a directory doesn't record its arch."""
        m = _KEY_DIR.fullmatch(rest)
        if m is None:
            return None
        return cls(product=product, arch=None, **m.groupdict())

    def differences(self, other: StackKey) -> list[str]:
        """What keeps ``other``'s entries from being read on this stack. A field
        either side doesn't know (a manifest-less bundle's arch) is no reason."""
        out = []
        for name in ("arch", "torch", "runtime", "triton"):
            here, there = getattr(self, name), getattr(other, name)
            if here and there and here != there:
                out.append(f"{name} {there} (here {here})")
        return out


def stack_key(device: int | None = None) -> StackKey:
    """This process's key, from ``device`` (the current GPU by default)."""
    import torch
    import triton

    from mstar.platforms import current_platform

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA or ROCm GPU is visible; a bundle is keyed by the GPU that compiled it")
    name = torch.cuda.get_device_name(device)
    product = re.sub(r"[^a-z0-9]+", "-", name.lower().replace("amd instinct", "")).strip("-")
    runtime = f"hip-{torch.version.hip}" if torch.version.hip else f"cuda-{torch.version.cuda}"
    return StackKey(product=product, arch=current_platform.arch(device), torch=torch.__version__,
                    runtime=runtime, triton=triton.__version__)


def cache_dirs() -> dict[str, Path]:
    """Each kind's directory as a server started from this environment
    resolves it: an exported variable, else M*'s default for Inductor and
    Triton (set when :mod:`mstar.engine` is imported), else the library's own."""
    # Importing torch._inductor writes torch's default into the environment, so M*'s goes in first.
    importlib.import_module("mstar.engine")
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    from mstar.platforms import current_platform

    dirs = {
        "inductor": Path(cache_dir()),
        "triton": Path(os.environ.get("TRITON_CACHE_DIR") or Path.home() / ".triton" / "cache"),
    }
    if current_platform.is_rocm():
        db = Path(os.environ.get("MIOPEN_USER_DB_PATH") or "~/.config/miopen").expanduser()
        kernels = Path(os.environ.get("MIOPEN_CUSTOM_CACHE_DIR") or "~/.cache/miopen").expanduser()
        dirs["miopen"] = db
        if kernels.resolve() != db.resolve():
            dirs["miopen-kernels"] = kernels
    return dirs


# --- reading and writing tars -------------------------------------------------------------------

def _stamp(st: os.stat_result, link: bool) -> tuple[int, int]:
    # what a tar member records of a file: a symlink's size is 0 there
    return (0 if link else st.st_size, int(st.st_mtime))


def _files(root: Path, skip: list[Path]) -> dict[str, tuple[int, int]]:
    """Files and symlinks under ``root`` as tar names (``./a/b``), less the
    marker and anything under ``skip``."""
    skip_set = {p.resolve() for p in skip}
    out: dict[str, tuple[int, int]] = {}
    for top, dirnames, filenames in os.walk(root):
        here = Path(top)
        dirnames[:] = [d for d in dirnames if (here / d).resolve() not in skip_set]
        for d in dirnames:
            if (here / d).is_symlink():
                rel = (here / d).relative_to(root).as_posix()
                out[f"./{rel}"] = _stamp(os.lstat(here / d), True)
        for f in filenames:
            p = here / f
            rel = p.relative_to(root).as_posix()
            if rel == MARKER:
                continue
            st = os.lstat(p)
            out[f"./{rel}"] = _stamp(st, p.is_symlink())
    return out


def _members(tar: Path) -> dict[str, tuple[int, int]]:
    """A tar's file and symlink members by name; the last copy of a name wins,
    as it does on extraction."""
    out: dict[str, tuple[int, int]] = {}
    with tarfile.open(tar, "r:") as tf:
        while (m := tf.next()) is not None:
            # tarfile keeps every header it reads, and a bundle's tar can hold a million
            tf.members.clear()
            if m.isfile() or m.issym():
                name = m.name if m.name.startswith("./") else f"./{m.name}"
                out[name] = (0 if m.issym() else m.size, int(m.mtime))
    return out


def _add(tar: Path, src: Path, names: list[str]) -> None:
    """Append ``names`` (relative to ``src``) to ``tar``, or create it. A copy is
    written and renamed into place, so a reader never sees a partial tar."""
    tmp = tar.with_name(f"{tar.name}.{os.getpid()}")
    with tempfile.NamedTemporaryFile("wb", suffix=".list") as listing:
        listing.write(b"\0".join(n.encode() for n in names) + b"\0")
        listing.flush()
        try:
            if tar.exists():
                shutil.copyfile(tar, tmp)
            subprocess.run(["tar", "-rf" if tar.exists() else "-cf", str(tmp), "-C", str(src),
                            "--no-recursion", "--null", "-T", listing.name],
                           check=True, capture_output=True, text=True)
            os.replace(tmp, tar)
        except subprocess.CalledProcessError as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"tar failed adding {len(names)} files from {src} to {tar}: {exc.stderr.strip()}") from exc
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """One writer per bundle at a time, across hosts sharing the filesystem
    when its locks work (they do on local disks and NFSv4)."""
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# --- pack, unpack, find -------------------------------------------------------------------------

@dataclass(frozen=True)
class KindPack:
    kind: str
    source: str
    added: int
    members: int
    tar_bytes: int


def pack(root: Path, key: StackKey, dirs: dict[str, Path], note: str | None = None,
         log: Callable[[str], None] = print) -> tuple[Path, list[KindPack]]:
    """Add the caches in ``dirs`` to ``key``'s bundle under ``root``: new files,
    and files whose size or mtime differ from the bundle's last copy (MIOpen
    rewrites its databases in place; Inductor and Triton entries never change).
    Run it with no server writing to the caches."""
    bundle = root / key.path
    bundle.mkdir(parents=True, exist_ok=True)
    out = []
    with _locked(bundle / LOCK):
        manifest = read_manifest(bundle) or {"key": asdict(key), "created": _now(), "kinds": {}, "history": []}
        for kind, src in dirs.items():
            if not src.is_dir():
                log(f"{kind}: {src} does not exist, left out")
                continue
            # a Triton dir inside Inductor's (M*'s default) goes in its own tar
            inner = [d for k, d in dirs.items() if k != kind and d.resolve().is_relative_to(src.resolve())
                     and d.resolve() != src.resolve()]
            files = _files(src, inner)
            tar = bundle / f"{kind}.tar"
            have = _members(tar) if tar.exists() else {}
            new = sorted(n for n, stamp in files.items() if have.get(n) != stamp)
            if new:
                _add(tar, src, new)
            members = len(have.keys() | set(new))
            result = KindPack(kind, str(src), len(new), members, tar.stat().st_size if tar.exists() else 0)
            out.append(result)
            manifest["kinds"][kind] = {"members": result.members, "tar_bytes": result.tar_bytes}
            log(f"{kind}: {result.added} files added from {src}, {result.members} in {tar.name} "
                f"({result.tar_bytes / 2**30:.2f} GiB)")
        manifest["updated"] = _now()
        manifest["history"].append({
            "time": manifest["updated"], "host": socket.gethostname(), "note": note,
            "added": {r.kind: r.added for r in out}, "mstar": _mstar_commit(),
        })
        _write_json(bundle / MANIFEST, manifest)
    return bundle, out


def read_manifest(bundle: Path) -> dict | None:
    try:
        return json.loads((bundle / MANIFEST).read_text())
    except FileNotFoundError:
        return None


def bundle_key(bundle: Path) -> StackKey | None:
    """A bundle's key: its manifest's, else read off its path."""
    manifest = read_manifest(bundle)
    if manifest is not None:
        return StackKey(**manifest["key"])
    return StackKey.from_path(bundle.parent.name, bundle.name)


def is_bundle(path: Path) -> bool:
    return (path / MANIFEST).is_file() or any((path / f"{k}.tar").is_file() for k in KINDS)


def find_bundles(root: Path) -> list[Path]:
    """``root`` itself if it is a bundle, else the bundles in its
    ``<product>/<stack>/`` directories."""
    if is_bundle(root):
        return [root]
    return sorted(p for p in root.glob("*/*") if p.is_dir() and is_bundle(p))


def select_bundle(src: Path, key: StackKey) -> Path:
    """The bundle in ``src`` for ``key``: ``src`` itself when it is a bundle,
    else ``src/<key path>``. Raises BundleMismatchError naming what ``src`` holds."""
    if is_bundle(src):
        return src
    bundle = src / key.path
    if is_bundle(bundle):
        return bundle
    have = [str(p.relative_to(src)) for p in find_bundles(src)]
    raise BundleMismatchError(f"{src} has no bundle for {key.path}; it has: {', '.join(have) or 'none'}")


@dataclass(frozen=True)
class Unpacked:
    bundle: Path
    key: StackKey | None
    kinds: dict[str, str]  # kind -> directory it went to


def unpack(src: Path, key: StackKey, dirs: dict[str, Path], force: bool = False,
           log: Callable[[str], None] = print) -> Unpacked:
    """Extract the bundle for ``key`` from ``src`` into ``dirs``, over what is
    there (the bundle's copy of a file wins), and leave a marker for the
    startup line. A bundle for another stack is refused unless ``force``."""
    bundle = select_bundle(src, key)
    bkey = bundle_key(bundle)
    diff = key.differences(bkey) if bkey is not None else []
    if diff and not force:
        raise BundleMismatchError(
            f"{bundle} was built for {bkey.path}: {', '.join(diff)}; its entries would not be read")
    done: dict[str, str] = {}
    for kind in KINDS:
        tar = bundle / f"{kind}.tar"
        if not tar.is_file():
            continue
        # MIOpen's kernel cache shares the database dir unless it has one of its own here
        dest = dirs.get(kind) or (dirs.get("miopen") if kind == "miopen-kernels" else None)
        if dest is None:
            log(f"{kind}: no {kind} cache on this platform, left out")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        p = subprocess.run(["tar", "-xf", str(tar), "-C", str(dest), "--no-same-owner"],
                           capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"tar failed extracting {tar} into {dest}: {p.stderr.strip()}")
        done[kind] = str(dest)
    dirs["inductor"].mkdir(parents=True, exist_ok=True)
    _write_json(dirs["inductor"] / MARKER, {
        "bundle": str(bundle), "key": asdict(bkey) if bkey else None, "unpacked": _now(), "kinds": done,
    })
    return Unpacked(bundle, bkey, done)


def startup_line(key: StackKey, dirs: dict[str, Path]) -> tuple[bool, str]:
    """What the server logs at startup: this stack's key and whether the last
    bundle unpacked into its caches matches it. False when it doesn't."""
    try:
        info = json.loads((dirs["inductor"] / MARKER).read_text())
    except FileNotFoundError:
        return True, f"compile cache: stack {key.path}; no bundle unpacked into {dirs['inductor']}"
    except (OSError, ValueError) as exc:
        return False, f"compile cache: stack {key.path}; unreadable {MARKER} in {dirs['inductor']}: {exc}"
    head = f"compile cache: stack {key.path}; bundle {info.get('bundle')} unpacked {info.get('unpacked')}"
    if not info.get("key"):
        return True, f"{head}, stack unknown"
    bkey = StackKey(**info["key"])
    diff = key.differences(bkey)
    if diff:
        return False, f"{head} was built for {bkey.path}: {', '.join(diff)}, so its entries are not used"
    if bkey.product != key.product:
        return True, f"{head} matches the stack, tuned on {bkey.product}"
    return True, f"{head} matches"


def log_startup(who: str, device: int | None = None) -> None:
    """Log :func:`startup_line` for ``device``, as a warning when the bundle
    doesn't match. Never fails a startup."""
    try:
        ok, line = startup_line(stack_key(device), cache_dirs())
    except Exception as exc:
        logger.warning("%s: compile cache: could not read this stack's key: %s", who, exc)
        return
    (logger.info if ok else logger.warning)("%s: %s", who, line)


def _mstar_commit() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        p = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip() or None
