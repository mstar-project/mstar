"""Compile-cache bundles (mstar/utils/compile_bundle.py, ``mstar cache``): packing
appends only what is new, unpacking restores the caches and marks them, a
bundle for another stack is refused, and a bundle root laid out like the
golden copies the AMD benchmark scripts seed from (uncompressed per-kind tars
of ``./`` members, no manifest) reads as a bundle."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from mstar.cli import cache as cache_cli
from mstar.cli.main import build_parser
from mstar.utils import compile_bundle as cb

KEY = cb.StackKey(product="mi325x-vf", arch="gfx942", torch="2.12.1+rocm7.2", runtime="hip-7.2.53211",
                  triton="3.7.1")


def _write(path: Path, data: str, mtime: int = 1_700_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    os.utime(path, (mtime, mtime))


def _caches(root: Path) -> dict[str, Path]:
    """M*'s default layout: Triton's dir inside Inductor's; MIOpen beside them."""
    dirs = {"inductor": root / "inductor", "triton": root / "inductor" / "triton", "miopen": root / "miopen"}
    _write(dirs["inductor"] / "fxgraph" / "ab" / "entry", "graph")
    _write(dirs["inductor"] / "wq" / "kernel.py", "code")
    _write(dirs["triton"] / "HASH" / "k.hsaco", "binary")
    _write(dirs["miopen"] / "gfx942130.ukdb", "kernels")
    _write(dirs["miopen"] / "gfx942130.HIP.ufdb.txt", "find-db v1")
    return dirs


def _names(tar: Path) -> list[str]:
    with tarfile.open(tar) as tf:
        return [m.name for m in tf.getmembers() if m.isfile()]


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


def test_key_path_round_trips_and_names_what_differs():
    assert KEY.path == "mi325x-vf/torch-2.12.1+rocm7.2-hip-7.2.53211-triton-3.7.1"
    parsed = cb.StackKey.from_path(*KEY.path.split("/"))
    assert parsed == cb.StackKey(**{**KEY.__dict__, "arch": None})
    assert KEY.differences(parsed) == []  # a manifest-less bundle doesn't know its arch
    other = cb.StackKey("mi300x", "gfx942", KEY.torch, KEY.runtime, "3.6.0")
    assert KEY.differences(other) == ["triton 3.6.0 (here 3.7.1)"]  # the product alone is no reason
    assert cb.StackKey.from_path("x", "not-a-stack") is None


def test_pack_then_unpack_restores_the_caches(tmp_path):
    dirs = _caches(tmp_path / "built")
    bundle, results = cb.pack(tmp_path / "bundles", KEY, dirs, note="warm-up", log=lambda _: None)
    assert bundle == tmp_path / "bundles" / KEY.path
    assert {r.kind: r.added for r in results} == {"inductor": 2, "triton": 1, "miopen": 2}
    # Triton's dir inside Inductor's goes in its own tar only
    assert sorted(_names(bundle / "inductor.tar")) == ["./fxgraph/ab/entry", "./wq/kernel.py"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["key"]["arch"] == "gfx942" and manifest["history"][0]["note"] == "warm-up"

    fresh = {"inductor": tmp_path / "new" / "i", "triton": tmp_path / "new" / "t", "miopen": tmp_path / "new" / "m"}
    done = cb.unpack(tmp_path / "bundles", KEY, fresh, log=lambda _: None)
    assert done.bundle == bundle and set(done.kinds) == {"inductor", "triton", "miopen"}
    assert _tree(fresh["triton"]) == _tree(dirs["triton"])
    assert _tree(fresh["miopen"]) == _tree(dirs["miopen"])
    restored = _tree(fresh["inductor"])
    marker = json.loads(restored.pop(cb.MARKER))
    assert marker["bundle"] == str(bundle) and marker["key"]["triton"] == "3.7.1"
    assert restored == {k: v for k, v in _tree(dirs["inductor"]).items() if not k.startswith("triton/")}
    assert os.stat(fresh["triton"] / "HASH" / "k.hsaco").st_mtime == 1_700_000_000


def test_a_second_pack_appends_only_new_and_changed_files(tmp_path):
    dirs = _caches(tmp_path / "built")
    root = tmp_path / "bundles"
    bundle, _ = cb.pack(root, KEY, dirs, log=lambda _: None)
    _, again = cb.pack(root, KEY, dirs, log=lambda _: None)
    assert all(r.added == 0 for r in again)
    _write(dirs["inductor"] / "fxgraph" / "cd" / "entry", "another graph")
    _write(dirs["miopen"] / "gfx942130.HIP.ufdb.txt", "find-db v2", mtime=1_700_000_100)
    _, third = cb.pack(root, KEY, dirs, log=lambda _: None)
    assert {r.kind: (r.added, r.members) for r in third} == {"inductor": (1, 3), "triton": (0, 1), "miopen": (1, 2)}
    assert len(_names(bundle / "miopen.tar")) == 3  # appended, not rewritten
    out = {"inductor": tmp_path / "o" / "i", "triton": tmp_path / "o" / "t", "miopen": tmp_path / "o" / "m"}
    cb.unpack(bundle, KEY, out, log=lambda _: None)
    assert (out["miopen"] / "gfx942130.HIP.ufdb.txt").read_text() == "find-db v2"  # the last copy wins
    assert len(json.loads((bundle / "manifest.json").read_text())["history"]) == 3


def test_a_bundle_for_another_stack_is_refused_unless_forced(tmp_path):
    dirs = _caches(tmp_path / "built")
    other = cb.StackKey(KEY.product, KEY.arch, "2.11.0+rocm7.1", KEY.runtime, KEY.triton)
    bundle, _ = cb.pack(tmp_path / "bundles", other, dirs, log=lambda _: None)
    out = {"inductor": tmp_path / "o" / "i", "triton": tmp_path / "o" / "t"}
    with pytest.raises(cb.BundleMismatchError, match="has no bundle for"):
        cb.unpack(tmp_path / "bundles", KEY, out, log=lambda _: None)
    with pytest.raises(cb.BundleMismatchError, match="torch 2.11.0"):
        cb.unpack(bundle, KEY, out, log=lambda _: None)
    cb.unpack(bundle, KEY, out, force=True, log=lambda _: None)
    ok, line = cb.startup_line(KEY, out)
    assert not ok and "torch 2.11.0+rocm7.1 (here 2.12.1+rocm7.2)" in line


def test_a_golden_copy_layout_reads_as_a_bundle(tmp_path):
    """$WORK/.cache/compile-golden/<product>/<stack>/{inductor,triton,miopen}.tar,
    made with ``tar -cf X.tar -C dir .``: no manifest, a ``./`` directory member."""
    built = _caches(tmp_path / "built")
    golden = tmp_path / "golden" / KEY.path
    golden.mkdir(parents=True)
    for kind in ("inductor", "triton", "miopen"):
        exclude = ["--exclude=./triton"] if kind == "inductor" else []
        subprocess.run(["tar", "-cf", str(golden / f"{kind}.tar"), *exclude, "-C", str(built[kind]), "."],
                       check=True)
    assert cb.find_bundles(tmp_path / "golden") == [golden]
    assert cb.bundle_key(golden) == cb.StackKey(**{**KEY.__dict__, "arch": None})
    out = {"inductor": tmp_path / "o" / "i", "triton": tmp_path / "o" / "t", "miopen": tmp_path / "o" / "m"}
    cb.unpack(tmp_path / "golden", KEY, out, log=lambda _: None)
    assert _tree(out["miopen"]) == _tree(built["miopen"])
    # packing into it appends to the same tars and adds a manifest
    _write(built["triton"] / "NEW" / "k.hsaco", "binary 2")
    _, res = cb.pack(tmp_path / "golden", KEY, built, log=lambda _: None)
    assert {r.kind: r.added for r in res} == {"inductor": 0, "triton": 1, "miopen": 0}
    assert (golden / "manifest.json").is_file()


def test_startup_line_says_whether_the_unpacked_bundle_matches(tmp_path):
    dirs = _caches(tmp_path / "built")
    ok, line = cb.startup_line(KEY, dirs)
    assert ok and "no bundle unpacked" in line and KEY.path in line
    cb.pack(tmp_path / "b", KEY, dirs, log=lambda _: None)
    cb.unpack(tmp_path / "b", KEY, dirs, log=lambda _: None)
    ok, line = cb.startup_line(KEY, dirs)
    assert ok and line.endswith("matches")
    elsewhere = cb.StackKey("mi300x", KEY.arch, KEY.torch, KEY.runtime, KEY.triton)
    ok, line = cb.startup_line(elsewhere, dirs)
    assert ok and line.endswith("tuned on mi325x-vf")


def test_miopen_kernels_without_a_dir_of_their_own_go_to_the_db_dir(tmp_path):
    built = {"inductor": tmp_path / "i", "miopen": tmp_path / "db", "miopen-kernels": tmp_path / "kc"}
    _write(built["miopen"] / "x.ufdb.txt", "db")
    _write(built["miopen-kernels"] / "3.5.1" / "x.ukdb", "kernels")
    bundle, _ = cb.pack(tmp_path / "b", KEY, built, log=lambda _: None)
    assert _names(bundle / "miopen-kernels.tar") == ["./3.5.1/x.ukdb"]
    shared = {"inductor": tmp_path / "o" / "i", "miopen": tmp_path / "o" / "m"}
    cb.unpack(bundle, KEY, shared, log=lambda _: None)
    assert (shared["miopen"] / "3.5.1" / "x.ukdb").read_text() == "kernels"


def test_cli_wires_cache_subcommands(monkeypatch, tmp_path, capsys):
    calls = {}
    monkeypatch.setattr(cb, "stack_key", lambda device=None: KEY)
    monkeypatch.setattr(cb, "cache_dirs", lambda: {"inductor": tmp_path / "i", "triton": tmp_path / "t"})

    def pack(root, key, dirs, note=None):
        calls["pack"] = (root, dirs, note)
        return root, []

    monkeypatch.setattr(cb, "pack", pack)
    args = build_parser().parse_args(["cache", "pack", str(tmp_path / "r"), "--kinds", "triton", "--note", "n"])
    args.func(args)
    assert calls["pack"] == (tmp_path / "r", {"triton": tmp_path / "t"}, "n")

    def refuse(*a, **k):
        raise cb.BundleMismatchError("nothing for this stack")

    monkeypatch.setattr(cb, "unpack", refuse)
    args = build_parser().parse_args(["cache", "unpack", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == cache_cli.EXIT_NO_MATCH
    args = build_parser().parse_args(["serve", "qwen3_5_9b", "--compile-bundle", str(tmp_path)])
    assert args.compile_bundle == str(tmp_path)
