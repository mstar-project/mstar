"""The JIT build-lock cleanup removes only locks that are both unheld locally and old: the
extension cache is on a shared filesystem, so a young unheld lock may belong to a rank on
another node (removing it made that rank's load fail and fall back to a slower kernel)."""
import os
import time

import torch.utils.cpp_extension as cpp_ext

from mstar.utils.fused_moe.align import _clear_stale_build_lock


def _with_build_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp_ext, "_get_build_directory", lambda name, verbose=False: str(tmp_path))
    lock = tmp_path / "lock"
    lock.write_text("")
    return lock


def test_young_unheld_lock_is_left_alone(monkeypatch, tmp_path):
    lock = _with_build_dir(monkeypatch, tmp_path)
    _clear_stale_build_lock("_x", min_age_s=1200)
    assert lock.exists()


def test_old_unheld_lock_is_removed(monkeypatch, tmp_path):
    lock = _with_build_dir(monkeypatch, tmp_path)
    old = time.time() - 3600
    os.utime(lock, (old, old))
    (tmp_path / ".ninja_lock").write_text("")
    _clear_stale_build_lock("_x", min_age_s=1200)
    assert not lock.exists() and not (tmp_path / ".ninja_lock").exists()


def test_held_lock_is_kept_even_when_old(monkeypatch, tmp_path):
    lock = _with_build_dir(monkeypatch, tmp_path)
    old = time.time() - 3600
    os.utime(lock, (old, old))
    fd = os.open(str(lock), os.O_RDONLY)  # this process holds it, like torch's FileBaton
    try:
        _clear_stale_build_lock("_x", min_age_s=1200)
        assert lock.exists()
    finally:
        os.close(fd)


def test_missing_lock_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp_ext, "_get_build_directory", lambda name, verbose=False: str(tmp_path))
    _clear_stale_build_lock("_x")
