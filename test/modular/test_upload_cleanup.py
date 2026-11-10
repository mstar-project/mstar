"""A request's uploads are deleted after the cleanup delay by one shared
thread, not a thread per request asleep for the delay."""
import threading
import time

from mstar.api_server import media_io


def _gone(paths, timeout_s=5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not any(p.exists() for p in paths):
            return True
        time.sleep(0.02)
    return False


def test_many_cleanups_share_one_thread_and_each_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(media_io, "UPLOAD_CLEANUP_DELAY_S", 0.5)
    files = [tmp_path / f"{i}.png" for i in range(50)]
    before = threading.active_count()
    for f in files:
        f.write_bytes(b"x")
        media_io.schedule_upload_cleanup({"image": [str(f)]}, tmp_path)
    assert threading.active_count() - before <= 1, "a thread per scheduled cleanup"
    assert all(f.exists() for f in files), "deleted before the delay"
    assert _gone(files)


def test_a_short_delay_after_a_long_one_is_not_held_behind_it(tmp_path, monkeypatch):
    late, soon = tmp_path / "late.png", tmp_path / "soon.png"
    late.write_bytes(b"x")
    soon.write_bytes(b"x")
    monkeypatch.setattr(media_io, "UPLOAD_CLEANUP_DELAY_S", 30.0)
    media_io.schedule_upload_cleanup({"image": [str(late)]}, tmp_path)
    monkeypatch.setattr(media_io, "UPLOAD_CLEANUP_DELAY_S", 0.0)
    media_io.schedule_upload_cleanup({"image": [str(soon)]}, tmp_path)
    assert _gone([soon]) and late.exists()
