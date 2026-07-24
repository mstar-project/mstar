"""Boot-time torch.compile mega-cache + boot-phase timing.

Two independent, default-OFF, byte-identical-when-off features. Both are boot
hygiene only (they change *how fast* a worker boots, never *what* it computes)
and both degrade gracefully: any failure here logs a warning and falls back to a
normal (uncached / unlogged) boot. A cache problem must NEVER fail a boot.

1. MEGA-CACHE (env ``MSTAR_MEGA_CACHE=<path-prefix>``, default unset = off)
   Persists torch's compile cache artifacts (inductor / dynamo / autotune)
   across boots via ``torch.compiler.{save,load}_cache_artifacts`` (torch>=2.6,
   verified on 2.9.1). ``load_mega_cache(role)`` runs once per worker BEFORE the
   first ``torch.compile`` fires; ``save_mega_cache(role)`` runs once AFTER
   warmup+capture completes. This is the same mechanism vLLM uses to avoid the
   15-25 min compile-dominated boot.

   The on-disk artifact path is::

       <path-prefix>.<git-sha>.<role>.bin

   The git sha of the repo HEAD is folded in automatically so a code change
   auto-misses the cache (a stale artifact from different code would otherwise
   silently mis-compile or, worse, load kernels for the wrong graph). Override
   the sha with ``MSTAR_MEGA_CACHE_SHA`` (e.g. to force reuse across a no-op
   commit). Set ``MSTAR_MEGA_CACHE_REFRESH=1`` to overwrite an existing artifact
   (otherwise an existing file is left untouched, so steady-state boots do no
   write).

2. BOOT PHASES (env ``MSTAR_BOOT_PHASES=1``, default off)
   ``boot_phase(name)`` emits one WARNING line per boot phase (WARNING so it
   survives the default INFO/quieted log config and lands in server.log), with
   elapsed-since-process-start and delta-since-previous-phase. First-occurrence
   wins per name so the many per-submodule / per-slot compile+capture calls
   collapse to a single clean line each. Zero cost when off (one env read).
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ─── Boot-phase timing ──────────────────────────────────────────────────────

# Per-process (module-global) state. Each worker is its own process, so these
# are naturally scoped to one boot.
_phase_t0: float | None = None
_phase_last: float | None = None
_phase_seen: set[str] = set()


def boot_phases_enabled() -> bool:
    return os.environ.get("MSTAR_BOOT_PHASES", "0") == "1"


def boot_phase(name: str) -> None:
    """Record and log boot phase ``name`` (first occurrence per process wins).

    No-op unless ``MSTAR_BOOT_PHASES=1``. Never raises.
    """
    if not boot_phases_enabled():
        return
    try:
        global _phase_t0, _phase_last
        now = time.monotonic()
        if _phase_t0 is None:
            _phase_t0 = now
            _phase_last = now
        if name in _phase_seen:
            return
        _phase_seen.add(name)
        elapsed = now - _phase_t0
        delta = now - (_phase_last if _phase_last is not None else now)
        _phase_last = now
        # WARNING level: the server runs at INFO but noisy loggers are quieted;
        # WARNING guarantees the line reaches server.log regardless of level.
        logger.warning(
            "BOOT_PHASE %-16s t=%8.2fs (+%7.2fs)", name, elapsed, delta
        )
    except Exception:  # phase logging must never perturb a boot
        pass


# ─── Mega-cache ─────────────────────────────────────────────────────────────

_MEGA_CACHE_ENV = "MSTAR_MEGA_CACHE"
_REFRESH_ENV = "MSTAR_MEGA_CACHE_REFRESH"
_SHA_ENV = "MSTAR_MEGA_CACHE_SHA"

_git_sha_cache: str | None = None
# Guard so a repeated call (e.g. two engines in one worker) can't double-load.
_loaded = False


def mega_cache_enabled() -> bool:
    return bool(os.environ.get(_MEGA_CACHE_ENV, "").strip())


def _git_sha() -> str:
    """Short repo HEAD sha, or the ``MSTAR_MEGA_CACHE_SHA`` override, or
    ``nosha``. Cached per process. Never raises."""
    global _git_sha_cache
    if _git_sha_cache is not None:
        return _git_sha_cache
    override = os.environ.get(_SHA_ENV, "").strip()
    if override:
        _git_sha_cache = override
        return _git_sha_cache
    sha = "nosha"
    try:
        repo = Path(__file__).resolve().parents[2]  # .../<repo>/mstar/utils/..
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            sha = out.stdout.strip()
    except Exception:
        pass
    _git_sha_cache = sha
    return _git_sha_cache


def _artifact_path(role: str) -> Path:
    prefix = os.environ[_MEGA_CACHE_ENV].strip()
    role = role or "unknown"
    return Path(f"{prefix}.{_git_sha()}.{role}.bin")


def load_mega_cache(role: str) -> bool:
    """Hot-load persisted compile artifacts for ``role`` before first compile.

    Call once per worker process before any ``torch.compile``. No-op unless
    ``MSTAR_MEGA_CACHE`` is set. Returns True if artifacts were loaded. A
    missing / stale / corrupt artifact degrades to a normal compile (returns
    False, logs at most a warning). Never raises.
    """
    global _loaded
    if not mega_cache_enabled() or _loaded:
        return False
    _loaded = True  # attempt-once; a failed load must not retry mid-boot
    try:
        path = _artifact_path(role)
        if not path.exists():
            logger.info(
                "mega_cache[%s]: no artifact at %s (cold compile)", role, path
            )
            return False
        data = path.read_bytes()
        info = torch.compiler.load_cache_artifacts(data)
        if info is None:
            # torch swallows a corrupt/unreadable artifact internally (logs its
            # own traceback) and returns None instead of raising. Nothing was
            # loaded -> treat as a miss and cold-compile.
            logger.warning(
                "mega_cache[%s]: artifact at %s unusable (stale/corrupt), "
                "falling back to normal compile", role, path,
            )
            return False
        logger.warning(
            "mega_cache[%s]: loaded %d bytes from %s (%s)",
            role, len(data), path, info,
        )
        return True
    except Exception:
        logger.warning(
            "mega_cache[%s]: load failed, falling back to normal compile",
            role, exc_info=True,
        )
        return False


def save_mega_cache(role: str) -> bool:
    """Persist compile artifacts for ``role`` after warmup+capture completes.

    Call once per worker after all compile/capture is done. No-op unless
    ``MSTAR_MEGA_CACHE`` is set. Skips the write if the artifact already exists
    unless ``MSTAR_MEGA_CACHE_REFRESH=1`` (so steady-state boots do no I/O).
    Writes tmp + atomic rename. Returns True on write. Never raises.
    """
    if not mega_cache_enabled():
        return False
    tmp = None
    try:
        path = _artifact_path(role)
        refresh = os.environ.get(_REFRESH_ENV, "0") == "1"
        if path.exists() and not refresh:
            logger.info(
                "mega_cache[%s]: artifact exists at %s, not overwriting "
                "(set %s=1 to refresh)", role, path, _REFRESH_ENV,
            )
            return False
        result = torch.compiler.save_cache_artifacts()
        if not result or not result[0]:
            logger.warning(
                "mega_cache[%s]: nothing to save (no compile artifacts)", role
            )
            return False
        data = result[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic within a filesystem
        logger.warning(
            "mega_cache[%s]: saved %d bytes to %s", role, len(data), path
        )
        return True
    except Exception:
        logger.warning(
            "mega_cache[%s]: save failed (boot unaffected)", role, exc_info=True
        )
        try:
            if tmp is not None and tmp.exists():  # best-effort tmp cleanup
                tmp.unlink()
        except Exception:
            pass
        return False
