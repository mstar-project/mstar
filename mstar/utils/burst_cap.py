"""MSTAR_BURST_CAP — coordinated host-CPU budget across M* processes.

WHY THIS EXISTS:
    Under a prefill/admission wave, M* bursts many CPU cores at once — a
    burst *orchestrated across processes* (api_server preprocess + conductor
    + per-rank workers). vLLM stays flat because its host-CPU demand per
    step is small and constant. Under *same-priority* neighbor CPU load our
    burst stalls stochastically, giving multi-second, bimodal TTFT. The
    variance — not the mean — is the gap.

THE ROOT FAN-OUT:
    Nothing in the codebase ever calls ``torch.set_num_threads``. So every
    spawned process (mp "spawn": api_server main, conductor, one Worker per
    rank, plus the optional detok/sidecar children) inherits torch's DEFAULT
    intra-op (OpenMP/MKL) pool, which is sized to the machine's physical core
    count. On the bench box that is 128 cores PER PROCESS. When a 32-request
    wave hits, several processes each try to fan a CPU op (image resize /
    HF feature-extract in preprocess, host-side plan/reshape work in the
    workers) across dozens of cores at the same instant. On a quiet box the
    OS soaks it; under neighbor load at the same priority the threads are
    scheduled stochastically and the wave stalls. Capping each process to a
    small, fixed thread count makes our per-step host demand small and
    constant — the vLLM property — at the cost of a slightly slower burst in
    isolation.

DESIGN TENSION (must be measured, not assumed):
    A smaller cap is SLOWER on a quiet box (fewer threads for the resize /
    feature-extract) but ROBUST under contention (no oversubscription to
    thrash). The win is p95/p99 TTFT under load, not p50 on an idle box. The
    A/B must report BOTH. Tune MSTAR_BURST_THREADS to trade quiet-box cost
    against loaded-box robustness.

CONTRACT:
    - Default OFF (MSTAR_BURST_CAP unset/0): this module touches nothing —
      no set_num_threads, no env writes — so today's all-core behavior is
      preserved BYTE-FOR-BYTE. ``apply_process_thread_cap`` returns None.
    - Boot-time, per process: torch's thread-pool size is a process property
      set once at startup, so the cap is applied at each process's entry
      point and cannot be changed at runtime. A/B is via two boots. (The
      value is still read from the environment, so per-role overrides work.)
    - Sizing: MSTAR_BURST_THREADS (default 8) is the per-process budget; a
      role may override with MSTAR_BURST_THREADS_<ROLE> (e.g.
      MSTAR_BURST_THREADS_WORKER=16). The intended discipline is that the
      per-process caps SUM to the host-CPU budget you want the whole engine
      to occupy during a wave (e.g. api_server 8 + conductor 4 + worker 8 on
      a single-GPU i2t config ≈ 20, replacing an unbounded 3×128 fan-out).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Env flag / knob names (single source of truth).
ENV_ENABLE = "MSTAR_BURST_CAP"
ENV_THREADS = "MSTAR_BURST_THREADS"
ENV_THREADS_ROLE = "MSTAR_BURST_THREADS_{role}"  # per-role override, ROLE upper

_DEFAULT_THREADS = 8

# Env vars that native thread pools (OpenMP, MKL, OpenBLAS, numexpr) read at
# their own init. We set these too so a library that spins its pool up
# independently of torch — or a grandchild process — inherits the same cap.
_NATIVE_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Guard so a double call (e.g. a child that both inherits and re-applies) is a
# cheap no-op and logs once per process.
_applied_role: str | None = None


def enabled() -> bool:
    """True when MSTAR_BURST_CAP is on. Cheap; safe to call at any entry."""
    return os.environ.get(ENV_ENABLE, "0") == "1"


def threads_for_role(role: str) -> int:
    """Per-process thread budget for ``role``.

    MSTAR_BURST_THREADS_<ROLE> wins over MSTAR_BURST_THREADS; the fallback
    default is 8. A non-positive or unparseable value falls back to the
    default rather than pinning to 0 (0 would mean "let torch pick" = no cap).
    """
    role_key = ENV_THREADS_ROLE.format(role=role.upper())
    raw = os.environ.get(role_key) or os.environ.get(ENV_THREADS)
    if raw is None:
        return _DEFAULT_THREADS
    try:
        n = int(raw)
    except ValueError:
        logger.warning("MSTAR_BURST_THREADS bad value %r; using %d", raw, _DEFAULT_THREADS)
        return _DEFAULT_THREADS
    return n if n > 0 else _DEFAULT_THREADS


def apply_process_thread_cap(role: str) -> int | None:
    """Cap this process's host-CPU thread fan-out. Call ONCE at process entry.

    No-op (returns None) when MSTAR_BURST_CAP is off, so the default path is
    byte-identical. When on, sets torch's intra-op pool (``set_num_threads``),
    attempts the inter-op pool (guarded — it can only be set before any
    inter-op work), and exports the native-pool env vars so OpenMP/MKL and any
    subprocess inherit the same cap. Returns the applied thread count.

    ``role`` selects the per-role override and tags the log line
    (api_server / conductor / worker / detok / sidecar).
    """
    global _applied_role
    if not enabled():
        return None
    n = threads_for_role(role)

    # Native env FIRST: some libraries read these at import; set before the
    # torch import below can trigger MKL/OMP pool creation.
    for var in _NATIVE_THREAD_ENV:
        os.environ[var] = str(n)

    try:
        import torch
    except Exception:  # pragma: no cover - torch always present in serving procs
        # Env vars are still set for native libs; nothing more we can do.
        logger.info("MSTAR_BURST_CAP[%s]=%d (env only; torch unavailable)", role, n)
        _applied_role = role
        return n

    # Intra-op pool: the big one (resize, feature-extract, host tensor ops).
    # Changeable at any time, so this always takes effect.
    torch.set_num_threads(n)

    # Inter-op pool: can only be set before any inter-op parallel work has run.
    # At a fresh process entry that holds; guard so a late/duplicate call (or a
    # build that already initialized it) degrades to a warning instead of
    # crashing the process.
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError as e:
        logger.debug("MSTAR_BURST_CAP[%s]: interop pool already fixed (%s)", role, e)

    if _applied_role is None:
        logger.info(
            "MSTAR_BURST_CAP[%s]: intra-op threads capped to %d "
            "(was default all-core); native pools + subprocesses capped via %s",
            role, n, ",".join(_NATIVE_THREAD_ENV),
        )
    _applied_role = role
    return n


def capped_workers(default: int, role: str = "worker") -> int:
    """Thread count for an EXPLICIT pool (e.g. a transport ThreadPoolExecutor),
    clamped to the burst budget when the cap is on.

    Never raises the count above ``default`` — an existing pool sized for a
    reason (transport concurrency) is only ever narrowed, never widened. When
    the cap is off, returns ``default`` unchanged (byte-identical).
    """
    if not enabled():
        return default
    return min(default, threads_for_role(role))
