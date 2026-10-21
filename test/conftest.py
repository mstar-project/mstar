"""Tests run under the env defaults serving sets for its platform.

The conductor applies ``process_env_defaults()`` before it spawns workers. On
ROCm they turn off MIOpen's split-K conv solvers, among other things, so a GPU
test without them can run kernels serving never picks and report numbers that
don't match serving.
"""

import os


def apply_rocm_env_defaults() -> None:
    """Apply the platform's env defaults when this process drives ROCm GPUs,
    keeping any value already set. Other builds are never asked for devices."""
    try:
        from mstar import platforms
    except ImportError:  # test/ci also runs without torch
        return
    built_for = platforms._built_for()
    if not built_for.is_rocm():
        return
    # set before resolving, since counting devices starts the HIP runtime
    added = [key for key in built_for.process_env_defaults() if key not in os.environ]
    platforms.apply_process_env_defaults()
    if not platforms.resolve_platform().is_rocm():
        # GPUs hidden, so this is a CPU run and keeps the CPU env
        for key in added:
            del os.environ[key]


def pytest_configure(config):
    # before collection, which imports test modules that may touch the device
    apply_rocm_env_defaults()
