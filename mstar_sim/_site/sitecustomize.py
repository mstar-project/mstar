"""Installs the mstar_sim emulated GPU in every Python process the launcher starts.

``python -m mstar_sim`` puts this directory on ``PYTHONPATH`` with ``MSTAR_SIM``
set, so mstar's worker, conductor and API-server processes (spawned, not
forked) come up on the emulated GPU too. A ``sitecustomize`` that was already
on the path still runs.
"""

import importlib.util
import os
import sys

if os.environ.get("MSTAR_SIM"):
    import mstar_sim.install

    mstar_sim.install.install()

_here = os.path.dirname(os.path.abspath(__file__))
for _path in sys.path:
    if _path and os.path.abspath(_path) != _here:
        _spec = importlib.util.find_spec  # noqa: F841
        _candidate = os.path.join(_path, "sitecustomize.py")
        if os.path.isfile(_candidate):
            _s = importlib.util.spec_from_file_location("_mstar_sim_next_sitecustomize", _candidate)
            _s.loader.exec_module(importlib.util.module_from_spec(_s))
            break
