"""Stub out broken lerobot subpackages before any test imports them.

Some lerobot versions have a malformed ``@dataclass`` in
``lerobot.policies.groot.groot_n1`` that crashes at import time. We don't
need groot for Pi0.5 tests, so we pre-register fake stub modules in
``sys.modules`` to satisfy the eager import chain in
``lerobot.policies.__init__``.
"""

import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []
    return m


if "lerobot.policies.groot.groot_n1" not in sys.modules:
    pkg = _make_stub("lerobot.policies.groot")
    g_n1 = _make_stub("lerobot.policies.groot.groot_n1")
    g_n1.GR00TN15 = type("GR00TN15", (), {})
    cfg = _make_stub("lerobot.policies.groot.configuration_groot")
    cfg.GrootConfig = type("GrootConfig", (), {})
    modg = _make_stub("lerobot.policies.groot.modeling_groot")
    modg.GrootPolicy = type("GrootPolicy", (), {})

    sys.modules["lerobot.policies.groot"] = pkg
    sys.modules["lerobot.policies.groot.groot_n1"] = g_n1
    sys.modules["lerobot.policies.groot.configuration_groot"] = cfg
    sys.modules["lerobot.policies.groot.modeling_groot"] = modg
