"""An ``Engine`` for tests, with no model loaded.

``Engine.__init__`` only sets fields, so ``make_engine`` runs the real one and a
new field reaches every test without an edit. ``load_model`` does the rest of the
setup: it needs submodules, specs and process groups, so a test supplies only the
collaborators the code under test reaches, usually a ``SimpleNamespace`` of what
it calls. One it forgot is an ``AttributeError`` rather than a stub answering for it.
"""
import torch

from mstar.engine.engine import Engine, SubmoduleManagement


def make_engine(**overrides) -> Engine:
    """An ``Engine`` on the CPU holding no resources or submodules. ``overrides``
    replace fields and methods by name; a name ``Engine`` does not have raises
    ``TypeError``, so a test cannot go on setting a field that has moved."""
    engine = Engine(graph_runtime=None)
    engine._device = torch.device("cpu")
    # what ``load_model`` sets for a model that declares nothing
    engine._loopback_steps = {}
    engine._keyed_walks = {}
    engine._item_encoders = set()
    engine._prefix_model = "StubModel"

    # updated in place, since ``_chunking`` holds this dict
    engine._submodules.update(overrides.pop("_submodules", {}))
    for name, value in overrides.items():
        if not hasattr(engine, name):
            raise TypeError(f"Engine has no field {name!r}")
        setattr(engine, name, value)
    return engine


def make_mgmt(submodule=None, **overrides) -> SubmoduleManagement:
    """A node's entry in ``Engine._submodules``, as ``load_model`` builds it, for a
    node with no resources or process group. ``overrides`` replace its fields."""
    defaults = {
        "forward": getattr(submodule, "forward", None),
        "forward_batched": getattr(submodule, "forward_batched", None),
        "joint_comm_group": None,
        "resources": {},
    }
    return SubmoduleManagement(submodule=submodule, **(defaults | overrides))
