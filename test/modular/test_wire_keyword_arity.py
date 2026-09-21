"""Every dataclass constructed in-tree is called with keywords it actually has.

The rid refactor renamed a lot of `request_id` identifiers to `rid` with a
regex. It also caught keyword arguments at call sites, which silently broke
eight wire-message constructors -- silently because those are integration
paths the unit suite never builds. This test is the cheap structural guard:
it re-derives the answer from the AST rather than trusting the next sweep.
"""
import ast
import dataclasses
import pathlib
import sys

sys.path.insert(0, ".")

import mstar.communication.wire_types  # noqa: F401  (imports the wire dataclasses)

ROOT = pathlib.Path("mstar")


def _known_dataclasses() -> dict[str, set[str]]:
    """Every dataclass reachable from the loaded mstar modules, by class name."""
    fields: dict[str, set[str]] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("mstar") or module is None:
            continue
        for attr in vars(module).values():
            if not (dataclasses.is_dataclass(attr) and isinstance(attr, type)):
                continue
            # Only our own types: third-party configs (HF PretrainedConfig and
            # friends) take **kwargs, so their fields do not describe what
            # __init__ accepts and every extra kwarg would read as an error.
            if not getattr(attr, "__module__", "").startswith("mstar"):
                continue
            fields.setdefault(
                attr.__name__, {f.name for f in dataclasses.fields(attr)}
            )
    return fields


def test_no_call_passes_a_keyword_its_dataclass_does_not_have():
    known = _known_dataclasses()
    bad = []
    for path in ROOT.rglob("*.py"):
        if "/tests/" in path.as_posix():
            # In-tree model test dirs are not part of this suite and some are
            # already stale (mstar/model/cosmos3/tests/* still pass
            # sampling_config= to CurrentForwardPassInfo, dropped in #228).
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(
                func, "id", None
            )
            allowed = known.get(name)
            if allowed is None:
                continue  # not a dataclass we can resolve by name
            if any(kw.arg is None for kw in node.keywords):
                continue  # **kwargs splat; nothing to check statically
            for kw in node.keywords:
                if kw.arg not in allowed:
                    bad.append(
                        f"{path}:{node.lineno}: {name}({kw.arg}=...) "
                        f"but fields are {sorted(allowed)}"
                    )
    assert not bad, "constructor keyword does not match the dataclass:\n" + "\n".join(bad)
