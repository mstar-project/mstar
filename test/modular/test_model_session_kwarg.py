"""Every model takes the request's session, used or not.

The conductor passes ``session=`` to ``get_initial_forward_pass_args`` on every
request, ``None`` when there is no session. A model whose override does not
accept it fails every request at ingest, sessions or not; two models merged in
from main did exactly that. Parsed, not imported, so a model whose optional
dependencies are missing is still checked.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODEL_ROOT = Path("mstar/model")


def _overrides():
    for path in sorted(MODEL_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "get_initial_forward_pass_args"
            ):
                yield pytest.param(node, id=f"{path}:{node.lineno}")


@pytest.mark.parametrize("fn", list(_overrides()))
def test_get_initial_forward_pass_args_accepts_the_session(fn):
    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    assert "session" in names or fn.args.kwarg is not None, (
        "add `session=None` or `**kwargs`: the conductor passes `session=` on "
        "every request"
    )


def test_there_are_models_to_check():
    assert len(list(_overrides())) > 10
