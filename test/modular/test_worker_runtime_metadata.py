"""Worker.__init__ tells the graph runtime which nodes are TP nodes. Without it the
runtime sees none, and a TP node passes the speculation filter meant to hold it to
the leader. No CPU test builds a whole Worker, so check the call statically."""
import ast
import pathlib


def _init_calls(cls: str, method: str) -> set[str]:
    tree = ast.parse(pathlib.Path("mstar/worker/worker.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    return {ast.unparse(c.func) for c in ast.walk(init) if isinstance(c, ast.Call)}


def test_init_hands_the_runtime_its_node_metadata():
    assert "self._graph_runtime.set_node_metadata" in _init_calls("Worker", "__init__")
