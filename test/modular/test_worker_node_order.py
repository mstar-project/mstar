"""A worker builds and captures its nodes in the same order on every boot."""

import json
import os
import subprocess
import sys
import textwrap

# more names than a set keeps in insertion order by chance
NODES = [f"node_{i}" for i in range(12)]

_PRINT_ORDER = textwrap.dedent(f"""
    import json, sys
    sys.path.insert(0, ".")
    from types import SimpleNamespace
    from mstar.worker.worker import _nodes_in_graph_order

    def graph(names):
        return SimpleNamespace(section=SimpleNamespace(
            get_nodes=lambda: {{name: None for name in names}}
        ))

    nodes = {NODES!r}
    # a node two graphs share is listed once, where it first appears
    print(json.dumps(_nodes_in_graph_order([graph(nodes[:8]), graph(nodes[4:])])))
""")


def _order(hash_seed: str) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-c", _PRINT_ORDER],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_node_order_does_not_depend_on_string_hashing():
    first, second = _order("1"), _order("2")

    assert first == second, (
        "two boots of the same config built their nodes in different orders"
    )
    assert first == NODES, "the nodes are not in the order the graphs name them"
