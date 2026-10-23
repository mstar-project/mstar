"""GLM-5.3 calls only what CommGroup defines. The TP-only paths don't run in the CPU tests,
so a missing method would surface only when TP>1 serves."""
import pathlib
import re
import sys

sys.path.insert(0, ".")

from mstar.distributed.communication import CommGroup  # noqa: E402

ROOTS = ("mstar/model/glm5_next", "mstar/engine/resources/linear_attn", "mstar/utils/fused_moe")


def test_every_comm_group_call_exists():
    used = {
        name
        for root in ROOTS
        for path in pathlib.Path(root).rglob("*.py")
        for name in re.findall(r"comm_group\.([a-z_]+)\(", path.read_text())
    }
    assert used, "the pattern found no calls: the check is stale"
    assert {name for name in used if not hasattr(CommGroup, name)} == set()
