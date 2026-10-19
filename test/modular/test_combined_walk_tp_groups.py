"""A combined walk's step runs every walk's rows in one forward, so on a
worker its walks must share one TP group, or all have world size 1."""
from types import SimpleNamespace

import pytest

from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.worker.engine_manager import refuse_combined_walks_across_tp_groups

COMBINED = {("LLM", "prefill"): "mixed", ("LLM", "decode"): "mixed"}
MODEL = SimpleNamespace(combined_walk_of=lambda: COMBINED)
# the (node, walk) pairs of a worker that runs both halves
BOTH = frozenset({("LLM", "prefill"), ("LLM", "decode")})


def _check(*groups, walks=BOTH):
    sharding = ShardingConfig(groups=list(groups), tp_enabled_nodes={"LLM"}, shard_dim={})
    refuse_combined_walks_across_tp_groups(MODEL, sharding, set(walks))


def test_one_group_for_every_walk_passes():
    _check(ShardingGroup(nodes={"LLM"}, tp_size=2))


def test_no_tp_at_all_passes():
    _check()


def test_a_walk_outside_the_group_is_refused():
    with pytest.raises(ValueError, match="spans TP groups"):
        _check(ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks={"decode"}))


def test_two_groups_are_refused():
    with pytest.raises(ValueError, match="spans TP groups"):
        _check(
            ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks={"decode"}),
            ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks={"prefill"}),
        )


def test_a_group_of_one_counts_as_none():
    _check(ShardingGroup(nodes={"LLM"}, tp_size=1, graph_walks={"decode"}))


def test_another_workers_node_is_not_checked():
    _check(
        ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks={"decode"}),
        walks={("snac", "decode")},
    )


@pytest.mark.parametrize("walk", ["prefill", "decode"])
def test_a_pd_worker_running_one_half_is_not_refused(walk):
    """P/D at TP2 on both sides, which the docs allow: each worker runs one
    half of the combined walk, so no worker can build a step that mixes
    them and no config change is needed."""
    _check(
        ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks={"prefill"}),
        ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks={"decode"}),
        walks={("LLM", walk)},
    )
