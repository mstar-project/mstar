"""Tests for mstar.distributed.config.ShardingConfig.

Covers:
- ShardingGroup.register_workers
- ShardingConfig.setup (explicit groups + auto-generated singletons)
- compute_fanout for replicated and sharded signals across matched / mismatched
  TP sizes, with and without producer/consumer colocation
"""
import pytest

from mstar.distributed.base import ShardDestination, ShardingConfig, ShardingGroup
from mstar.graph.base import NodeAndGraphWalk

# ---------------------------------------------------------------------------
# ShardingGroup
# ---------------------------------------------------------------------------


class TestShardingGroup:
    def test_register_workers_basic(self):
        g = ShardingGroup(nodes={"LLM"}, tp_size=2)
        g.register_workers(["w0", "w1"], my_tp_rank=1)
        assert g._workers == ["w0", "w1"]
        assert g._workers_set == {"w0", "w1"}
        assert g._tp_rank == 1

    def test_register_workers_my_tp_rank_optional(self):
        g = ShardingGroup(nodes={"LLM"}, tp_size=2)
        g.register_workers(["w0", "w1"])
        assert g._tp_rank is None
        assert g._workers == ["w0", "w1"]

    def test_register_workers_length_mismatch(self):
        g = ShardingGroup(nodes={"LLM"}, tp_size=4)
        with pytest.raises(AssertionError):
            g.register_workers(["w0", "w1"], my_tp_rank=0)

    def test_post_init_coerces_to_sets(self):
        g = ShardingGroup(nodes=["A", "B"], tp_size=1, graph_walks=["decode"])
        assert g.nodes == {"A", "B"}
        assert g.graph_walks == {"decode"}

    def test_graph_walks_none_sentinel_preserved(self):
        g = ShardingGroup(nodes=["A"], tp_size=1, graph_walks=None)
        assert g.graph_walks is None


# ---------------------------------------------------------------------------
# ShardingConfig.setup
# ---------------------------------------------------------------------------


class TestSetup:
    def test_setup_maps_explicit_groups(self):
        llm = ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks={"decode"})
        llm.register_workers(["w0", "w1"], my_tp_rank=0)
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[llm], shard_dim={})
        cfg.setup({
            NodeAndGraphWalk("LLM", "decode"): ["w0", "w1"],
            NodeAndGraphWalk("encoder", "prefill"): ["w0"],
        })
        assert cfg.group_mapping[NodeAndGraphWalk("LLM", "decode")] is llm
        # encoder is auto-generated as a singleton
        enc = cfg.group_mapping[NodeAndGraphWalk("encoder", "prefill")]
        assert enc.tp_size == 1
        assert enc._workers == ["w0"]
        assert enc._workers_set == {"w0"}
        assert enc._tp_rank == 0

    def test_setup_graph_walks_none_expands_to_all(self):
        """A group with graph_walks=None should map every observed walk."""
        all_walks = ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks=None)
        all_walks.register_workers(["w0", "w1"], my_tp_rank=0)
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[all_walks], shard_dim={})
        cfg.setup({
            NodeAndGraphWalk("LLM", "prefill"): ["w0", "w1"],
            NodeAndGraphWalk("LLM", "decode"): ["w0", "w1"],
        })
        assert cfg.group_mapping[NodeAndGraphWalk("LLM", "prefill")] is all_walks
        assert cfg.group_mapping[NodeAndGraphWalk("LLM", "decode")] is all_walks

    def test_setup_extracts_graph_walks_from_key_tuples(self):
        """Regression: setup used to extract x[0] (node) instead of x[1] (walk).

        With graph_walks=None we expand to "all observed walks." If extraction
        is wrong, the explicit group's mapping will be keyed by node-name
        strings instead of walk names and won't match lookup keys.
        """
        g = ShardingGroup(nodes={"LLM"}, tp_size=1, graph_walks=None)
        g.register_workers(["w0"], my_tp_rank=0)
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[g], shard_dim={})
        cfg.setup({
            NodeAndGraphWalk("LLM", "decode"): ["w0"],
            NodeAndGraphWalk("encoder", "prefill"): ["w0"],
        })
        # The (LLM, decode) key must land on the explicit group.
        assert cfg.group_mapping[NodeAndGraphWalk("LLM", "decode")] is g

    def test_setup_singleton_workers_are_unwrapped(self):
        """Regression: setup used to do set([list_of_workers]) which TypeErrors."""
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
        cfg.setup({NodeAndGraphWalk("encoder", "prefill"): ["w3"]})
        enc = cfg.group_mapping[NodeAndGraphWalk("encoder", "prefill")]
        assert enc._workers_set == {"w3"}
        assert enc._workers == ["w3"]

    def test_compute_fanout_requires_setup(self):
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
        with pytest.raises(RuntimeError, match="setup"):
            cfg.compute_fanout(
                signal="x", source_graph_walk="decode",
                source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[4],
                source_tp_rank=0,
            )


# ---------------------------------------------------------------------------
# compute_fanout — replicated signal
# ---------------------------------------------------------------------------


def _make_config(
    groups: dict[tuple[tuple[str, ...], int, tuple[str, ...] | None], list[str]],
    node_to_worker: dict,
    shard_dim: dict,
) -> ShardingConfig:
    """Build a ShardingConfig from a compact spec.

    ``groups`` keys are ``(nodes_tuple, tp_size, walks_tuple_or_None)``;
    values are the worker list. Registers workers automatically.
    """
    sg_list = []
    for (nodes, tp_size, walks), workers in groups.items():
        gw = set(walks) if walks is not None else None
        g = ShardingGroup(nodes=set(nodes), tp_size=tp_size, graph_walks=gw)
        g.register_workers(workers, my_tp_rank=0)
        sg_list.append(g)
    cfg = ShardingConfig(tp_enabled_nodes=set(), groups=sg_list, shard_dim=shard_dim)
    cfg.setup(node_to_worker)
    return cfg


class TestReplicatedFanout:
    def test_replicated_disjoint_workers(self):
        """Source TP=2, dest TP=2, disjoint workers. Only rank 0 sends; sends
        the full tensor to every dest worker.
        """
        cfg = _make_config(
            groups={
                (("A",), 2, ("decode",)): ["w0", "w1"],
                (("B",), 2, ("decode",)): ["w2", "w3"],
            },
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
                NodeAndGraphWalk("B", "decode"): ["w2", "w3"],
            },
            shard_dim={"x": None},  # replicated
        )
        fanout_rank0 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[1], source_tp_rank=0,
        )
        workers = {d.worker for d in fanout_rank0}
        assert workers == {"w2", "w3"}
        assert all(d.full_tensor for d in fanout_rank0)

        # Non-rank-0 source emits nothing (its worker isn't in dest, no send).
        fanout_rank1 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[1], source_tp_rank=1,
        )
        assert fanout_rank1 == []

    def test_replicated_colocated_source_and_dest(self):
        """Source TP=2 [w0,w1], dest TP=2 [w1,w2]: w1 already has the tensor
        (no comm); w0 (source rank 0) sends to w2 only.
        """
        cfg = _make_config(
            groups={
                (("A",), 2, ("decode",)): ["w0", "w1"],
                (("B",), 2, ("decode",)): ["w1", "w2"],
            },
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
                NodeAndGraphWalk("B", "decode"): ["w1", "w2"],
            },
            shard_dim={"x": None},
        )
        fanout_rank0 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[1], source_tp_rank=0,
        )
        # rank 0's worker (w0) is not in dest, so no self-target. It sends to
        # dest_workers \ source_workers = {w2}.
        assert fanout_rank0 == [ShardDestination(worker="w2", full_tensor=True, tp_rank=1)]

        fanout_rank1 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[1], source_tp_rank=1,
        )
        # rank 1's worker (w1) IS in dest — emits the "already in memory" marker.
        assert fanout_rank1 == [ShardDestination(worker="w1", full_tensor=True,  tp_rank=0)]

    def test_replicated_tp1_source_to_tp1_dest_same_worker(self):
        """Both non-TP and on the same worker: just the self-marker."""
        cfg = _make_config(
            groups={},  # all auto-generated singletons
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0"],
                NodeAndGraphWalk("B", "decode"): ["w0"],
            },
            shard_dim={"x": None},
        )
        fanout = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[1], source_tp_rank=0,
        )
        assert fanout == [ShardDestination(worker="w0", full_tensor=True, tp_rank=0)]


# ---------------------------------------------------------------------------
# compute_fanout — sharded signal
# ---------------------------------------------------------------------------


class TestShardedFanout:
    def test_matched_layout_tp2_to_tp2(self):
        """Sharded TP=2 -> TP=2, same layout. Each source sends to its
        corresponding dest rank, full local shard.
        """
        cfg = _make_config(
            groups={
                (("A",), 2, ("decode",)): ["w0", "w1"],
                (("B",), 2, ("decode",)): ["w2", "w3"],
            },
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
                NodeAndGraphWalk("B", "decode"): ["w2", "w3"],
            },
            shard_dim={"x": 1},
        )
        # Source rank 0 holds [0,4), sends to dest rank 0 (w2).
        f0 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[4], source_tp_rank=0,
        )
        assert f0 == [ShardDestination(worker="w2", full_tensor=False, tp_rank=0, start_idxs=[0], end_idxs=[4])]
        # Source rank 1 holds [4,8), sends to dest rank 1 (w3).
        f1 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[4], source_tp_rank=1,
        )
        assert f1 == [ShardDestination(worker="w3", full_tensor=False, tp_rank=1, start_idxs=[4], end_idxs=[8])]

    def test_scatter_tp1_to_tp2(self):
        """Non-TP source -> TP=2 dest: source holds the full tensor, sends a
        slice to each dest rank.
        """
        cfg = _make_config(
            groups={
                (("B",), 2, ("decode",)): ["w1", "w2"],
            },
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0"],
                NodeAndGraphWalk("B", "decode"): ["w1", "w2"],
            },
            shard_dim={"x": 1},
        )
        fanout = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[8], source_tp_rank=0,
        )
        assert fanout == [
            ShardDestination(worker="w1", full_tensor=False, tp_rank=0, start_idxs=[0], end_idxs=[4]),
            ShardDestination(worker="w2", full_tensor=False, tp_rank=1, start_idxs=[4], end_idxs=[8]),
        ]

    def test_scatter_tp2_to_tp4(self):
        """TP=2 -> TP=4: each source rank sends two half-shards."""
        cfg = _make_config(
            groups={
                (("A",), 2, ("decode",)): ["w0", "w1"],
                (("B",), 4, ("decode",)): ["w2", "w3", "w4", "w5"],
            },
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
                NodeAndGraphWalk("B", "decode"): ["w2", "w3", "w4", "w5"],
            },
            shard_dim={"x": 1},
        )
        # shard_dim_size=4, total=8, dest_shard_size=2.
        # Source rank 0 holds [0,4): goes to dest ranks 0 and 1.
        f0 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[4], source_tp_rank=0,
        )
        assert f0 == [
            ShardDestination(worker="w2", full_tensor=False, tp_rank=0, start_idxs=[0], end_idxs=[2]),
            ShardDestination(worker="w3", full_tensor=False, tp_rank=1, start_idxs=[2], end_idxs=[4]),
        ]
        # Source rank 1 holds [4,8): goes to dest ranks 2 and 3.
        f1 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[4], source_tp_rank=1,
        )
        assert f1 == [
            ShardDestination(worker="w4", full_tensor=False, tp_rank=2, start_idxs=[4], end_idxs=[6]),
            ShardDestination(worker="w5", full_tensor=False, tp_rank=3, start_idxs=[6], end_idxs=[8]),
        ]

    def test_gather_tp4_to_tp2(self):
        """TP=4 -> TP=2: each source rank sends its full shard to one dest."""
        cfg = _make_config(
            groups={
                (("A",), 4, ("decode",)): ["w0", "w1", "w2", "w3"],
                (("B",), 2, ("decode",)): ["w4", "w5"],
            },
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0", "w1", "w2", "w3"],
                NodeAndGraphWalk("B", "decode"): ["w4", "w5"],
            },
            shard_dim={"x": 1},
        )
        # shard_dim_size=2, total=8, dest_shard_size=4.
        # Source ranks 0,1 -> dest 0; ranks 2,3 -> dest 1.
        expected = [
            (0, "w4", 0, 2, 0),
            (1, "w4", 2, 4, 0),
            (2, "w5", 4, 6, 1),
            (3, "w5", 6, 8,  1),
        ]
        for src_rank, dst_worker, lo, hi, dst_rank in expected:
            f = cfg.compute_fanout(
                signal="x", source_graph_walk="decode",
                source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[2],
                source_tp_rank=src_rank,
            )
            assert f == [
                ShardDestination(worker=dst_worker, tp_rank=dst_rank, full_tensor=False, start_idxs=[lo], end_idxs=[hi])
            ], f"src_rank={src_rank}: {f}"

    def test_sharded_to_non_tp_consumer(self):
        """TP=2 -> TP=1: each source rank sends its shard to the single dest."""
        cfg = _make_config(
            groups={
                (("A",), 2, ("decode",)): ["w0", "w1"],
            },
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
                NodeAndGraphWalk("B", "decode"): ["w2"],
            },
            shard_dim={"x": 1},
        )
        f0 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[4], source_tp_rank=0,
        )
        assert f0 == [ShardDestination(worker="w2", tp_rank=0, full_tensor=False, start_idxs=[0], end_idxs=[4])]
        f1 = cfg.compute_fanout(
            signal="x", source_graph_walk="decode",
            source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[4], source_tp_rank=1,
        )
        assert f1 == [ShardDestination(worker="w2", tp_rank=0, full_tensor=False, start_idxs=[4], end_idxs=[8])]

    def test_uneven_tp_partition_asserts(self):
        """total_size must be divisible by dest TP size."""
        cfg = _make_config(
            groups={
                (("A",), 2, ("decode",)): ["w0", "w1"],
                (("B",), 4, ("decode",)): ["w2", "w3", "w4", "w5"],
            },
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
                NodeAndGraphWalk("B", "decode"): ["w2", "w3", "w4", "w5"],
            },
            shard_dim={"x": 1},
        )
        # source.tp_size=2, shard_dim_size=3, total=6, dest.tp_size=4 -> 6 % 4 != 0.
        with pytest.raises(AssertionError, match="divisible"):
            cfg.compute_fanout(
                signal="x", source_graph_walk="decode",
                source_node="A", dest_node="B", dest_graph_walk="decode", shard_dim_sizes=[3], source_tp_rank=0,
            )


# ---------------------------------------------------------------------------
# Cross-graph-walk and helpers
# ---------------------------------------------------------------------------


class TestFanin:
    def test_replicated_fanin_is_one(self):
        cfg = _make_config(
            groups={(("B",), 2, ("decode",)): ["w0", "w1"]},
            node_to_worker={
                NodeAndGraphWalk("A", "decode"): ["w0"],
                NodeAndGraphWalk("B", "decode"): ["w0", "w1"],
            },
            shard_dim={"x": None},
        )
        assert cfg.compute_fanin(
            signal="x", source_tp_size=4,
            dest_node="B", dest_graph_walk="decode",
            dest_tp_rank=1
        ) == 1

    @pytest.mark.parametrize("dest_rank,expected", [(0, 1), (1, 1)])
    def test_fanin_matched_tp2_to_tp2(self, dest_rank, expected):
        a = ShardingGroup(nodes={"A"}, tp_size=2, graph_walks={"decode"})
        a.register_workers(["w0", "w1"], my_tp_rank=0)
        b = ShardingGroup(nodes={"B"}, tp_size=2, graph_walks={"decode"})
        b.register_workers(["w2", "w3"], my_tp_rank=dest_rank)
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[a, b], shard_dim={"x": 1})
        cfg.setup({
            NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
            NodeAndGraphWalk("B", "decode"): ["w2", "w3"],
        })
        assert cfg.compute_fanin(
            signal="x", source_tp_size=2,
            dest_node="B", dest_graph_walk="decode",
            dest_tp_rank=dest_rank
        ) == expected

    @pytest.mark.parametrize("dest_rank", [0, 1, 2, 3])
    def test_fanin_scatter_tp1_to_tp4(self, dest_rank):
        """Source covers entire range, every dest reads from it: fanin=1."""
        b = ShardingGroup(nodes={"B"}, tp_size=4, graph_walks={"decode"})
        b.register_workers(["w1", "w2", "w3", "w4"], my_tp_rank=dest_rank)
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[b], shard_dim={"x": 1})
        cfg.setup({
            NodeAndGraphWalk("A", "decode"): ["w0"],
            NodeAndGraphWalk("B", "decode"): ["w1", "w2", "w3", "w4"],
        })
        assert cfg.compute_fanin(
            signal="x", source_tp_size=1,
            dest_node="B", dest_graph_walk="decode",
            dest_tp_rank=dest_rank
        ) == 1

    @pytest.mark.parametrize("dest_rank", [0, 1])
    def test_fanin_gather_tp4_to_tp2(self, dest_rank):
        """Each dest covers 2 source shards: fanin=2 for every dest rank."""
        a = ShardingGroup(nodes={"A"}, tp_size=4, graph_walks={"decode"})
        a.register_workers(["w0", "w1", "w2", "w3"], my_tp_rank=0)
        b = ShardingGroup(nodes={"B"}, tp_size=2, graph_walks={"decode"})
        b.register_workers(["w4", "w5"], my_tp_rank=dest_rank)
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[a, b], shard_dim={"x": 1})
        cfg.setup({
            NodeAndGraphWalk("A", "decode"): ["w0", "w1", "w2", "w3"],
            NodeAndGraphWalk("B", "decode"): ["w4", "w5"],
        })
        assert cfg.compute_fanin(
            signal="x", source_tp_size=4,
            dest_node="B", dest_graph_walk="decode",
            dest_tp_rank=dest_rank
        ) == 2

    @pytest.mark.parametrize("dest_rank,expected", [(0, 1), (1, 2), (2, 1)])
    def test_fanin_unaligned_tp2_to_tp3(self, dest_rank, expected):
        """Non-divisible TP sizes: fanin varies per dest rank.

        Source TP=2 shards [0,3) and [3,6); dest TP=3 shards [0,2), [2,4), [4,6).
        Dest 0 reads source 0 only; dest 1 reads both; dest 2 reads source 1 only.
        """
        a = ShardingGroup(nodes={"A"}, tp_size=2, graph_walks={"decode"})
        a.register_workers(["w0", "w1"], my_tp_rank=0)
        b = ShardingGroup(nodes={"B"}, tp_size=3, graph_walks={"decode"})
        b.register_workers(["w2", "w3", "w4"], my_tp_rank=dest_rank)
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[a, b], shard_dim={"x": 1})
        cfg.setup({
            NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
            NodeAndGraphWalk("B", "decode"): ["w2", "w3", "w4"],
        })
        assert cfg.compute_fanin(
            signal="x", source_tp_size=2,
            dest_node="B", dest_graph_walk="decode",
            dest_tp_rank=dest_rank
        ) == expected


class TestCrossGraphWalkAndAllFanouts:
    def test_dest_graph_walk_resolves_to_different_group(self):
        """Prefill TP=4 -> Decode TP=2 reshard across graph walks."""
        prefill_grp = ShardingGroup(nodes={"LLM"}, tp_size=4, graph_walks={"prefill"})
        prefill_grp.register_workers(["w0", "w1", "w2", "w3"], my_tp_rank=0)
        decode_grp = ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks={"decode"})
        decode_grp.register_workers(["w4", "w5"], my_tp_rank=0)
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[prefill_grp, decode_grp], shard_dim={"kv": 0})
        cfg.setup({
            NodeAndGraphWalk("LLM", "prefill"): ["w0", "w1", "w2", "w3"],
            NodeAndGraphWalk("LLM", "decode"): ["w4", "w5"],
        })
        # Source rank 1 of prefill TP=4: shard_dim_size=2, total=8, dest_shard_size=4.
        # Source range [2,4) overlaps dest 0 [0,4). One fanout to w4.
        fanout = cfg.compute_fanout(
            signal="kv", source_graph_walk="prefill", dest_graph_walk="decode",
            source_node="LLM", dest_node="LLM", shard_dim_sizes=[2], source_tp_rank=1,
        )
        assert fanout == [
            ShardDestination(worker="w4", tp_rank=0, full_tensor=False, start_idxs=[2], end_idxs=[4])
        ]

