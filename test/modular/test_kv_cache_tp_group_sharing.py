"""A KV cache shared by several nodes must stay inside one TP group.

The TP scheduling paths (serial and ``MSTAR_TP_ASYNC_SCHED``) derive every
per-step verdict on each rank from allocator state assumed identical across
the group's ranks. A node outside the group sharing the cache breaks that
silently; ``KVCacheEngine`` refuses the configuration at warmup instead.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.engine.kv_cache_engine import KVCacheEngine  # noqa: E402


def _engine(node_to_cache_and_group):
    return types.SimpleNamespace(
        submodule_management={
            name: types.SimpleNamespace(kv_management=cache, tp_group=group)
            for name, (cache, group) in node_to_cache_and_group.items()
        }
    )


def _check(engine):
    KVCacheEngine._verify_shared_kv_caches_stay_in_one_tp_group(engine)


def _group(*members, rank=0):
    return CommGroup(my_global_rank=members[rank], my_group_rank=rank, group_members=list(members))


def test_one_tp_group_sharing_a_cache_is_fine():
    cache = object()
    _check(_engine({
        "prefill": (cache, _group(0, 1)),
        "decode": (cache, _group(0, 1)),
    }))


def test_separate_caches_per_group_are_fine():
    _check(_engine({
        "thinker": (object(), _group(0, 1)),
        "talker": (object(), _group(2, 3)),
    }))


def test_non_tp_nodes_sharing_a_cache_are_fine():
    cache = object()
    _check(_engine({
        "a": (cache, CommGroup.trivial()),
        "b": (cache, CommGroup.trivial()),
    }))


def test_cache_shared_with_a_node_outside_the_group_is_refused():
    cache = object()
    with pytest.raises(RuntimeError, match="shared across TP groups"):
        _check(_engine({
            "thinker": (cache, _group(0, 1)),
            "talker": (cache, _group(2, 3)),
        }))


def test_cache_shared_between_a_tp_node_and_a_non_tp_node_is_refused():
    cache = object()
    with pytest.raises(RuntimeError, match="shared across TP groups"):
        _check(_engine({
            "thinker": (cache, _group(0, 1)),
            "side": (cache, CommGroup.trivial()),
        }))
