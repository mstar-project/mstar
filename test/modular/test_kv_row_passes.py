"""A step's row passes in the page arena (``admit_rows``, ``commit_rows``,
``set_rows_in_flight``) leave every stream as the manager's per-row admit and
commit do, on either arena: two managers take the same steps, one with every
row sent down the per-row path."""
from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

import pytest  # noqa: E402
import torch  # noqa: E402

from mstar.engine.resources.kv import arena as arena_mod  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext  # noqa: E402

PAGE = 4
LABELS = ("main", "aux")


class _StubTransfer:
    def __init__(self, *a, **k):
        pass

    def get_kv_transfer_info(self, **k):
        pass

    def owns_transfer_info(self, *a, **k):
        return False

    def cleanup(self):
        pass

    def remove_request(self, rid):
        pass


class _PerRow:
    """An arena that hands every row of a step back to the manager's own loop."""

    def __init__(self, arena):
        self._arena = arena

    def __getattr__(self, name):
        return getattr(self._arena, name)

    def admit_rows(self, streams, segments, real, sink, page_size):
        return [], list(range(len(segments)))

    def commit_rows(self, streams, segments, real, sink, keep):
        return list(range(len(segments)))


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


@pytest.fixture(params=["rust", "python"])
def backend(request, monkeypatch):
    if request.param == "rust":
        if arena_mod.RustPageArena is None:
            pytest.skip("needs mstar_rust")
    else:
        monkeypatch.setattr(arena_mod, "RustPageArena", None)
    return request.param


def _manager() -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512,
            max_num_pages=1024, page_size=PAGE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _streams(kv: KVManager) -> dict:
    return {
        (rid, label): (s.page_indices.tolist(), s.stored_len, s.generation, s.step_in_flight)
        for rid, labels in kv._streams.items() for label, s in labels.items()
    }


@pytest.mark.parametrize("seed", range(8))
def test_the_row_passes_move_the_streams_as_the_per_row_path_does(seed, backend):
    rng = random.Random(seed)
    fast, slow = _manager(), _manager()
    slow._arena = _PerRow(slow._arena)
    live: list[str] = []
    for step_no in range(80):
        if rng.random() < 0.2 or not live:
            rid = f"r{step_no}"
            for kv in (fast, slow):
                kv.ingest_request(rid, KVReqConfig())
            live.append(rid)
        elif rng.random() < 0.08:
            rid = rng.choice(live)
            for kv in (fast, slow):
                kv.remove_request(rid)
            live.remove(rid)
            continue
        rows = rng.sample(live, rng.randrange(1, len(live) + 1))
        # decode, prefill and zero-span rows (only on a stream that exists:
        # a zero-span row reads one); a label a row has no stream for yet;
        # padding rows, and sometimes a sink row
        segs = []
        for r in rows:
            for label in LABELS:
                if label == "main" or rng.random() < 0.7:
                    spans = [1, 1, 1, rng.randrange(2, 3 * PAGE)] + [0] * (label in fast._streams[r])
                    segs.append(Segment(r, label, rng.choice(spans)))
        pads = [f"pad{i}" for i in range(rng.randrange(0, 3))]
        segs += [Segment(p, "main", 1) for p in pads]
        sink = frozenset({(rows[0], "aux")}) if rng.random() < 0.2 else frozenset()
        step = KVStep(segments=tuple(segs), sink=sink, commit=rng.random() < 0.9)
        preplan = rng.random() < 0.15
        for kv in (fast, slow):
            ctx = StepContext(
                request_ids=tuple(rows), graph_walk="decode", slot=0, capture=False, is_preplan=preplan,
            )
            if pads:
                ctx.set_padded_rids(tuple(rows) + tuple(pads))
            assert kv.admit(step, ctx).ok
            if preplan:
                # abandoned: its rows leave flight without committing
                kv.clear_preplan()
            else:
                kv.commit(step, ctx)
        assert _streams(fast) == _streams(slow), f"step {step_no}"
