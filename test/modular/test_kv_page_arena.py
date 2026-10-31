"""The two page arena backends against a plain-list model of each stream, and
against each other: the same operations give the same pages, owners, seals
and plan arrays."""
from __future__ import annotations

import random

import numpy as np
import pytest

from mstar.engine.resources.kv.arena import RustPageArena, new_page_arena, plan_columns

BACKENDS = ["python", pytest.param("rust", marks=pytest.mark.skipif(RustPageArena is None, reason="needs mstar_rust"))]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", [0, 1, 20261101])
def test_stream_pages_read_and_write_like_a_list(backend, seed):
    rng = random.Random(seed)
    arena = new_page_arena(64, backend)
    streams = [arena.new_stream() for _ in range(6)]
    model = [[] for _ in streams]
    for _ in range(400):
        i = rng.randrange(len(streams))
        pages, want = streams[i], model[i]
        op = rng.choice(["extend", "assign", "clear", "del", "delslice", "free"])
        if op == "extend":
            new = [rng.randrange(64) for _ in range(rng.randrange(0, 40))]
            pages.extend(new)
            want.extend(new)
        elif op == "assign":
            new = [rng.randrange(64) for _ in range(rng.randrange(0, 20))]
            pages.assign(new)
            want[:] = new
        elif op == "clear":
            pages.clear()
            want.clear()
        elif op == "free":
            pages.free()
            want.clear()
        elif op == "del" and want:
            k = rng.randrange(-len(want), len(want))
            del pages[k]
            del want[k]
        elif op == "delslice":
            a, b = rng.randrange(-5, 50), rng.randrange(-5, 50)
            del pages[a:b]
            del want[a:b]
        assert pages.tolist() == want and len(pages) == pages.n == len(want)
        assert list(pages) == want and list(reversed(pages)) == want[::-1]
        assert pages == want and pages == tuple(want) and not (pages != want)
        for a, b, step in [(0, None, None), (2, 9, None), (-3, None, None), (None, None, 2), (5, 1, -1)]:
            assert pages[a:b:step] == want[a:b:step]
        if want:
            k = rng.randrange(-len(want), len(want))
            assert pages[k] == want[k]
        with pytest.raises(IndexError):
            pages[len(want)]


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.skipif(RustPageArena is None, reason="needs mstar_rust")
def test_both_backends_hand_out_the_same_pages_and_plan_the_same(seed):
    rng = random.Random(seed)
    arenas = [new_page_arena(48, "python"), new_page_arena(48, "rust")]
    streams = [[arena.new_stream() for _ in range(8)] for arena in arenas]
    held: list[list[int]] = [[] for _ in range(8)]
    for _ in range(300):
        i = rng.randrange(8)
        op = rng.choice(["grow", "grow", "share", "drop", "plan"])
        if op == "grow":
            n = rng.randrange(1, 4)
            got = [arena.acquire(n) for arena in arenas]
            assert got[0] == got[1]
            if got[0] is not None:
                for s in streams:
                    s[i].extend(got[0])
                held[i] += got[0]
        elif op == "share" and held[i]:
            j = rng.randrange(8)
            for arena, s in zip(arenas, streams, strict=True):
                arena.seal(s[i])
                arena.retain(s[i])
                s[j].extend(s[i].tolist())
            held[j] += held[i]
        elif op == "drop" and held[i]:
            for arena, s in zip(arenas, streams, strict=True):
                arena.release(s[i])
                s[i].clear()
            held[i] = []
        elif op == "plan":
            rows = rng.sample(range(8), rng.randrange(1, 8))
            lens = [max(0, len(held[r]) * 4 - rng.randrange(0, 6)) for r in rows]
            spans = [rng.randrange(0, 3) for _ in rows]
            cols = [
                plan_columns(arena, [s[r].sid for r in rows], lens, spans, 4)
                for arena, s in zip(arenas, streams, strict=True)
            ]
            for a, b in zip(cols[0], cols[1], strict=True):
                assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
        for name in ("num_free", "max_num_pages"):
            assert getattr(arenas[0], name) == getattr(arenas[1], name)
        assert arenas[0].free_pages() == arenas[1].free_pages()
        everything = range(48)
        assert arenas[0].owners_of(everything) == arenas[1].owners_of(everything)
        assert [a.is_sealed(p) for a in arenas for p in everything] == [
            arenas[0].is_sealed(p) for _ in arenas for p in everything
        ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_index_hears_each_page_that_gains_or_loses_a_second_owner(backend):
    arena = new_page_arena(8, backend)
    heard = []

    class Index:
        def set_shared(self, page, shared):
            # the counts of the pages before it are already updated
            heard.append((page, shared, arena.owners_of([0, 1, 2])))

    arena.index = Index()
    pages = arena.acquire(3)
    arena.retain(pages[:2])
    arena.retain(pages[:1])
    arena.release(pages)
    arena.release(pages[:2])
    # page 0 had three owners, so the first release unshares only page 1
    assert heard == [
        (0, True, [2, 1, 1]), (1, True, [2, 2, 1]),
        (1, False, [2, 1, 1]), (0, False, [1, 1, 0]),
    ]
    # pages that reach zero go back after the whole release, in its order
    assert arena.free_pages()[-2:] == [2, 1]
    arena.release(pages[:1])
    assert arena.free_pages()[-3:] == [2, 1, 0]
    with pytest.raises(AssertionError):
        arena.release(pages[:1])
