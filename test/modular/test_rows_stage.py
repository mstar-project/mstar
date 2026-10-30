"""A step's stage packs the same block on both backends: the Rust
``RowStage`` and ``PyRowStage``, fed the same calls, give the same segment
tables, payload and placement."""
import random

import numpy as np
import pytest

from mstar.engine.rows import stage

mstar_rust = pytest.importorskip("mstar_rust")
if not hasattr(mstar_rust, "RowStage"):
    pytest.skip("the Rust extension predates RowStage", allow_module_level=True)


def _calls(rng: random.Random, phases: int):
    """A random step's registrations, as (method, args)."""
    calls = []
    for _ in range(rng.randint(0, 12)):
        kind = rng.choice(["copy", "write", "write_at", "put"])
        n = rng.randint(0, 5)
        slots = np.array(rng.sample(range(40), n), dtype=np.int64)
        rb = rng.choice([4, 8, 24, 64])
        base = rng.randrange(1 << 40)
        if kind == "copy":
            calls.append(("copy_rows", (
                rng.randrange(phases), base, rb, slots, rng.randrange(1 << 40), rng.choice([1, rb]),
                np.array([rng.randrange(1000) for _ in range(n)], dtype=np.int64), rb,
            )))
        elif kind == "write":
            values = np.frombuffer(rng.randbytes(n * rb), dtype=np.uint8)
            calls.append(("write", (base, rb, slots, values, rb)))
        elif kind == "write_at":
            values = np.frombuffer(rng.randbytes(n * 4), dtype=np.uint8)
            offsets = np.array([4 * rng.randrange(rb // 4) for _ in range(n)], dtype=np.int64)
            calls.append(("write", (base, rb, slots, values, 4, offsets)))
        else:
            calls.append(("put", (np.frombuffer(rng.randbytes(rng.randint(1, 300)), dtype=np.uint8), base)))
    return calls


@pytest.mark.parametrize("seed", range(40))
def test_both_stages_pack_the_same_block(seed):
    rng = random.Random(seed)
    phases = rng.randint(1, 3)
    calls = _calls(rng, phases)
    stages = [stage.PyRowStage(phases), mstar_rust.RowStage(phases)]
    for st in stages:
        for name, args in calls:
            getattr(st, name)(*args)
    py, rs = stages
    assert py.is_empty() == rs.is_empty()
    assert py.size() == rs.size()
    base = rng.randrange(1 << 40)
    blocks = [np.zeros(py.size() + 32, dtype=np.uint8) for _ in stages]
    placed = [st.pack(block, base) for st, block in zip(stages, blocks, strict=True)]
    assert [tuple(p) for p in placed[1]] == [tuple(p) for p in placed[0]]
    np.testing.assert_array_equal(blocks[1], blocks[0])


def test_both_refuse_the_same_misuse():
    for make in (stage.PyRowStage, mstar_rust.RowStage):
        with pytest.raises(ValueError):
            make(0)
        st = make(2)
        with pytest.raises(ValueError):
            st.copy_rows(2, 0, 8, np.zeros(0, np.int64), 0, 8, np.zeros(0, np.int64), 8)
        with pytest.raises(ValueError):
            st.write(0, 8, np.arange(2, dtype=np.int64), np.zeros(8, np.uint8), 8)
        with pytest.raises(ValueError):
            st.write(0, 8, np.arange(4, dtype=np.int64)[::2], np.zeros(16, np.uint8), 8)
        with pytest.raises(ValueError):
            st.write(0, 8, np.arange(2, dtype=np.int64), np.zeros(8, np.uint8), 4, np.zeros(3, np.int64))
        with pytest.raises(ValueError):
            st.copy_rows(0, 0, 8, np.arange(2, dtype=np.int64), 0, 8, np.arange(3, dtype=np.int64), 8)
        assert st.size() == 0, "a refused call registers nothing"
        st.put(np.ones(40, dtype=np.uint8), 0)
        with pytest.raises(ValueError):
            st.pack(np.zeros(st.size() - 1, dtype=np.uint8), 0)


def test_both_copy_the_values_when_they_are_written():
    """A caller may reuse its buffer for the next write."""
    def block(make, reuse):
        st = make(1)
        values = np.full(8, 1, np.uint8)
        st.write(0, 8, np.array([0], np.int64), values, 8)
        if reuse:
            values.fill(2)
        else:
            values = np.full(8, 2, np.uint8)
        st.write(0, 8, np.array([1], np.int64), values, 8)
        out = np.zeros(st.size(), np.uint8)
        st.pack(out, 0)
        return out

    for make in (stage.PyRowStage, mstar_rust.RowStage):
        assert np.array_equal(block(make, reuse=True), block(make, reuse=False))
