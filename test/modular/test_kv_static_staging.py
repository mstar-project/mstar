"""A captured step's write addressing is staged in one copy into its slot's
buffer (pages then offsets, padded with SINK_PAGE / 0), and not at all when
the buffer already holds exactly that; a slot's capture lengths share the
buffer, so another length's staging in between forces the copy again."""
from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("the stager's skip is on the pinned H2D path, which needs CUDA", allow_module_level=True)

from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.plan import SINK_PAGE  # noqa: E402


class _StubTransfer:
    def __init__(self, *a, **k):
        pass

    def cleanup(self):
        pass


@pytest.fixture
def kv(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = KVManager(
        cfg=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=256, max_num_pages=64, page_size=4),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cuda"), dtype=torch.float32,
    )
    kv._cg_max_seq_len = 16
    return kv


def _staged(kv, state):
    torch.cuda.synchronize()
    return state.token_to_page.tolist(), state.token_to_cache.tolist()


def test_one_copy_padded_and_skipped_when_unchanged(kv):
    s4 = kv._static_plan_state(0, "main", 4)
    assert s4.staging.data_ptr() == s4.token_to_page.data_ptr()
    kv._stage_static(s4, [5, 6], [1, 2], 2)
    assert _staged(kv, s4) == ([5, 6, SINK_PAGE, SINK_PAGE], [1, 2, 0, 0])
    assert s4.total_tokens == 2
    copies = kv._plan_stager.copies
    kv._stage_static(s4, [5, 6], [1, 2], 2)
    assert kv._plan_stager.copies == copies, "the same addressing staged twice copied twice"
    kv._stage_static(s4, [5, 7], [1, 2], 2)
    assert kv._plan_stager.copies == copies + 1
    assert _staged(kv, s4) == ([5, 7, SINK_PAGE, SINK_PAGE], [1, 2, 0, 0])


def test_capture_lengths_share_a_slots_buffer(kv):
    s2, s4 = kv._static_plan_state(0, "main", 2), kv._static_plan_state(0, "main", 4)
    assert s2.staging.data_ptr() == s4.staging.data_ptr()
    kv._stage_static(s4, [9, 9, 9, 9], [3, 3, 3, 3], 4)
    kv._stage_static(s2, [5], [1], 1)
    assert _staged(kv, s2) == ([5, SINK_PAGE], [1, 0])
    # s4's values were overwritten by s2's: staging them again must copy
    copies = kv._plan_stager.copies
    kv._stage_static(s4, [9, 9, 9, 9], [3, 3, 3, 3], 4)
    assert kv._plan_stager.copies == copies + 1
    assert _staged(kv, s4) == ([9, 9, 9, 9], [3, 3, 3, 3])
    # another slot or label has its own buffer
    assert kv._static_plan_state(1, "main", 4).staging.data_ptr() != s4.staging.data_ptr()
    assert kv._static_plan_state(0, "ctx", 4).staging.data_ptr() != s4.staging.data_ptr()
    with pytest.raises(ValueError):
        kv._stage_static(s2, [1, 2, 3], [0, 0, 0], 3)


def test_an_eager_write_uploads_its_addressing_in_one_copy(kv):
    """Two rows appending 3 and 2 tokens after 5 and 0 stored ones (page size
    4): the addressing lands on the device whole, from one stager copy."""
    from mstar.engine.resources.kv.plan import PagedIndptrs

    i32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    indptrs = PagedIndptrs(
        qo_indptr=i32([0, 3, 5]), paged_kv_indptr=i32([0, 2, 3]), paged_kv_indices=i32([7, 9, 4]),
        paged_kv_last_page_len=i32([4, 2]), kv_lens=None,
    )
    copies = kv._plan_stager.copies
    state = kv._host_plan_state(indptrs, 5)
    assert kv._plan_stager.copies == copies + 1
    torch.cuda.synchronize()
    assert state.token_to_page.is_cuda and state.total_tokens == 5
    assert state.token_to_page.tolist() == [9, 9, 9, 4, 4]
    assert state.token_to_cache.tolist() == [1, 2, 3, 0, 1]
