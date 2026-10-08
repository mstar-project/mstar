"""StepContext row tests: constant-time membership that follows the request
list through padding and rewrites."""
from mstar.engine.resources.step import StepContext


def _ctx(rids):
    return StepContext(request_ids=list(rids), graph_walk="decode", slot=0, capture=False)


def test_padding_rows_are_the_padded_tail():
    ctx = _ctx([5, 7])
    assert not ctx.is_padding_row(5) and not ctx.is_padding_row(99)
    assert ctx.is_real_row(7) and not ctx.is_real_row(99)
    ctx.set_padded_rids([5, 7, -1, -2])
    assert [ctx.is_padding_row(r) for r in (5, 7, -1, -2)] == [False, False, True, True]
    assert ctx.padded_request_ids == [5, 7, -1, -2]


def test_rewriting_the_request_list_refreshes_the_set():
    ctx = _ctx([1, 2, 3])
    ctx.set_padded_rids([1, 2, 3, -1])
    assert ctx.is_padding_row(-1) and not ctx.is_padding_row(2)
    # a dropped request: the batch rewrites the list and clears the padding
    ctx.set_padded_rids(None)
    ctx.request_ids = [1, 3]
    assert not ctx.is_padding_row(2)  # no padded list: nothing is padding
    assert not ctx.is_real_row(2) and ctx.is_real_row(3)
    ctx.set_padded_rids([1, 3, -1, -2])
    assert ctx.is_padding_row(2) and ctx.is_padding_row(-2) and not ctx.is_padding_row(3)
