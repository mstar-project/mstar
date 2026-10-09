"""The row store: a decode step's outputs are row views of one batch clone,
stored from one row's description and the stop check's host rows."""
import pytest
import torch

from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.model.submodule_base import BatchedModelOutput, HostRows

SIGNALS = ["text_inputs", "new_token"]


def _manager(tmp_path):
    class _NullCommunicator:
        def send(self, *args, **kwargs):
            pass

    return SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=_NullCommunicator(), shm_dir=str(tmp_path),
    )


def _columnar(mgr):
    """The row store only exists for the columnar (mstar_rust) tensor store;
    without it the manager takes the general path, which is tested elsewhere."""
    if not mgr.tensor_store.has_put_tensor_batch_columns:
        pytest.skip("the row store needs the columnar tensor store (mstar_rust)")
    return mgr


def _batch(rids):
    rows = torch.arange(10, 10 + len(rids), dtype=torch.long).view(len(rids), 1)
    clone = rows.clone()
    views = clone.split(1)
    per_rid = {}
    for i, rid in enumerate(rids):
        per_rid[rid] = {"new_token": [views[i]]}
        # the model's postprocess rebinds the loop-back name to the same list
        per_rid[rid]["text_inputs"] = per_rid[rid]["new_token"]
    outputs = BatchedModelOutput(
        per_rid_outputs=per_rid, row_views={"new_token": views},
        row_request_ids=tuple(rids),
    )
    host = rows.clone()
    host_rows = HostRows(request_ids=tuple(rids), buffers={"new_token": host})
    cpu_per_rid = {
        rid: {"new_token": [host[i : i + 1]], "text_inputs": [host[i : i + 1]]}
        for i, rid in enumerate(rids)
    }
    return outputs, host_rows, cpu_per_rid


def test_row_store_matches_the_general_path(tmp_path):
    rids = [3, 5, 8, 13]
    outputs, host_rows, cpu_per_rid = _batch(rids)
    fast_mgr, slow_mgr = _columnar(_manager(tmp_path / "a")), _manager(tmp_path / "b")
    fast = fast_mgr.store_row_outputs_batch(
        rids, outputs, SIGNALS, outputs.row_views, outputs.row_request_ids,
        node_name="text", graph_walk="decode", host_rows=host_rows,
    )
    slow = slow_mgr.store_and_return_tensor_info_batch(
        rids, outputs, SIGNALS, node_name="text", graph_walk="decode",
        cpu_tensors=cpu_per_rid,
    )
    assert fast is not None
    assert fast.flat_rids == slow.flat_rids == [3, 3, 5, 5, 8, 8, 13, 13]
    assert fast.signal_idxs == slow.signal_idxs == [0, 1] * 4
    assert fast.num_tensors == slow.num_tensors == [1] * 8
    assert len(set(fast.flat_uuids)) == 8
    for uf, us in zip(fast.flat_uuids, slow.flat_uuids, strict=True):
        tf, ts = fast_mgr.get_tensor(uf), slow_mgr.get_tensor(us)
        assert tf.data_ptr() == ts.data_ptr() and tf.shape == ts.shape
        assert tf.dtype == ts.dtype and tf.stride() == ts.stride()
        info_f, info_s = fast_mgr.tensor_store.get_info(uf), slow_mgr.tensor_store.get_info(us)
        assert (info_f.dims, info_f.stride, info_f.nbytes, info_f.dtype, info_f.address) == (
            info_s.dims, info_s.stride, info_s.nbytes, info_s.dtype, info_s.address
        )
        cf = fast_mgr.tensor_store._tensors_cpu.get(uf)
        cs = slow_mgr.tensor_store._tensors_cpu.get(us)
        assert cf is not None and cs is not None
        assert torch.equal(cf, cs) and cf.data_ptr() == cs.data_ptr()
    for rid in rids:
        assert len(fast_mgr.tensor_store.get_all_uuids(rid)) == 2


def test_row_store_declines_what_is_not_a_row_view(tmp_path):
    rids = [1, 2, 3]
    outputs, host_rows, _ = _batch(rids)
    mgr = _manager(tmp_path)
    # one request's output was cloned per request: not a row of the batch
    outputs.per_rid_outputs[2]["new_token"] = [outputs.per_rid_outputs[2]["new_token"][0].clone()]
    outputs.per_rid_outputs[2]["text_inputs"] = outputs.per_rid_outputs[2]["new_token"]
    assert mgr.store_row_outputs_batch(
        rids, outputs, SIGNALS, outputs.row_views, outputs.row_request_ids,
        host_rows=host_rows,
    ) is None
    assert mgr.tensor_store.get_all_uuids(1) == []

    # a request that has no row at all
    outputs2, host_rows2, _ = _batch(rids)
    assert mgr.store_row_outputs_batch(
        rids + [9], outputs2, SIGNALS, outputs2.row_views, outputs2.row_request_ids,
        host_rows=host_rows2,
    ) is None


def test_row_store_keeps_the_general_path_shape_for_missing_signals(tmp_path):
    rids = [1, 2, 3]
    outputs, host_rows, cpu_per_rid = _batch(rids)
    del outputs.per_rid_outputs[2]["text_inputs"]
    mgr = _columnar(_manager(tmp_path))
    fast = mgr.store_row_outputs_batch(
        rids, outputs, SIGNALS, outputs.row_views, outputs.row_request_ids,
        host_rows=host_rows,
    )
    assert fast is not None
    assert fast.num_tensors == [1, 1, 0, 1, 1, 1]
    assert fast.flat_rids == [1, 1, 2, 3, 3]
    assert fast.signal_idxs == [0, 1, 1, 0, 1]


def test_row_store_without_host_rows_stores_no_host_copy(tmp_path):
    rids = [1, 2]
    outputs, _, _ = _batch(rids)
    mgr = _columnar(_manager(tmp_path))
    fast = mgr.store_row_outputs_batch(
        rids, outputs, SIGNALS, outputs.row_views, outputs.row_request_ids,
    )
    assert fast is not None and len(fast.flat_uuids) == 4
    assert not mgr.tensor_store._tensors_cpu
