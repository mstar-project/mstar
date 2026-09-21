from mstar.communication.tensors import TensorStore
from mstar.graph.runtime.base import (
    EdgeSpec,
    GraphRuntime,
    ParallelList,
    PopRidsOutput,
    ReadyNodeSpec,
    RouteInput,
    RouteOutput,
    SendInput,
    SpeculationOutput,
    SpeculationPrepInput,
    SpeculationPrepOutput,
)


class PythonGraphRuntime(GraphRuntime):
    """The reference implementation, and the thing a Rust backend is diffed
    against. Everything below is today's behavior behind the new contract; no
    new semantics belong here.
    """

    def __init__(self):
        self._rids: list[str | None] = []
        self._rid_to_handle: dict[str, int] = {}
        self._available_handles: list[int] = []

    # --------- Bookkeeping ----------

    def set_node_metadata(
        self, parallel_nodes: set[str],
        parallel_leader_nodes: set[str],
        tp_async_nodes: set[str]
    ):
        raise NotImplementedError

    def add_request(
        self, request_id: str,
        partition: str,
        graph_walk: str,
        partition_worker_graph_ids: list[int],
        worker_graph_to_worker: ParallelList[int, str]
    ) -> int:
        if not self._available_handles:
            handle = len(self._rids)
            self._rids.append(request_id)
        else:
            handle = self._available_handles.pop()
            self._rids[handle] = request_id

        # TODO: rest of the request ingestion
        self._rid_to_handle[request_id] = handle
        return handle

    def remove_request(
        self, rid: int
    ):
        request_id = self._rids[rid]
        if request_id is None:
            return  # already removed; remove is idempotent by design
        # TODO: rest of the request teardown
        del self._rid_to_handle[request_id]
        self._rids[rid] = None
        # Recycling means a stale handle held anywhere else now points at a
        # DIFFERENT request; see GraphRuntime.remove_request for what has to be
        # purged alongside this.
        self._available_handles.append(rid)

    def get_rid_string(self, handle: int) -> str:
        return self._rids[handle]

    def get_rid_handle(self, rid: str) -> int | None:
        return self._rid_to_handle.get(rid)

    def set_walk(self, rid: int, partition: str, walk: str):
        raise NotImplementedError

    def set_speculatively_scheduled(
        self, node: str, wg_id: int, rids: list[int],
        speculatively_scheduled: bool
    ):
        raise NotImplementedError

    def get_dynamic_loop_iters(
        self, request_ids: list[int],
        partition: str,
    ) -> ParallelList[int, dict[str, int]]:
        raise NotImplementedError

    def get_worker_graph_id_for_node(
        self, node: str, graph_walk: str,
    ) -> int:
        raise NotImplementedError

    # --------- Inputs ----------

    def ingest_inputs_batch(
        self,
        signals: ParallelList[int, EdgeSpec],
        can_buffer: bool = True,
        is_streaming: bool = False,
    ) -> list[int]:
        raise NotImplementedError

    # --------- Scheduling ----------

    def pop_rids(
        self, node_name: str,
        graph_walk: str,
        request_ids: list[int],
        check_ready: bool = False,
    ) -> PopRidsOutput | None:
        raise NotImplementedError

    def has_ready_excluding(
        self, exclude_rids: set[int],
        exclude_target: tuple[str, str] | None = None,
    ) -> bool:
        raise NotImplementedError

    def get_ready_nodes(
        self, exclude_rids: set[int],
        target: tuple[str, str] | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> list[ReadyNodeSpec]:
        raise NotImplementedError

    def push_back_node(
        self, node_name: str,
        rids: list[int],
        wg_ids: list[int]
    ):
        raise NotImplementedError

    # --------- Speculation ----------

    def speculate_node(
        self, node_name: str,
        graph_walk: str,
        sample_rid: int,
    ) -> list[SpeculationOutput]:
        raise NotImplementedError

    def prep_spec_rids(
        self, input: SpeculationPrepInput
    ) -> SpeculationPrepOutput:
        raise NotImplementedError

    # --------- Postprocess ----------

    def stop_loops_batched(
        self, partition: str, graph_walk: str,
        loop_names: ParallelList[int, list[str]]
    ):
        raise NotImplementedError

    def complete_and_route_batch(
        self, input: RouteInput,
        tensor_store: TensorStore
    ) -> RouteOutput:
        raise NotImplementedError

    def send_outputs(
        self,
        input: SendInput,
    ):
        raise NotImplementedError
