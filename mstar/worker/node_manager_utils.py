import logging
from dataclasses import dataclass, field

from mstar.conductor.request_info import CurrentForwardPassInfo, merge_publish_info
from mstar.engine.resources import PublishedInfo
from mstar.streaming.stream_buffer import StreamBuffer, StreamChunkInfo

logger = logging.getLogger(__name__)


@dataclass
class PerPartitionInfo:
    current_fwd_info: CurrentForwardPassInfo
    # Publication produced by this worker since its last completion message.
    # Kept separate from current_fwd_info, which also contains peer state.
    pending_resource_publish_info: dict[str, PublishedInfo] = field(
        default_factory=dict
    )


@dataclass
class PerRequestInfo:
    """What the worker still owns for a request.

    Everything graph-shaped -- node/loop routing, the sharding config, the
    worker graph ids, the live queues -- belongs to the graph runtime. What
    is left is the forward-pass info (a wire object) and the stream buffers
    (which hold real tensors), neither of which can live behind the runtime's
    contract.
    """
    # edge_name -> StreamBuffer
    stream_buffers: dict[str, StreamBuffer] = field(default_factory=dict)
    # consumer node -> edge_name -> StreamBuffer, the same buffers indexed by consumer
    stream_buffers_by_consumer: dict[str, dict[str, StreamBuffer]] = field(default_factory=dict)
    # edges whose final chunk a built step consumed; a returned chunk leaves again
    ended_streams: set[str] = field(default_factory=set)
    per_partition_info: dict[str, PerPartitionInfo] = field(default_factory=dict)


@dataclass
class RequestStateManager:
    """Per-request forward-pass info and stream buffers.

    What remains here is request state the graph runtime deliberately does not
    take: CurrentForwardPassInfo is a wire object it treats as opaque, and a
    StreamBuffer holds tensors.
    """
    per_request_info: dict[int, PerRequestInfo] = field(default_factory=dict)

    # node_name -> partition_name, from the model's partitions and graph walk
    # definitions. Used to find a node's partition in the colocated case.
    node_to_partition: dict[str, str] = field(default_factory=dict)

    # partition -> rid -> current_fwd_info, the same objects as above, so a
    # batch's infos are one C-level map rather than three lookups per row.
    # Handle-keyed: purged in remove_request.
    _fwd_by_partition: dict[str, dict[int, CurrentForwardPassInfo]] = field(
        default_factory=dict, repr=False,
    )

    def add_request(self, request_id: int, current_fwd_info: CurrentForwardPassInfo):
        """Register a partition for a request, adding the request if new.

        The conductor sends one NewRequest per partition, so this is called
        once per partition for a given request_id.
        """
        info = self.per_request_info.setdefault(request_id, PerRequestInfo())
        partition = current_fwd_info.partition_name
        info.per_partition_info[partition] = (
            PerPartitionInfo(current_fwd_info=current_fwd_info)
        )
        self._fwd_by_partition.setdefault(partition, {})[request_id] = current_fwd_info

    def remove_request(self, request_id: int):
        self.per_request_info.pop(request_id, None)
        for by_rid in self._fwd_by_partition.values():
            by_rid.pop(request_id, None)

    def update_request_info(
        self, request_id: int,
        partition_name: str,
        current_fwd_info: CurrentForwardPassInfo | None = None,
        resource_publish_info: dict[str, PublishedInfo] | None = None,
    ):
        info = self.per_request_info.get(request_id)
        if info is None:
            return
        part_info = info.per_partition_info.get(partition_name)
        if part_info is None:
            return
        if current_fwd_info is not None:
            part_info.current_fwd_info = current_fwd_info
            self._fwd_by_partition.setdefault(partition_name, {})[request_id] = (
                current_fwd_info
            )
        if resource_publish_info is not None:
            part_info.current_fwd_info.update_publish_info(resource_publish_info)

    def buffer_publish_info(
        self,
        request_id: int,
        partition_name: str,
        published: dict[str, PublishedInfo],
    ) -> None:
        pending = self.per_request_info[request_id].per_partition_info[
            partition_name
        ].pending_resource_publish_info
        # finalize_batch also merges these objects into current_fwd_info.
        # Detach the pending delta so a later peer update to that aggregate
        # cannot leak another rank into this worker's completion message.
        merge_publish_info(
            pending, {name: info.clone() for name, info in published.items()}
        )

    def flush_publish_info(
        self, request_id: int, partition_name: str,
    ) -> dict[str, PublishedInfo]:
        part_info = self.per_request_info[request_id].per_partition_info[
            partition_name
        ]
        published = part_info.pending_resource_publish_info
        part_info.pending_resource_publish_info = {}
        return published

    def get_fwd_info(self, request_id: int, partition_name: str):
        return self.per_request_info[request_id].per_partition_info[
            partition_name
        ].current_fwd_info

    def fwd_infos(
        self, request_ids: list[int], partition_name: str,
    ) -> list[CurrentForwardPassInfo]:
        """``get_fwd_info`` for each rid, in order; KeyError if one has none."""
        return list(map(self._fwd_by_partition[partition_name].__getitem__, request_ids))

    def get_pending_publish_info(
        self, request_id: int, partition_name: str,
    ) -> dict[str, PublishedInfo]:
        return self.per_request_info[request_id].per_partition_info[
            partition_name
        ].pending_resource_publish_info

    def get_partition_for_node(self, node_name: str) -> str | None:
        return self.node_to_partition.get(node_name)

    def ingested_stream_chunks(self, request_id: int, node_name: str) -> dict[str, StreamChunkInfo]:
        """Edge name -> the chunk each of ``node_name``'s input streams last
        handed the graph: the one the request's next step there consumes."""
        req_info = self.per_request_info.get(request_id)
        if req_info is None:
            return {}
        return {
            name: sbuf.ingested_chunk
            for name, sbuf in req_info.stream_buffers_by_consumer.get(node_name, {}).items()
            if sbuf.ingested_chunk is not None
        }
