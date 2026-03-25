from dataclasses import dataclass

import torch

from mminf.communication.tensors import MooncakeCommunicationManager
from mminf.engine.paged_attention import KVRequestState, PageAllocator
from mminf.graph.base import GraphEdge
from mminf.utils.ipc_format import KVLayerTransfer, KVTransferMeta, WorkerMessage, WorkerMessageType


KV_TRANSFER_TO_DECODE = "KV_TRANSFER_TO_DECODE"

@dataclass
class PendingLayerTransfer:
    layer_idx: int
    label: str
    page_idxs: torch.Tensor


def _get_tensor_name(label: str, layer: int):
    return f"kv_cache_label_{label}_layer_{layer}"


class KVCacheSender:
    def __init__(
        self,
        request_id: str,
        decode_worker_id: str,
        tensor_manager: MooncakeCommunicationManager,
        kv_cache: torch.Tensor,
        delayed_transfer: bool=False # for cuda graphs, need to transfer at the end
                                     # after graph replay finishes 
    ):
        self.request_id = request_id
        self.decode_worker_id = decode_worker_id
        self.tensor_manager = tensor_manager
        self.communicator = tensor_manager.communicator
        self.kv_cache = kv_cache
        self.delayed_transfer = delayed_transfer
        self.labels = set()
        self.pending: list[PendingLayerTransfer] = []
    
    def _transfer(
        self,
        layer_idx: int,
        label: str,
        page_idxs: torch.Tensor,
        seq_len_at_start: int,
    ):
        page_idxs = page_idxs.tolist()
        layer_cache = [self.kv_cache[layer_idx, idx] for idx in page_idxs]
        name = _get_tensor_name(label, layer_idx)
        graph_edge = GraphEdge(
            next_node=KV_TRANSFER_TO_DECODE,
            name=name
        )
        self.tensor_manager.store_and_populate_graph_edges(
            request_id=self.request_id,
            tensors={
                name: layer_cache
            },
            graph_edges=[graph_edge]
        )

        uuids = [
            info.uuid for info in graph_edge.tensor_info
        ]

        self.tensor_manager.register_for_send(
            request_id=self.request_id,
            uuids=uuids
        )

        self.communicator.send(
            entity_id=self.decode_worker_id,
            msg=WorkerMessage(
                message_type=WorkerMessageType.KV_TRANSFER_LAYER,
                body=KVLayerTransfer(
                    request_id=self.request_id,
                    label=label,
                    kv_cache_graph_edge=graph_edge,
                    seq_len_at_start=seq_len_at_start
                )

            )
        )

    def transfer_layer(
        self, 
        layer_idx: int,
        label: str,
        page_idxs: torch.Tensor,
        seq_len_at_start: int,
    ):
        self.labels.add(label)
        if self.delayed_transfer:
            self.pending.append(PendingLayerTransfer(
                layer_idx=layer_idx,
                label=label,
                page_idxs=page_idxs
            ))
            return
        self._transfer(layer_idx, label, page_idxs, seq_len_at_start)

    def finish_transfer(
        self, num_layers: int,
        final_seq_len: dict[str,int],
        next_pos_id: dict[str, int],
        seq_len_at_start: dict[str, int]
    ):
        for pend in self.pending:
            self._transfer(pend.layer_idx, pend.label, pend.page_idxs)
        self.pending.clear()

        self.communicator.send(
            entity_id=self.decode_worker_id,
            msg=KVTransferMeta(
                request_id=self.request_id,
                labels=list(self.labels),
                num_layers=num_layers,
                final_seq_len=final_seq_len,
                next_pos_id=next_pos_id,
                seq_len_at_start=seq_len_at_start
            )
        )


@dataclass
class KVLayer:
    info: KVLayerTransfer
    # UUID -> tensor
    tensors: dict[str, torch.Tensor] | None = None


@dataclass
class WaitingDecodeRequest:
    graph_edge: GraphEdge
    pending_labels: set[str]


class KVCacheReceiver:
    def __init__(
        self,
        request_id: str,
        tensor_manager: MooncakeCommunicationManager,
        page_allocator: PageAllocator,
        kv_cache: torch.Tensor,
        page_size: int,
        request_states: dict[str, KVRequestState] = {}
    ):
        self.request_id = request_id
        self.tensor_manager = tensor_manager
        self.page_allocator = page_allocator
        self.kv_cache = kv_cache
        self.page_size = page_size
        self.request_states: dict[str, KVRequestState] = request_states

        # tensor name to KVLayer object
        self.waiting_for_tensors: dict[str, KVLayerTransfer]

        # both are dicts of (label, start_seq_len) -> list[KVLayer] | KVTransferMeta
        self.pending_layer_transfer: dict[tuple[str, int], list[KVLayer]] = {}
        self.pending_transfer_meta: dict[tuple[str, int], KVTransferMeta] = {}

        # i.e., for each label, the seq_len_at_start for any transfers that are
        # pending due to the KVRequestState sequence length not matching
        # seq_len_at_start. dtype is label -> set(start_seq_lens)
        self.pending_start_seq_lens: dict[str, set[int]] = {}

        # (label, start_seq_len) -> count
        self.layers_collected: dict[tuple[str, int], int] = {}

        # (label, start_seq_len) -> list of page idxs
        self.pages: dict[tuple[str, int], list[int]] = []

        # decode requests that are waiting for a KV cache
        self.waiting_decode_requests: dict[tuple[str, int], WaitingDecodeRequest] = {}
        self.ready_decode_requests: list[GraphEdge] = []
    
    def _get_request_state(self, label: str):
        if label not in self.request_states:
            self.request_states[label] = KVRequestState()
        return self.request_states[label]

    def _update_pages(self, layer: KVLayer):
        label = layer.info.label
        rs = self._get_request_state(label)

        tensor_uuids = [info.uuid for info in layer.info.kv_cache_graph_edge.tensor_info]

        key = (label, layer.info.seq_len_at_start)
        if key not in self.pages:
            self.pages[key] = []
            # allocate pages
            i = 0
            if rs.seq_len % self.page_size != 0:
                self.pages[key].append(rs.page_indices[-1])
                i = 1
            new_page_count = len(tensor_uuids) - i
            if new_page_count > 0:
                self.pages[key].extend(self.page_allocator.allocate(new_page_count))

        for i, page_idx in enumerate(self.pages[key]):
            self.kv_cache[layer.info.layer_idx, page_idx] = layer.tensors[tensor_uuids[i]]

        self.layers_collected[key] = self.layers_collected.get(0) + 1
    
    def _clear_from_pending_layer_transfer(
        self, label: str, seq_len_at_start: int
    ):
        key = (label, seq_len_at_start)
        for layer in self.pending_layer_transfer[key]:
            self._update_pages(layer, key)

    def _update_kv_state(self, label: str):
        rs = self._get_request_state(label)
        key = (label, rs.seq_len)
        self._clear_from_pending_layer_transfer(*key)
        meta = self.pending_transfer_meta[key]
        rs.position_id_start = meta.next_pos_id[label]
        rs.seq_len = meta.final_seq_len[label]
        rs.page_indices.extend(self.pages[key])

        del self.pages[key]
        del self.pending_transfer_meta[key]
        del self.pending_layer_transfer[key]

        prev_start_seq_len = key[1]
        if prev_start_seq_len in self.pending_start_seq_lens[label]:
            self.pending_start_seq_lens[label].remove(prev_start_seq_len)
    
    def _apply_all_possible(
        self, label: str, seq_len_at_start: int
    ):
        key = (label, seq_len_at_start)
        self._clear_from_pending_layer_transfer(*key)
        if key not in self.pending_transfer_meta:
            return
        meta = self.pending_transfer_meta[key]
        assert meta.final_seq_len[label] > meta.seq_len_at_start[label]
        if self.layers_collected.get(key, 0) == meta.num_layers:
            self._update_kv_state(label)
        
        rs = self.request_states[label]
        if rs.seq_len in self.pending_start_seq_lens.get(label, set()):
            self._apply_all_possible(label, rs.seq_len) # recursive call with new seq_len

    def ingest_layer_meta(self, meta: KVTransferMeta):
        for label in meta.labels:
            key = (label, meta.seq_len_at_start[label])
            self.pending_transfer_meta[key] = meta

            rs = self._get_request_state(label)
            if rs.seq_len < meta.seq_len_at_start[label]:
                self.pending_start_seq_lens.setdefault(label, set()).add(meta.seq_len_at_start[label])
                continue
            self._apply_all_possible(label, meta.seq_len_at_start[label])
    
    def ingest_layer_transfer(
        self, layer_transfer: KVLayerTransfer
    ):
        self.waiting_for_tensors[layer_transfer.kv_cache_graph_edge.name]  = layer_transfer

    def ingest_layer_tensors(
        self, name: str, tensors: dict[str, torch.Tensor]
    ):
        assert name in self.waiting_for_tensors
        layer_info = self.waiting_for_tensors[name]
        del self.waiting_for_tensors[name]

        key = (layer_info.label, layer_info.seq_len_at_start)
        self.pending_layer_transfer.setdefault(key, []).append(
            KVLayer(info=layer_info, tensors=tensors)
        )

        rs = self._get_request_state(layer_info.label)
        if rs.seq_len < layer_info.seq_len_at_start:
            self.pending_start_seq_lens.setdefault(
                layer_info.label, set()
            ).add(layer_info.seq_len_at_start)
            return
        self._apply_all_possible(layer_info.label, layer_info.seq_len_at_start)
    
    # def ingest_decode_graph_edge(self, edge: GraphEdge, prefill_seq_len: dict):

