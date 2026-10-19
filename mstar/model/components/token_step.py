"""The parts of a decoder node that samples one token per decode step which
don't depend on the model: the decode capture, the batch caps, feeding the
sampled token back in, and the stop checks."""
from __future__ import annotations

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.model.submodule_base import ARNodeInputs, HostRows, NodeInputs


class TokenStepNode:
    """Mixin for an ``ARNodeSubmodule`` whose forward returns ``new_token``.

    A subclass binds ``SAMPLER`` to its sampler resource's key and sets
    ``MAX_PREFILL_BATCH_SIZE`` and ``stop_token_ids``; it overrides the walk
    and loop names if its graph uses others.
    """

    DECODE_WALK = "decode"
    DECODE_LOOP = "decode_loop"
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    SAMPLER: str
    MAX_PREFILL_BATCH_SIZE: int
    stop_token_ids: frozenset[int]

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=self.DECODE_WALK,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
        ]

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        # decode is bounded by the largest capture; prefill isn't captured
        return None if graph_walk == self.DECODE_WALK else self.MAX_PREFILL_BATCH_SIZE

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # the decode loop feeds the sampled token back in as text_inputs
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def _stops(self, token: int, request_info: CurrentForwardPassInfo) -> bool:
        ignore_eos = request_info.resource_configs[self.SAMPLER].ignore_eos
        decoded = request_info.dynamic_loop_iter_counts.get(self.DECODE_LOOP, 0) + 1
        return (not ignore_eos and token in self.stop_token_ids) or decoded >= request_info.max_tokens

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        return {self.DECODE_LOOP} if self._stops(outputs["new_token"][0].item(), request_info) else set()

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
        step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        """``check_stop`` for every request off one ``tolist``. ``step_walk``
        changes nothing: as in ``check_stop``, prefill's token can end the
        decode loop too."""
        tokens = host_rows.buffers.get("new_token")
        if not torch.is_tensor(tokens) or tokens.dim() == 0:
            return None
        values = tokens.reshape(tokens.shape[0], -1)[:, 0].tolist()
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        stops: dict[str, set[str]] = {}
        for rid in request_ids:
            i = row_of.get(rid)
            if i is not None and i < len(values) and self._stops(values[i], request_infos[rid]):
                stops[rid] = {self.DECODE_LOOP}
        return stops
