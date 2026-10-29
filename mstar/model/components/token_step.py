"""The parts of a decoder node that samples one token per decode step which
don't depend on the model: the decode capture, the batch caps, feeding the
sampled token back in, and the stop checks."""
from __future__ import annotations

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.model.submodule_base import END_PARTITION, ARNodeInputs, HostRows, NodeInputs


class TokenStepNode:
    """Mixin for an ``ARNodeSubmodule`` whose forward returns ``new_token``.

    A subclass binds ``SAMPLER`` to its sampler resource's key and sets
    ``MAX_PREFILL_BATCH_SIZE`` and ``stop_token_ids``; it overrides the walk
    and loop names if its graph uses others.
    """

    DECODE_WALK = "decode"
    DECODE_LOOP = "decode_loop"
    # The walk that samples the first token, before DECODE_LOOP exists. A stop
    # there ends the partition (END_PARTITION); a stop id on any other walk
    # outside the loop stops nothing, since the model's transition may still
    # have a walk to run.
    PREFILL_WALK = "prefill"
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    SAMPLER: str
    MAX_PREFILL_BATCH_SIZE: int | None
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
        # the base default, which subclasses may narrow: the engine caps decode at
        # its largest capture config (uncapped with accelerator graphs off)
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
        """A stop id, or the token budget, on a PREFILL_WALK or decode row.
        The budget counts the token PREFILL_WALK samples."""
        walk = request_info.graph_walk
        if walk == self.DECODE_WALK:
            generated = request_info.dynamic_loop_iter_counts.get(self.DECODE_LOOP, 0) + 2
        elif walk == self.PREFILL_WALK:
            generated = 1
        else:
            return False
        ignore_eos = request_info.resource_configs[self.SAMPLER].ignore_eos
        return (not ignore_eos and token in self.stop_token_ids) or generated >= request_info.max_tokens

    def _stop_names(self, token: int, request_info: CurrentForwardPassInfo) -> set[str]:
        """``check_stop``'s answer for one row: a decode row stops the loop,
        and a PREFILL_WALK row, which runs before the loop exists, ends the
        partition."""
        if not self._stops(token, request_info):
            return set()
        return {self.DECODE_LOOP} if request_info.graph_walk == self.DECODE_WALK else {END_PARTITION}

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        # a batched decode step routes its token as text_inputs alone
        tokens = outputs.get("text_inputs") or outputs.get("new_token")
        if not tokens:
            return set()
        return self._stop_names(tokens[0].item(), request_info)

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
        step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        """``check_stop`` for every request off one ``tolist``."""
        pairs = host_rows.first_values("new_token", request_ids)
        if pairs is None:
            return None
        stops = {}
        for rid, token in pairs:
            if names := self._stop_names(token, request_infos[rid]):
                stops[rid] = names
        return stops
