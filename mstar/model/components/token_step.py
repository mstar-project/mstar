"""The parts of a decoder node that samples one token per decode step which
don't depend on the model: the decode capture, the batch caps and feeding the
sampled token back in. Its stop checks are ``ARNodeSubmodule``'s, from the
node's ``token_stop``."""
from __future__ import annotations

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.model.submodule_base import ARNodeInputs, CarryPair, NodeInputs


class TokenStepNode:
    """Mixin for an ``ARNodeSubmodule`` whose forward returns ``new_token``,
    which a batched step also hands its stop check (``check_stop_buffers``)
    whatever signal it routes the token as.

    A subclass binds ``SAMPLER`` to its sampler resource's key, sets
    ``MAX_PREFILL_BATCH_SIZE``, and sets ``token_stop`` in ``__init__`` as any
    ``ARNodeSubmodule`` does; it overrides the walk and loop names if its graph
    uses others.
    """

    DECODE_WALK = "decode"
    DECODE_LOOP = "decode_loop"
    # the walk that samples the first token, before DECODE_LOOP exists: a
    # subclass whose stop there ends the reply names it in ``prefill_walks``
    PREFILL_WALK = "prefill"
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    SAMPLER: str
    MAX_PREFILL_BATCH_SIZE: int | None

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

    def repeat_carry(self, graph_walk: str, signals) -> dict[str, CarryPair]:
        """The decode loop's token goes back in as ``text_inputs`` (``postprocess``),
        which ``preprocess`` stages as ``input_ids``: a run carries it from the
        sampled ``new_token`` there."""
        pairs = super().repeat_carry(graph_walk, signals)
        if "text_inputs" in pairs:
            pairs["text_inputs"] = CarryPair("new_token", "input_ids")
        return pairs

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # the decode loop feeds the sampled token back in as text_inputs: a
        # run's row holds a token a replay, of which the next step reads the last
        if "new_token" in outputs:
            token = outputs["new_token"][-1]
            outputs["text_inputs"] = [token if token.numel() == 1 else token.reshape(-1)[-1:]]
