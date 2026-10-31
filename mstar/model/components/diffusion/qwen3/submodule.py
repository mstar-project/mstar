"""The ``text_encoder`` node of the scaffold's Qwen3-conditioned DiTs (FLUX.2 [klein], Z-Image).

The prompt reaches the node unpadded (``text_inputs [L]``) and is right-padded on the
device to the encoder's ``max_sequence_length``, as the pipelines pad it: the real
tokens' values do not depend on the padding mathematically, but the attention kernel's
accumulation order does, so the padded length is part of matching the reference.
Requests of any length then batch together, and the batched forward emits the whole
``[B, S, D]`` batch, cut per request after the forward by ``request_embeds``, the one
thing a model overrides. The forward has fixed shapes and no host syncs, so it is
captured at each of ``geometric_batch_sizes(max_batch_size)``: powers of two below
``max_batch_size``, then ``max_batch_size``.

Only the masked (padded-prompt) path is captured. Whether a prompt has padding is
``text_len`` in the step metadata, which the model must set at intake from the prompt
tensor's length (a row without it is a ValueError); a batch holding a prompt that fills
all S tokens runs the encoder's unmasked path eagerly.
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig, geometric_batch_sizes
from mstar.model.components.diffusion.qwen3.encoder import Qwen3EncoderConfig
from mstar.model.submodule_base import NodeInputs, NodeSubmodule

TEXT_INPUTS, TEXT_MASK, TEXT_EMBEDS = "text_inputs", "text_mask", "text_embeds"
# step metadata: the prompt's token count
TEXT_LEN = "text_len"


class Qwen3TextEncoderSubmodule(NodeSubmodule):
    """Qwen3 hidden-state taps over the right-padded prompt (see the module docstring)."""

    disable_torch_compile = True
    # the batched forward's whole output, cut per request after the forward
    PACKED_EMBEDS = "__text_embeds__"

    def __init__(
        self, encoder: nn.Module, text_config: Qwen3EncoderConfig, max_batch_size: int = 16,
        capture_walk: str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.text_config = text_config
        self._max_batch_size = max_batch_size
        self._capture_walk = capture_walk
        # Geometric, not every size: a one-shot encode pads a batch of 3 or 5-7
        # to the next graph for a fraction of one batch-4 encode, so it needs no
        # graph per size the way a denoise step's batch does.
        self._capture_batch_sizes = geometric_batch_sizes(max_batch_size)

    def request_embeds(self, hidden: torch.Tensor, text_len: int) -> torch.Tensor:
        """One request's ``text_embeds`` from its ``[1, S, D]`` row; all S positions
        (pad tokens included) unless the model cuts them."""
        del text_len
        return hidden

    def _pad_row(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``ids [L]`` -> right-padded ``(ids, mask)``, each ``[max_sequence_length]`` int64,
        built where ``ids`` lives (device ops only, no host round trip)."""
        width = self.text_config.max_sequence_length
        n = int(ids.shape[0])
        padded = torch.full((width,), self.text_config.pad_token_id, dtype=torch.long, device=ids.device)
        padded[:n].copy_(ids)
        mask = torch.zeros(width, dtype=torch.long, device=ids.device)
        mask[:n] = 1
        return padded, mask

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        ids = inputs[TEXT_INPUTS][0]
        padded, mask = self._pad_row(ids)
        return NodeInputs(
            tensor_inputs={TEXT_INPUTS: padded, TEXT_MASK: mask},
            input_seq_len=self.text_config.max_sequence_length,
            # the real token count, from the shape: where this row's output is cut
            resource_step_info=int(ids.shape[0]),
        )

    def can_batch(self, batch, model_inputs) -> bool:
        # one path for any batch size, as a captured B=1 replay already runs it
        return True

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size

    def cg_key_info(self, graph_walk: str, per_request_info, per_request_input_metadata=None, **kwargs):
        """True when every row is a padded prompt: the masked path, the only captured one."""
        del graph_walk, per_request_input_metadata, kwargs
        width = self.text_config.max_sequence_length
        lengths = [info.step_metadata.get(TEXT_LEN) for info in per_request_info.values()]
        if None in lengths:
            # without it no batch would ever replay, silently
            raise ValueError(f"{type(self).__name__} needs the prompt's token count in step metadata {TEXT_LEN!r}")
        return all(n < width for n in lengths)

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]) -> dict:
        width = self.text_config.max_sequence_length
        return {
            TEXT_INPUTS: torch.stack([inp.tensor_inputs[TEXT_INPUTS] for inp in inputs]),
            TEXT_MASK: torch.stack([inp.tensor_inputs[TEXT_MASK] for inp in inputs]),
            # any padded row puts the batch on the masked path, as the reference batches it
            "padded": any(inp.resource_step_info < width for inp in inputs),
        }

    def _encode(self, ids: torch.Tensor, mask: torch.Tensor, padded: bool | None = None) -> torch.Tensor:
        device = self.get_device()
        return self.encoder(ids.to(device), mask.to(device), padded=padded).to(self.encoder.dtype)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("the text encoder runs batched at any size (can_batch is always True)")

    def forward_batched(self, graph_walk, engine_inputs, text_inputs, text_mask, padded=None, **kwargs):
        return {self.PACKED_EMBEDS: self._encode(text_inputs, text_mask, padded)}

    def unpack_packed_outputs(self, static_output, request_ids, real_seq_lens, inputs, per_request_info):
        hidden = static_output.get(self.PACKED_EMBEDS)
        if hidden is None:
            return {}
        # cloned: a captured forward's output buffer is overwritten by the next replay
        return {
            rid: {TEXT_EMBEDS: [self.request_embeds(hidden[i : i + 1], inp.resource_step_info).clone()]}
            for i, (rid, inp) in enumerate(zip(request_ids, inputs, strict=True))
        }

    def get_accelerator_graph_configs(self, device: torch.device, tp_world_size: int = 1):
        if self._capture_walk is None:
            return []

        # a padding row: a one-token prompt, so every query row attends to something
        ids, mask = self._pad_row(torch.zeros(1, dtype=torch.long, device=device))
        single = NodeInputs(tensor_inputs={TEXT_INPUTS: ids, TEXT_MASK: mask},
                            input_seq_len=int(ids.shape[0]), resource_step_info=1)
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk=self._capture_walk,
            single_request_inputs=single,
            additional_key_info=True,
            compile=False,
            capture_batch_sizes=list(self._capture_batch_sizes),
            # a batch holding an unpadded prompt runs eager
            caps_eager_batch_size=False,
            input_seq_dims={TEXT_INPUTS: 0, TEXT_MASK: 0},
        )]
