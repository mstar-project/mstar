"""Sampling-parity test model.

A model-free harness for the CUDA-graph sampler: one KV_CACHE node, two graph
walks that run back to back with no data dependence between them.

  1. ``graphed`` — a ``Loop`` captured as a whole-walk CUDA graph. Each iter runs
     an MLP → main sample (penalty) → MLP → aux-sample loop through the real
     ``CudaGraphableSampler`` / ``MultiSamplerBuffers`` path.
  2. ``eager`` — the same ``Loop`` but eager; the node calls ``sample_tokens``
     directly at the same (seed, offset), with a sync.

Both walks feed the SAME fixed per-request embed back through the loop, so the
logits are identical every iteration and the only moving part is the RNG offset.
A token stream that differs between the two walks is a sampler bug.

model_kwargs:
  - ``iters``: loop length per walk (default 20).
  - ``seed``:  fixed base seed (so runs are reproducible; the conductor derives
    the aux seed from it via ``MultiSamplingConfig.set_seed``).

NOTE (needs GPU iteration): the per-iteration token emission wiring below and
the two-walk dispatch are the parts to verify on hardware — the sampler contract
lives in ``submodules.py``.
"""

from dataclasses import dataclass, field

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.dummy.sampling_test.submodules import (
    EAGER_WALK,
    GRAPHED_WALK,
    NODE_NAME,
    SamplingTestSubmodule,
)
from mstar.utils.sampling import SamplingConfig

DEFAULT_ITERS = 20
MAX_ITERS = 1000


@dataclass
class SamplingTestConfig:
    hidden_size: int = 512
    main_vocab_size: int = 4096
    aux_vocab_size: int = 2048
    num_aux_groups: int = 15
    num_hidden_layers: int = 1
    num_key_value_heads: int = 1
    num_attention_heads: int = 1
    head_dim: int = 64
    max_position_embeddings: int = 2048
    decode_capture_batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4]
    )
    # Main sampler knobs (mirror the omni Talker: penalty + no nucleus).
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    # Aux (code-predictor-like) knobs.
    aux_temperature: float = 1.0
    aux_top_k: int = 50
    aux_top_p: float = 0.8


def _resolve_iters(model_kwargs: dict | None) -> int:
    n = int((model_kwargs or {}).get("iters", DEFAULT_ITERS))
    if not 1 <= n <= MAX_ITERS:
        raise ValueError(f"iters={n} out of range; must be in [1, {MAX_ITERS}]")
    return n


class SamplingTest(Model):
    def __init__(
        self, model_path_hf: str = "", cache_dir: str | None = None, **kwargs
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.config = SamplingTestConfig()

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        # A fixed (per-request) random embed fed back through the loop each iter.
        # Shape [seq=1, hidden] to match the AR decode input contract.
        gen = torch.Generator().manual_seed(0)
        return {"x": [torch.randn(
            1, self.config.hidden_size, dtype=torch.float32, generator=gen
        )]}

    def postprocess(
        self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None,
    ) -> bytes:
        return output.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

    # ---- engine wiring -------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        c = self.config
        return [KVCacheConfig(
            num_layers=c.num_hidden_layers,
            num_kv_heads=c.num_key_value_heads,
            head_dim=c.head_dim,
            max_seq_len=c.max_position_embeddings,
            num_qo_heads=c.num_attention_heads,
            nodes=[NODE_NAME],
        )]

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {NODE_NAME: EngineType.KV_CACHE}

    def get_sampling_config(
        self, node_name: str, model_kwargs: dict | None = None,
    ) -> SamplingConfig | None:
        c = self.config
        return SamplingConfig(
            vocab_size=c.main_vocab_size,
            temperature=model_kwargs.get("temperature", c.temperature),
            top_k=c.top_k, top_p=c.top_p,
            repetition_penalty=c.repetition_penalty,
        )

    def get_aux_sampling_configs(
        self, node_name: str, model_kwargs: dict | None = None,
    ) -> dict[str, SamplingConfig]:
        c = self.config
        # No vocab_size: the aux path applies no penalty (matches code predictor).
        return {"aux": SamplingConfig(
            temperature=model_kwargs.get("aux_temperature", c.aux_temperature),
            top_k=c.aux_top_k, top_p=c.aux_top_p,
        )}

    def _loop(self, name: str, enable_async: bool=True) -> Loop:
        return Loop(
            name=name,
            section=GraphNode(
                name=NODE_NAME,
                input_names=["x"],
                enable_async_scheduling=enable_async,
                outputs=[
                    GraphEdge(next_node=NODE_NAME, name="x"),  # feedback
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT, name=f"{name}_tokens",
                        output_modality="tensor",
                    ),
                ],
            ),
            max_iters=MAX_ITERS,
            outputs=[],
        )

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        return {GRAPHED_WALK: self._loop("graphed_loop", enable_async=False),
                EAGER_WALK: self._loop("eager_loop", enable_async=False)}

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        iters = _resolve_iters(model_kwargs)
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=GRAPHED_WALK,
            is_prefill=False,
            kwargs={"iters": iters},
        )
        edge = GraphEdge(next_node=NODE_NAME, name="x")
        edge.tensor_info = input_signals["x"]
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=[edge],
            unpersist_tensors=[],  # keep "x" persisted to reuse for the eager walk
            step_metadata={"iters": iters},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        # After the graphed loop, run the eager loop on the SAME original input;
        # after the eager loop, done.
        if partition_metadata.graph_walk == EAGER_WALK:
            return ForwardPassArgs(
                full_metadata=partition_metadata, inputs=[],
                unpersist_tensors=persist_signals["x"], request_done=True,
            )
        partition_metadata.graph_walk = EAGER_WALK
        edge = GraphEdge(next_node=NODE_NAME, name="x")
        edge.tensor_info = persist_signals["x"]
        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=[edge],
            unpersist_tensors=[],
            step_metadata={"iters": partition_metadata.kwargs["iters"]},
        )

    def get_submodule(
        self, node_name: str, device="cpu", **kwargs
    ) -> torch.nn.Module | None:
        assert node_name == NODE_NAME
        return SamplingTestSubmodule(self.config)
