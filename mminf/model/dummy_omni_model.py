
from mminf.engine.kv_store import KVCacheConfig
from mminf.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mminf.graph.special_destinations import EMIT_TO_CLIENT
from mminf.model.base import CurrentForwardMetadata, Model


class DummyOmniModel(Model):
    """
    Qwen3-Omni-inspired dummy model for testing speech generation graphs.

    Graph walks:
      prefill: ThinkerLLM -> TalkerLLM -> MTP x16 -> AudioCodec
      decode:  ThinkerLLM -> TalkerLLM -> MTP x16 -> AudioCodec

    Full cycle: prefill -> decode -> decode -> ...
    """

    def _make_full_graph(self):
        """Build the full sequential graph shared by both graph walks."""
        return Sequential([
            GraphNode(
                name="ThinkerLLM",
                input_ids=["input_ids"],
                outputs=[
                    GraphEdge(next_node="TalkerLLM", name="thinker_hidden"),
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT, name="thinker_token", is_new_token=True,
                        output_modality="text"
                    ),
                ],
            ),
            GraphNode(
                name="TalkerLLM",
                input_ids=["thinker_hidden"],
                outputs=[
                    GraphEdge(next_node="MTP", name="codec_hidden"),
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="talker_token", is_new_token=True),
                ],
            ),
            Loop(
                section=GraphNode(
                    name="MTP",
                    input_ids=["codec_hidden"],
                    outputs=[
                        GraphEdge(next_node="MTP", name="codec_hidden"),
                        GraphEdge(next_node=EMIT_TO_CLIENT, name="mtp_token", is_new_token=True),
                    ],
                ),
                n_iters=16,
                outputs=[
                    GraphEdge(next_node="AudioCodec", name="codec_hidden"),
                ],
            ),
            GraphNode(
                name="AudioCodec",
                input_ids=["codec_hidden"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="audio_output",
                        output_modality="audio",
                        persist=True,
                    ),
                ],
            ),
        ])

    def get_kv_cache_config(self) -> KVCacheConfig:
        return KVCacheConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            max_seq_len=1,
        )

    def get_graph_walk_graphs(self):
        return dict(
            prefill=self._make_full_graph(),
            decode=self._make_full_graph(),
        )

    def get_initial_forward_pass_args(
        self, input_modalities, output_modalities,
    ):
        return CurrentForwardMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill",
            is_prefill=True,
        )

    def get_forward_pass_args(
        self, metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        prev_forward_metadata: CurrentForwardMetadata = None,
    ) -> list[GraphEdge]:
        graph_edge = GraphEdge(next_node="ThinkerLLM", name="input_ids")
        graph_edge.tensor_info = persist_signals.get("input_ids", [])
        return [graph_edge]

    def update_for_next_forward(
        self, metadata: CurrentForwardMetadata,
        new_tokens: dict[str, list[int]],
    ) -> CurrentForwardMetadata:
        if metadata.graph_walk == "prefill":
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        return metadata
