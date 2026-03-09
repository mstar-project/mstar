
from mminf.engine.ar_engine import KVCacheConfig
from mminf.graph.base import GraphPointer, GraphStage, Loop, Sequential, TensorPointerInfo
from mminf.model.base import STREAM_OUT, CurrentForwardMetadata, Model


class DummyOmniModel(Model):
    """
    Qwen3-Omni-inspired dummy model for testing speech generation graphs.

    Phases:
      prefill: ThinkerLLM -> TalkerLLM -> MTP x16 -> AudioCodec
      decode:  ThinkerLLM -> TalkerLLM -> MTP x16 -> AudioCodec

    Full cycle: prefill -> decode -> decode -> ...
    """

    def _make_full_graph(self):
        """Build the full sequential graph shared by both phases."""
        return Sequential([
            GraphStage(
                name="ThinkerLLM",
                input_ids=["input_ids"],
                outputs=[
                    GraphPointer(next_stage="TalkerLLM", name="thinker_hidden"),
                    GraphPointer(
                        next_stage=STREAM_OUT, name="thinker_token", is_new_token=True,
                        output_modality="text"
                    ),
                ],
            ),
            GraphStage(
                name="TalkerLLM",
                input_ids=["thinker_hidden"],
                outputs=[
                    GraphPointer(next_stage="MTP", name="codec_hidden"),
                    GraphPointer(next_stage=STREAM_OUT, name="talker_token", is_new_token=True),
                ],
            ),
            Loop(
                section=GraphStage(
                    name="MTP",
                    input_ids=["codec_hidden"],
                    outputs=[
                        GraphPointer(next_stage="MTP", name="codec_hidden"),
                        GraphPointer(next_stage=STREAM_OUT, name="mtp_token", is_new_token=True),
                    ],
                ),
                n_iters=16,
                outputs=[
                    GraphPointer(next_stage="AudioCodec", name="codec_hidden"),
                ],
            ),
            GraphStage(
                name="AudioCodec",
                input_ids=["codec_hidden"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        name="audio_output",
                        output_modality="audio",
                        back_to_conductor=True,
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

    def get_phase_graphs(self):
        return dict(
            prefill=self._make_full_graph(),
            decode=self._make_full_graph(),
        )

    def get_initial_forward_metadata(
        self, input_modalities, output_modalities,
    ):
        return CurrentForwardMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            phase="prefill",
            is_prefill=True,
        )

    def get_forward_pass_inputs(
        self, metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        prev_forward_metadata: CurrentForwardMetadata = None,
    ) -> list[GraphPointer]:
        ptr = GraphPointer(next_stage="ThinkerLLM", name="input_ids")
        ptr.tensor_info = persist_signals.get("input_ids", [])
        return [ptr]

    def update_for_next_forward(
        self, metadata: CurrentForwardMetadata,
        new_tokens: dict[str, list[int]],
    ) -> CurrentForwardMetadata:
        if metadata.phase == "prefill":
            metadata.is_prefill = False
            metadata.phase = "decode"
        return metadata
