import torch

from mminf.graph.base import GraphPointer, GraphStage, Loop, Sequential, TensorPointerInfo
from mminf.model.base import STREAM_OUT, CurrentForwardMetadata, Model


class DummyOmniModel(Model):
    """
    Qwen3-Omni-inspired dummy model for testing speech generation graphs.

    Phases (each is a separate forward pass):
      prefill/decode: ThinkerLLM only
      talker:         TalkerLLM -> MTP x16
      audio_gen:      AudioCodec (codec decoder)

    Full cycle: prefill -> talker -> audio_gen -> decode -> talker -> audio_gen -> ...
    """

    def get_phase_graphs(self):
        prefill = GraphStage(
            name="ThinkerLLM",
            input_ids=["input_ids"],
            outputs=[
                GraphPointer(
                    next_stage=STREAM_OUT,
                    name="thinker_hidden",
                    back_to_conductor=True,
                ),
                GraphPointer(
                    next_stage=STREAM_OUT,
                    name="thinker_token",
                    is_new_token=True,
                ),
            ],
        )

        decode = GraphStage(
            name="ThinkerLLM",
            input_ids=["input_ids"],
            outputs=[
                GraphPointer(
                    next_stage=STREAM_OUT,
                    name="thinker_hidden",
                    back_to_conductor=True,
                ),
                GraphPointer(
                    next_stage=STREAM_OUT,
                    name="thinker_token",
                    is_new_token=True,
                ),
            ],
        )

        talker = Sequential([
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
                        GraphPointer(
                            next_stage=STREAM_OUT,
                            name="mtp_token",
                            is_new_token=True,
                        ),
                    ],
                ),
                n_iters=16,
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        name="codec_hidden",
                        back_to_conductor=True,
                    ),
                ],
            ),
        ])

        audio_gen = GraphStage(
            name="AudioCodec",
            input_ids=["codec_hidden"],
            outputs=[
                GraphPointer(
                    next_stage=STREAM_OUT,
                    name="audio_output",
                    back_to_conductor=True,
                ),
            ],
        )

        return dict(
            prefill=prefill,
            decode=decode,
            talker=talker,
            audio_gen=audio_gen,
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
        persist_signals: dict[str, TensorPointerInfo],
        prev_forward_metadata: CurrentForwardMetadata = None,
    ) -> list[GraphPointer]:
        if metadata.phase in ("prefill", "decode"):
            ptr = GraphPointer(next_stage="ThinkerLLM", name="input_ids")
            ptr.tensor_info = persist_signals.get("input_ids", None)
            return [ptr]
        elif metadata.phase == "talker":
            ptr = GraphPointer(next_stage="TalkerLLM", name="thinker_hidden")
            ptr.tensor_info = persist_signals.get("thinker_hidden", None)
            return [ptr]
        elif metadata.phase == "audio_gen":
            ptr = GraphPointer(next_stage="AudioCodec", name="codec_hidden")
            ptr.tensor_info = persist_signals.get("codec_hidden", None)
            return [ptr]
        return []

    def update_for_next_forward(
        self, metadata: CurrentForwardMetadata,
        new_tokens: list[int],
    ) -> CurrentForwardMetadata:
        if metadata.phase == "prefill":
            metadata.is_prefill = False
            metadata.phase = "talker"
        elif metadata.phase == "decode":
            metadata.phase = "talker"
        elif metadata.phase == "talker":
            metadata.phase = "audio_gen"
        elif metadata.phase == "audio_gen":
            metadata.phase = "decode"
        return metadata

    def step(
        self, stage_name: str,
        phase: str,
        input_tensors: dict[str, torch.Tensor],
        state,
        **kwargs,
    ):
        return  # do nothing
