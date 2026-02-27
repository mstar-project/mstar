from copy import deepcopy

import numpy as np
import torch

from mminf.graph.base import GraphPointer, GraphStage, Loop, Parallel, Sequential, TensorPointerInfo
from mminf.model.base import STREAM_OUT, CurrentForwardMetadata, Model


class DummyModel(Model):
    """
    Show-o2-inspired model that does nothing, for testing and example purposes.
    """
    def _get_text_emb(self):
        return Sequential([
            GraphStage(
                name="text_emb",
                input_ids=["text"],
                outputs=[
                    GraphPointer(next_stage="concat_text", name="new_text_emb")
                ]
            ),
            GraphStage(
                name="concat_text",
                input_ids=["new_text_emb", "existing_text_emb"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="text_emb", back_to_conductor=True)
                ]
            )
        ])
    
    def _get_img_emb(self):
        return Sequential([
            GraphStage(
                name="image_emb",
                input_ids=["images"],
                outputs=[
                    GraphPointer(next_stage="concat_img", name="new_image_emb")
                ]
            ),
            GraphStage(
                name="concat_img",
                input_ids=["new_image_emb", "existing_image_emb"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="img_emb", back_to_conductor=True)
                ]
             )
        ])

    def get_phase_graphs(self):
        prefill = Sequential([
            Parallel([self._get_text_emb(), self._get_img_emb()]),
            GraphStage(
                name="LLM",
                input_ids=["text_emb", "img_emb"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        name="new_token",
                        is_new_token=True
                    )
                ]
            )
        ])
        decode = deepcopy(prefill)
        image_gen = Sequential([
            Parallel([self._get_text_emb(), self._get_img_emb()]),
            Loop(
                section=Sequential([
                    GraphStage(
                        name="LLM",
                        input_ids=["text_emb", "img_emb", "latents"],
                        outputs=[
                            GraphPointer(next_stage="flow", name="hidden_states")
                        ]
                    ),
                    GraphStage(
                        "flow",
                        input_ids=["hidden_states"],
                        outputs=[
                            GraphPointer(next_stage="LLM", name="latents")
                        ]
                    )
                ]),
                n_iters=10,
                outputs=[
                    GraphPointer(next_stage="VAE_dec", name="latents")
                ]
            ),
            GraphStage(
                name="VAE_dec",
                input_ids=["latents"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        name="image_output",
                        back_to_conductor=True
                    )
                ]
            )
        ])

        return dict(
            prefill=prefill,
            decode=decode,
            image_gen=image_gen
        )
    
    def get_initial_forward_metadata(
        self, input_modalities, output_modalities
    ):
        return CurrentForwardMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            phase="prefill",
            is_prefill=True
        )
    
    def get_forward_pass_inputs(
        self, metadata: CurrentForwardMetadata,
        persist_signals: dict[str, TensorPointerInfo],
        prev_forward_metadata: CurrentForwardMetadata=None,
    ) -> list[GraphPointer]:
        text_inp = GraphPointer(
            next_stage="text_emb",
            name="text",
        )
        img_inp = GraphPointer(
            next_stage="image_emb",
            name="images",
        )
        existing_text = GraphPointer(
            next_stage="concat_text",
            name="existing_text_emb",
        )
        existing_img = GraphPointer(
            next_stage="concat_img",
            name="existing_image_emb",
        )

        pointers = [
            text_inp, img_inp, existing_text, existing_img
        ]

        if metadata.is_prefill: # first forward
            text_inp.tensor_info = persist_signals.get("text", None)
            img_inp.tensor_info = persist_signals.get("images", None)
        else:
            existing_text.tensor_info = persist_signals.get("text_emb", None)
            existing_img.tensor_info = persist_signals.get("img_emb", None)
            if prev_forward_metadata.phase == "image_gen":
                img_inp.tensor_info = persist_signals.get("image_output", None)
                text_inp.tensor_info = persist_signals.get("new_token", None)
        
            if metadata.phase == "image_gen":
                pointers.append(
                    GraphPointer(
                        next_stage="LLM",
                        name="latents",
                        tensor_info=persist_signals.get("latents", None)
                    )
                )
        return pointers
        
    def update_for_next_forward(
        self, metadata: CurrentForwardMetadata,
        new_tokens: list[int],
    ) -> CurrentForwardMetadata:
        # dummy model doesn't actually do anything, so this function will just
        # randomly select a phase
        metadata.phase = str(np.random.choice(["decode", "image_gen"]))
        if metadata.phase == "decode":
            metadata.output_modalities = ["text"]
        else:
            metadata.output_modalities = ["image"]
        return metadata

    def step(
        self, stage_name: str,
        phase: str,
        input_tensors: dict[str, torch.Tensor],
        state, # TODO: figure out state
        **kwargs
    ):
        return # do nothing
