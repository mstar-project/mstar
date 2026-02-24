from copy import deepcopy

import numpy as np

from mminf.graph.base import GraphPointer, GraphStage, Loop, Parallel, Sequential, SignalToDestsAndFlags
from mminf.model.base import STREAM_OUT, CurrentForwardMetadata, ForwardPassInputs, Model, TensorData


class DummyModel(Model):
    """
    Show-o2-inspired model that does nothing, for testing and example purposes.
    """
    def _get_text_emb(self):
        return GraphStage(
            name="text_emb",
            input_ids=["text"],
            outputs={
                "text_emb": "LLM"
            }
        )
    
    def _get_img_emb(self):
        return GraphStage(
            name="image_emb",
            input_ids=["images"],
            outputs={
                "img_emb": "LLM"
            }
        )

    def get_phase_graphs(self):
        prefill = Sequential([
            Parallel([self._get_text_emb(), self._get_img_emb()]),
            GraphStage(
                name="LLM",
                input_ids=["text_emb", "img_emb"],
                outputs={
                    "new_token": GraphPointer(STREAM_OUT, back_to_conductor=True)
                }
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
                        outputs={
                            "hidden_states": GraphPointer("flow")
                        }
                    ),
                    GraphStage(
                        "flow",
                        input_ids=["hidden_states"],
                        outputs={
                            "latents": GraphPointer("LLM")
                        }
                    )
                ]),
                n_iters=10,
                outputs={
                    "latents": GraphPointer("VAE_dec")
                }
            ),
            GraphStage(
                name="VAE_dec",
                input_ids=["latents"],
                outputs={
                    "image_output": GraphPointer(STREAM_OUT)
                }
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
        self, input_tensors: dict[str, TensorData],
        metadata: CurrentForwardMetadata,
    ) -> ForwardPassInputs:
        pointers = {
            "images": [GraphPointer("image_emb")],
            "text": [GraphPointer("text_emb")]
        }
        if "images" not in input_tensors: # add null image for routing purposes
            input_tensors["images"] = TensorData(tensor=None, token_ranges=[])
        if "text" not in input_tensors: # add null text for routing purposes
            input_tensors["text"] = TensorData(tensor=None, token_ranges=[])
        
        if metadata.phase == "image_gen":
            # maybe this should be a random tensor in real life, or the latents
            # will be initialized on the worker level...
            input_tensors["latents"] = TensorData( tensor=None, token_ranges=[])
            pointers["latents"] = [GraphPointer("LLM")]
        
        return ForwardPassInputs(
            tensors=input_tensors,
            pointers=pointers
        )
    
    def update_for_next_forward(
        self, metadata: CurrentForwardMetadata,
        input_tensors: dict[str, TensorData],
        new_outputs: dict[str, TensorData]
    ):
        # dummy model doesn't actually do anything, so this function will just
        # randomly select a phase
        metadata.phase = str(np.random.choice(["decode", "image_gen"]))
        if metadata.phase == "decode":
            metadata.output_modalities = ["text"]
        else:
            metadata.output_modalities = ["image"]
        return metadata, input_tensors

    def step(
        self, stage_name: str,
        phase: str,
        input_tensors: dict[str, TensorData],
        state, # TODO: figure out state
        **kwargs
    ):
        return # do nothing
