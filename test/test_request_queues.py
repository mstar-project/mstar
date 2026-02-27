import sys
import time

from mminf.graph.request_queues import PerRequestStageQueues
sys.path.insert(0, ".")
from mminf.graph.base import GraphPointer, GraphStage, Loop, Parallel, Sequential
import numpy as np


if __name__ == "__main__":
    # show-o2-style graph with weird stuff added to stress-test

    loop = Loop(
        section=Parallel([
            Sequential([
                GraphStage(
                    name="LLM",
                    input_ids=["text_emb", "img_emb", "latents"],
                    outputs=[
                        GraphPointer(name="hidden_states", next_stage="flow"),
                        GraphPointer(name="some_random_external_output", next_stage="f")
                    ]
                ),
                Loop(
                    section=Sequential([
                        GraphStage(
                            name="flow",
                            input_ids=["hidden_states", "mystery", "mystery2"],
                            outputs=[
                                GraphPointer(name="partial_mystery2", next_stage="flow2"),
                                GraphPointer(name="partial_latents", next_stage="flow2")
                            ]
                        ),
                        GraphStage(
                            name="flow2",
                            input_ids=["partial_latents"],
                            outputs=[
                                GraphPointer(name="latents", next_stage="")
                            ]
                        ),
                    ]),
                    n_iters=2,
                    outputs=[
                        GraphPointer(name="latents", next_stage="LLM")
                    ]
                )
            ]),
            Sequential([
                GraphStage(
                    name="f",
                    input_ids=["mystery", "some_random_external_output"],
                    outputs=[
                        GraphPointer(name="xyz", next_stage="g")
                    ]
                ),
                GraphStage(
                    name="g",
                    input_ids=["xyz"],
                    outputs=[
                        GraphPointer(name="mystery", next_stage="f"),
                        GraphPointer(name="mystery", next_stage="flow")
                    ]
                )
            ])
        ]),
        n_iters=3,
        outputs=[
            GraphPointer(name="latents", next_stage="VAE_decoder"),
            GraphPointer(name="some_random_external_output", next_stage="STREAM_OUT")
        ]
    )

    loop = Parallel([
        Sequential([
            Loop(
                section=loop.section.sections[0].sections[0],
                n_iters=loop.n_iters,
                curr_iter=loop.curr_iter,
                external_inputs=loop.external_inputs,
                loop_back_signals=loop.loop_back_signals,
                outputs=loop.outputs
            ),
            Loop(
                section=loop.section.sections[0].sections[1],
                n_iters=loop.n_iters,
                curr_iter=loop.curr_iter,
                external_inputs=loop.external_inputs,
                loop_back_signals=loop.loop_back_signals,
                outputs=loop.outputs
            )
        ]),
        Loop(
            section=loop.section.sections[1],
            n_iters=loop.n_iters,
            curr_iter=loop.curr_iter,
            external_inputs=loop.external_inputs,
            loop_back_signals=loop.loop_back_signals,
            outputs=loop.outputs
        )
    ])


    network = Sequential([
        Parallel([
            GraphStage(
                name="text_emb",
                input_ids=["text"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="text_emb"),
                    GraphPointer(next_stage="f", name="mystery"),
                    GraphPointer(next_stage="flow", name="mystery")
                ]
            ),
            GraphStage(
                name="vit_encoder",
                input_ids=["image"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="img_emb"),
                    GraphPointer(next_stage="f", name="some_random_external_output"),
                ]
            )
        ]),
        Loop(
            section=Parallel([
                Sequential([
                    GraphStage(
                        name="LLM",
                        input_ids=["text_emb", "img_emb", "latents"],
                        outputs=[
                            GraphPointer(next_stage="flow", name="hidden_states"),
                            GraphPointer(next_stage="f", name="some_random_external_output")
                        ]
                    ),
                    Loop(
                        section=Sequential([
                            GraphStage(
                                name="flow",
                                input_ids=["hidden_states", "mystery", "mystery2"],
                                outputs=[
                                    GraphPointer(next_stage="flow2", name="partial_mystery2"),
                                    GraphPointer(next_stage="flow2", name="partial_latents")
                                ]
                            ),
                            GraphStage(
                                name="flow2",
                                input_ids=["partial_latents", "partial_mystery2"],
                                outputs=[
                                    GraphPointer(next_stage="", name="latents"),
                                    GraphPointer(next_stage="flow", name="mystery2")
                                ]
                            ),
                        ]),
                        n_iters=2,
                        outputs=[
                            GraphPointer(next_stage="LLM", name="latents")
                        ]
                    )
                ]),
                Sequential([
                    GraphStage(
                        name="f",
                        input_ids=["mystery", "some_random_external_output"],
                        outputs=[
                            GraphPointer(next_stage="g", name="xyz")
                        ]
                    ),
                    GraphStage(
                        name="g",
                        input_ids=["xyz"],
                        outputs=[
                            GraphPointer(next_stage="f", name="mystery"),
                            GraphPointer(next_stage="flow", name="mystery")
                        ]
                    )
                ])
            ]),
            n_iters=3,
            outputs=[
                GraphPointer(next_stage="VAE_decoder", name="latents"),
                GraphPointer(next_stage="STREAM_OUT", name="some_random_external_output")
            ]
        ),
        GraphStage(
            name="VAE_decoder",
            input_ids=["latents"],
            outputs=[
                GraphPointer(next_stage="STREAM_OUT", name="generated_image")
            ]
        )
    ])

    provided_inputs = [
        GraphPointer(name="text", next_stage="text_emb"),
        GraphPointer(name="image", next_stage="vit_encoder"),
        GraphPointer(name="latents", next_stage="LLM"),
        GraphPointer(name="mystery2", next_stage="flow")
    ]

    queues = PerRequestStageQueues(
        ready=[],
        waiting=network
    )

    tic = time.perf_counter()
    queues.process_new_inputs(provided_inputs)
    # loop until all stages are done and print out
    while len(queues.ready) > 0 or queues.waiting is not None:
        print("\n" + "="*60)
        print("Ready stages:", [stage.name for stage in queues.ready])
        if queues.waiting is not None:
            print("Waiting stages:", queues.waiting.get_stage_names())

        if len(queues.ready) == 0:
            # print(queues.waiting)
            raise Exception("No ready stages but still waiting stages, something's wrong")
        print()
        # pop a random ready stage and process it
        stage = queues.ready.pop(np.random.randint(0, len(queues.ready)))
        print(f"Processing stage {stage.name} with inputs {stage.input_ids}")
        new_inputs = stage.outputs
        print(f"New inputs: {[f'{ptr.name} -> {ptr.next_stage}' for ptr in new_inputs]}")
        external_outputs = queues.process_new_inputs(new_inputs)
        print(f"Outputs: {external_outputs}")
    toc = time.perf_counter()
    print(toc - tic)

    