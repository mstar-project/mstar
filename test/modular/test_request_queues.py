import sys
import time

from mminf.graph.request_queues import PerRequestStageQueues

sys.path.insert(0, ".")
import numpy as np

from mminf.graph.base import GraphEdge, GraphStage, Loop, Parallel, Sequential

if __name__ == "__main__":
    # show-o2-style graph with weird stuff added to stress-test

    loop = Loop(
        section=Parallel([
            Sequential([
                GraphStage(
                    name="LLM",
                    input_ids=["text_emb", "img_emb", "latents"],
                    outputs=[
                        GraphEdge(name="hidden_states", next_stage="flow"),
                        GraphEdge(name="some_random_external_output", next_stage="f")
                    ]
                ),
                Loop(
                    section=Sequential([
                        GraphStage(
                            name="flow",
                            input_ids=["hidden_states", "mystery", "mystery2"],
                            outputs=[
                                GraphEdge(name="partial_mystery2", next_stage="flow2"),
                                GraphEdge(name="partial_latents", next_stage="flow2")
                            ]
                        ),
                        GraphStage(
                            name="flow2",
                            input_ids=["partial_latents"],
                            outputs=[
                                GraphEdge(name="latents", next_stage="")
                            ]
                        ),
                    ]),
                    n_iters=2,
                    outputs=[
                        GraphEdge(name="latents", next_stage="LLM")
                    ]
                )
            ]),
            Sequential([
                GraphStage(
                    name="f",
                    input_ids=["mystery", "some_random_external_output"],
                    outputs=[
                        GraphEdge(name="xyz", next_stage="g")
                    ]
                ),
                GraphStage(
                    name="g",
                    input_ids=["xyz"],
                    outputs=[
                        GraphEdge(name="mystery", next_stage="f"),
                        GraphEdge(name="mystery", next_stage="flow")
                    ]
                )
            ])
        ]),
        n_iters=3,
        outputs=[
            GraphEdge(name="latents", next_stage="VAE_decoder"),
            GraphEdge(name="some_random_external_output", next_stage="STREAM_OUT")
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
                    GraphEdge(next_stage="LLM", name="text_emb"),
                    GraphEdge(next_stage="f", name="mystery"),
                    GraphEdge(next_stage="flow", name="mystery")
                ]
            ),
            GraphStage(
                name="vit_encoder",
                input_ids=["image"],
                outputs=[
                    GraphEdge(next_stage="LLM", name="img_emb"),
                    GraphEdge(next_stage="f", name="some_random_external_output"),
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
                            GraphEdge(next_stage="flow", name="hidden_states"),
                            GraphEdge(next_stage="f", name="some_random_external_output")
                        ]
                    ),
                    Loop(
                        section=Sequential([
                            GraphStage(
                                name="flow",
                                input_ids=["hidden_states", "mystery", "mystery2"],
                                outputs=[
                                    GraphEdge(next_stage="flow2", name="partial_mystery2"),
                                    GraphEdge(next_stage="flow2", name="partial_latents")
                                ]
                            ),
                            GraphStage(
                                name="flow2",
                                input_ids=["partial_latents", "partial_mystery2"],
                                outputs=[
                                    GraphEdge(next_stage="", name="latents"),
                                    GraphEdge(next_stage="flow", name="mystery2")
                                ]
                            ),
                        ]),
                        n_iters=2,
                        outputs=[
                            GraphEdge(next_stage="LLM", name="latents")
                        ]
                    )
                ]),
                Sequential([
                    GraphStage(
                        name="f",
                        input_ids=["mystery", "some_random_external_output"],
                        outputs=[
                            GraphEdge(next_stage="g", name="xyz")
                        ]
                    ),
                    GraphStage(
                        name="g",
                        input_ids=["xyz"],
                        outputs=[
                            GraphEdge(next_stage="f", name="mystery"),
                            GraphEdge(next_stage="flow", name="mystery")
                        ]
                    )
                ])
            ]),
            n_iters=3,
            outputs=[
                GraphEdge(next_stage="VAE_decoder", name="latents"),
                GraphEdge(next_stage="STREAM_OUT", name="some_random_external_output")
            ]
        ),
        GraphStage(
            name="VAE_decoder",
            input_ids=["latents"],
            outputs=[
                GraphEdge(next_stage="STREAM_OUT", name="generated_image")
            ]
        )
    ])

    provided_inputs = [
        GraphEdge(name="text", next_stage="text_emb"),
        GraphEdge(name="image", next_stage="vit_encoder"),
        GraphEdge(name="latents", next_stage="LLM"),
        GraphEdge(name="mystery2", next_stage="flow")
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
        print(f"New inputs: {[f'{edge.name} -> {edge.next_stage}' for edge in new_inputs]}")
        external_outputs = queues.process_new_inputs(new_inputs)
        print(f"Outputs: {external_outputs}")
    toc = time.perf_counter()
    print(toc - tic)
