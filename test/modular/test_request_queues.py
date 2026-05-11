import pytest

pytest.skip(
    "Legacy test of the deleted PerRequestNodeQueues API. Will be "
    "rewritten against WorkerGraphIO in Phase D of the graph refactor.",
    allow_module_level=True,
)

import sys  # noqa: E402
import time  # noqa: E402

from mminf.graph.request_queues import PerRequestNodeQueues  # noqa: E402

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from mminf.graph.base import GraphEdge, GraphNode, Loop, Parallel, Sequential  # noqa: E402

if __name__ == "__main__":
    # show-o2-style graph with weird stuff added to stress-test

    loop = Loop(
        section=Parallel([
            Sequential([
                GraphNode(
                    name="LLM",
                    input_ids=["text_emb", "img_emb", "latents"],
                    outputs=[
                        GraphEdge(name="hidden_states", next_node="flow"),
                        GraphEdge(name="some_random_external_output", next_node="f")
                    ]
                ),
                Loop(
                    section=Sequential([
                        GraphNode(
                            name="flow",
                            input_ids=["hidden_states", "mystery", "mystery2"],
                            outputs=[
                                GraphEdge(name="partial_mystery2", next_node="flow2"),
                                GraphEdge(name="partial_latents", next_node="flow2")
                            ]
                        ),
                        GraphNode(
                            name="flow2",
                            input_ids=["partial_latents"],
                            outputs=[
                                GraphEdge(name="latents", next_node="")
                            ]
                        ),
                    ]),
                    max_iters=2,
                    outputs=[
                        GraphEdge(name="latents", next_node="LLM")
                    ]
                )
            ]),
            Sequential([
                GraphNode(
                    name="f",
                    input_ids=["mystery", "some_random_external_output"],
                    outputs=[
                        GraphEdge(name="xyz", next_node="g")
                    ]
                ),
                GraphNode(
                    name="g",
                    input_ids=["xyz"],
                    outputs=[
                        GraphEdge(name="mystery", next_node="f"),
                        GraphEdge(name="mystery", next_node="flow")
                    ]
                )
            ])
        ]),
        max_iters=3,
        outputs=[
            GraphEdge(name="latents", next_node="VAE_decoder"),
            GraphEdge(name="some_random_external_output", next_node="EMIT_TO_CLIENT")
        ]
    )

    loop = Parallel([
        Sequential([
            Loop(
                section=loop.section.sections[0].sections[0],
                max_iters=loop.max_iters,
                curr_iter=loop.curr_iter,
                _external_inputs=loop._external_inputs,
                _loop_back_signals=loop._loop_back_signals,
                outputs=loop.outputs
            ),
            Loop(
                section=loop.section.sections[0].sections[1],
                max_iters=loop.max_iters,
                curr_iter=loop.curr_iter,
                _external_inputs=loop._external_inputs,
                _loop_back_signals=loop._loop_back_signals,
                outputs=loop.outputs
            )
        ]),
        Loop(
            section=loop.section.sections[1],
            max_iters=loop.max_iters,
            curr_iter=loop.curr_iter,
            _external_inputs=loop._external_inputs,
            _loop_back_signals=loop._loop_back_signals,
            outputs=loop.outputs
        )
    ])

    network = Sequential([
        Parallel([
            GraphNode(
                name="text_emb",
                input_ids=["text"],
                outputs=[
                    GraphEdge(next_node="LLM", name="text_emb"),
                    GraphEdge(next_node="f", name="mystery"),
                    GraphEdge(next_node="flow", name="mystery")
                ]
            ),
            GraphNode(
                name="vit_encoder",
                input_ids=["image"],
                outputs=[
                    GraphEdge(next_node="LLM", name="img_emb"),
                    GraphEdge(next_node="f", name="some_random_external_output"),
                ]
            )
        ]),
        Loop(
            section=Parallel([
                Sequential([
                    GraphNode(
                        name="LLM",
                        input_ids=["text_emb", "img_emb", "latents"],
                        outputs=[
                            GraphEdge(next_node="flow", name="hidden_states"),
                            GraphEdge(next_node="f", name="some_random_external_output")
                        ]
                    ),
                    Loop(
                        section=Sequential([
                            GraphNode(
                                name="flow",
                                input_ids=["hidden_states", "mystery", "mystery2"],
                                outputs=[
                                    GraphEdge(next_node="flow2", name="partial_mystery2"),
                                    GraphEdge(next_node="flow2", name="partial_latents")
                                ]
                            ),
                            GraphNode(
                                name="flow2",
                                input_ids=["partial_latents", "partial_mystery2"],
                                outputs=[
                                    GraphEdge(next_node="", name="latents"),
                                    GraphEdge(next_node="flow", name="mystery2")
                                ]
                            ),
                        ]),
                        max_iters=2,
                        outputs=[
                            GraphEdge(next_node="LLM", name="latents")
                        ]
                    )
                ]),
                Sequential([
                    GraphNode(
                        name="f",
                        input_ids=["mystery", "some_random_external_output"],
                        outputs=[
                            GraphEdge(next_node="g", name="xyz")
                        ]
                    ),
                    GraphNode(
                        name="g",
                        input_ids=["xyz"],
                        outputs=[
                            GraphEdge(next_node="f", name="mystery"),
                            GraphEdge(next_node="flow", name="mystery")
                        ]
                    )
                ])
            ]),
            max_iters=3,
            outputs=[
                GraphEdge(next_node="VAE_decoder", name="latents"),
                GraphEdge(next_node="EMIT_TO_CLIENT", name="some_random_external_output")
            ]
        ),
        GraphNode(
            name="VAE_decoder",
            input_ids=["latents"],
            outputs=[
                GraphEdge(next_node="EMIT_TO_CLIENT", name="generated_image")
            ]
        )
    ])

    provided_inputs = [
        GraphEdge(name="text", next_node="text_emb"),
        GraphEdge(name="image", next_node="vit_encoder"),
        GraphEdge(name="latents", next_node="LLM"),
        GraphEdge(name="mystery2", next_node="flow")
    ]

    queues = PerRequestNodeQueues(
        ready=[],
        waiting=network
    )

    tic = time.perf_counter()
    queues.process_new_inputs(provided_inputs)
    # loop until all nodes are done and print out
    while len(queues.ready) > 0 or queues.waiting is not None:
        print("\n" + "="*60)
        print("Ready nodes:", [node.name for node in queues.ready])
        if queues.waiting is not None:
            print("Waiting nodes:", queues.waiting.get_node_names())

        if len(queues.ready) == 0:
            # print(queues.waiting)
            raise Exception("No ready nodes but still waiting nodes, something's wrong")
        print()
        # pop a random ready node and process it
        node = queues.ready.pop(np.random.randint(0, len(queues.ready)))
        print(f"Processing node {node.name} with inputs {node.input_ids}")
        new_inputs = node.outputs
        print(f"New inputs: {[f'{edge.name} -> {edge.next_node}' for edge in new_inputs]}")
        external_outputs = queues.process_new_inputs(new_inputs)
        print(f"Outputs: {external_outputs}")
    toc = time.perf_counter()
    print(toc - tic)
