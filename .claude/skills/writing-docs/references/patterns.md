# Page patterns and worked edits

These are starting points, not a template to stamp onto every page. The examples below are original editorial adaptations. They illustrate structure; they are not evidence that an inference command was executed.

## A first successful request

The reader should be able to follow one path without choosing between unfamiliar models, deployment layouts, and APIs at every step.

1. Name the result and the supported setup. Put hardware and checkpoint-access requirements here, with links to installation details.
2. Start one server with a concrete model/config. Explain where to run the command and how to recognize readiness.
3. In a second terminal, send one request using that server's model, address, and interface.
4. Inspect the response field or file that establishes success. Explain variable output without fabricating a transcript.
5. Explain how to stop a process the tutorial started, where relevant. Offer next tasks with descriptive links.

Consider a quickstart that combines chat, image generation, and speech in one Python block, while its comments require different servers. A better structure completes one supported server/client pair, then gives image and speech their own sections or links with their prerequisites. Keep the useful alternative examples; remove the hidden state changes between them.

This combines the [SGLang quickstart's prerequisites and checkpoints](https://docs.sglang.io/docs/get-started/quickstart) with the [TGI quick tour's concrete launch and response](https://huggingface.co/docs/text-generation-inference/en/quicktour). Do not borrow their hardware requirements or model choices.

## A task with alternatives

Start with the decision the reader must make. If two paths solve the same task with equivalent prerequisites, tabs can keep the chosen path readable. If they require different hardware, permissions, models, or mental models, use separate linked sections or pages. A novice should not have to combine snippets across tabs to finish.

For a configuration task, show the smallest complete change, where it belongs, what it changes, and a check that distinguishes success from a server that merely starts. Keep the full list of settings in reference. Explain precedence where competing values are possible.

The [Fly.io deployment guide](https://docs.fly.io/launch/deploy) makes build-selection order explicit. The transferable lesson is to document precedence, not to assume M* has the same precedence.

## A reference entry

Use a table for short, comparable fields. Use a named subsection when an entry needs a long explanation or example.

| Field | What a reader needs |
| --- | --- |
| Name | Exact flag, YAML key, environment variable, endpoint, or symbol |
| Value | Type, unit, allowed values, and whether omission differs from an explicit value |
| Default | The verified value or how it is derived; distinguish unset, automatic, zero, and disabled |
| Scope | Which process, frontend, node, model, or deployment uses it |
| Effect | Observable behavior, limits, and important interactions |
| Override | Precedence when CLI, config, or environment overlap |
| Activation | Whether a change takes effect per request, at reload, or after a restart |
| Example | One valid use where syntax or interaction is non-obvious |

Omit fields that do not apply rather than filling cells with boilerplate. An environment-variable reference in M* also names the function reading the value, as AGENTS.md requires. Generated help is useful evidence but still needs an editorial check.

The [TGI launcher reference](https://huggingface.co/docs/text-generation-inference/en/reference/launcher) gives token-budget examples and explains consequences. Borrow the practice of explaining coupled limits; verify M*'s own units and arithmetic.

## An explanation that teaches a mechanism

Give the reader a small scenario, then follow what changes. Name the owner of each action and what crosses each boundary. Introduce abstractions when they explain the scenario, then state the general rule and its limits.

For an M* architecture page, a request trace can introduce the frontend, conductor, worker, and engine before a component inventory. A graph sketch should distinguish graph structure from hardware placement; a concurrency diagram should distinguish threads and in-flight stages. A general box-and-arrow picture cannot establish ordering guarantees.

Keep the details necessary to predict behavior. [SQLite's atomic-commit explanation](https://www.sqlite.org/atomiccommit.html) establishes assumptions before walking through state changes and failure recovery. Its length serves the reasoning. An M* lifecycle explanation should retain the same kind of causal depth, with its own code and tests as evidence.

## Troubleshooting and tuning

Use this sequence when it fits the problem:

**Symptom → check → interpretation → conditional action → verification.**

For diagnosis, a check should distinguish plausible causes or collect evidence someone can use. State what it does not establish. If the cause is uncertain, say so. Where useful, show a minimal failing example and its correction, clearly labeled as such. Document cleanup for diagnostic flags that change behavior or add overhead.

For tuning, first state the workload and metric. Tie a setting to the measured bottleneck, explain the mechanism and cost, then remeasure both the target and the likely regression. In M*, use the harnesses and evidence rules in AGENTS.md; a competitor's tuning heuristic is not an M* recommendation.

The [vLLM troubleshooting guide](https://docs.vllm.ai/en/stable/usage/troubleshooting/) separates download and load failures. [SGLang's tuning guide](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning) connects log fields to adjustments and costs. [Cloudflare's errors guide](https://developers.cloudflare.com/workers/observability/errors/) uses small broken examples. These are different tools for different reader questions, not a requirement to include all three on every page.

## Sentence-level edits

The weak examples here are invented. The stronger examples illustrate wording and must still be checked against the implementation when used in a guide.

| Weak draft | Stronger draft | Why |
| --- | --- | --- |
| M* seamlessly orchestrates powerful heterogeneous workflows. | The conductor selects workers and advances each request through its graph walks. | Names the actor and work; makes a checkable claim. |
| This section provides a comprehensive overview of configuration options. | Use `--config` to select a deployment YAML file. | Gives the useful action instead of announcing content. |
| Simply send a request to get started. | In a second terminal, send a request to the server you started above. | Supplies missing execution context. |

A claim such as “Increase this setting for optimal performance” needs evidence before it needs better prose. Identify the measured bottleneck, the cost of the setting, and how to compare the result. Without those facts, remove or qualify the recommendation; do not invent a stronger-sounding replacement.

When editing a real paragraph, preserve its information. For example, “The same model runs on one GPU or several by changing the YAML configuration” is clearer only if the model supports both layouts. Smooth wording does not justify broadening support.

## Mintlify components

| Format | Use it for | Avoid |
| --- | --- | --- |
| Cards | A small set of meaningful destinations on a landing page | Turning ordinary facts into promotional tiles |
| Tabs / code groups | Equivalent client languages or platform alternatives | Splitting required sequential steps across hidden panels |
| Steps / numbered headings | A procedure with a real order | Giving reference entries artificial sequence |
| Callouts | A consequential pitfall, exception, or limitation | Repeated “note” boxes for the main narrative |
| Accordions | Optional depth or lengthy diagnostic output | Hiding required setup or compatibility restrictions |
| Tables | Lookup and comparisons with repeated fields | Multi-paragraph cells that cannot be scanned |
| Diagrams | Ownership, data movement, states, or sequence | Decorative architecture boxes without a question to answer |

Use semantic headings and descriptive links. Describe a diagram's conclusion in text; do not depend on color alone. Keep code readable on a narrow screen and inspect both tab choices and expanded content in preview when changing them. Carry equivalent meaning into Sphinx without trying to reproduce every MDX component.
