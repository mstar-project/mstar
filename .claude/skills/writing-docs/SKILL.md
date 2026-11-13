---
name: writing-docs
description: Write, review, or restructure M* user and contributor documentation, including Mintlify MDX and Sphinx guides. Use for documentation changes and audits; not for PR descriptions or code comments.
---

# Writing M* documentation

Help a reader do something, find a fact, or understand a mechanism. Prefer concrete actions and explanations over claims about how powerful, flexible, or easy M* is. Concision means removing work for the reader, not deleting the conditions that make an instruction correct.

Read [AGENTS.md](../../../AGENTS.md), especially invariants 9 and 10, and [the Mintlify contributor guide](../../../docs/mintlify/CONTRIBUTING.md). These govern source parity, validation, and evidence. This skill adds editorial guidance; it does not authorize a wider rewrite than requested.

## Start with the reader's job

Identify the audience, what they already have, and the outcome this page should deliver. Choose a primary form:

| Reader needs to… | Write… | Organize around… |
| --- | --- | --- |
| Get a first working result | A tutorial or quickstart | One supported route from prerequisites to an observable result |
| Complete a known task | A how-to guide | Conditions, actions, verification, and relevant recovery |
| Look up exact behavior | Reference | Consistent entries with types, defaults, units, scope, and exceptions |
| Understand a design | An explanation | A concrete example, the mechanism, its consequences, and limits |
| Recover from a failure | Troubleshooting | The symptom, a discriminating check, a remedy, and a success check |

Use these as editorial choices, not mandatory headings. A short explanation beside a command often saves a page jump. Split a section when it serves a different reader or task, not because it crosses an arbitrary word count. See [patterns and examples](references/patterns.md) when drafting or restructuring; see [annotated sources](references/sources.md) for the observations behind these choices.

## Write from verified behavior

- Read the relevant implementation, config, tests, and existing canonical guide before rewriting technical claims. Existing docs are evidence of intent, not proof of current behavior.
- Keep a lightweight working list of claims that need checking: commands, defaults, model support, shapes, timing, ordering, and frontend coverage. Resolve them or report the uncertainty; do not turn plausible wording into a guarantee.
- For M*, distinguish the CLI, Python and Rust frontends, SDK, and model-specific adapters where their behavior differs. Do not generalize one model, modality, accelerator, or deployment to all of them.
- Performance guidance needs the workload, hardware, configuration, metric, and evidence required by AGENTS.md. Say when a recommendation trades memory for latency, startup for steady-state speed, or throughput for responsiveness. Do not invent a number to replace an adjective.
- Preserve constraints and exceptions while simplifying. If a detail interrupts the main task, move it to a named section and link it at the decision point. Do not remove it merely because it is complicated.

## Make examples usable

For a first-run path, choose one model, server, endpoint, and client that work together. Put alternative models and interfaces after the first successful result, or on separate linked paths. A code block should not silently switch servers or require incompatible setups.

State the prerequisites before the action they enable: hardware, package or extra, credentials or checkpoint access, working directory, and any required input file. Link established installation details rather than copying a second installation guide. Distinguish commands run on the server from those run in another terminal or on a client machine.

Use complete imports and defined inputs in runnable examples. Keep shell commands free of prompt characters; put output in a separate block. Mark excerpts and pseudocode explicitly, and explain replacement values before use. Use real repository examples where possible so the guide and runnable code have one behavior to maintain.

Show how to recognize success: a response field, generated file, status code, or other visible effect. For generated model text, describe the expected kind of output rather than promising an exact sentence. A health response verifies server health, not the correctness of a model's output. Do not fabricate logs, timings, or a successful execution.

## Edit for meaning

- Open with the useful fact, outcome, or decision. Cut introductions that only announce the page, feature inventories that do not help this reader, and conclusions that repeat the preceding section.
- Name the actor and action: which process receives, allocates, waits, or returns? Explain an unfamiliar term with the example that needs it. Keep technical names consistent rather than cycling through synonyms.
- Put a condition before its instruction. Put a default, limitation, or exception beside the claim it qualifies.
- Replace promotional adjectives with behavior and scope. Words such as “seamless,” “powerful,” “robust,” “simply,” and “just” are editing signals, not a banned-word test. A sentence can avoid all of them and still say nothing.
- Prefer connected prose for reasoning, numbered steps for an ordered procedure, bullets for parallel facts, and tables for comparisons or repeated fields. Do not turn every sentence into a bullet or every paragraph into a heading.
- Preserve useful depth. An explanation of resource ownership or concurrency may need several paragraphs and a diagram. Shorter is worse if the reader can no longer predict what happens.

## Use formatting to expose structure

Use sentence-case headings that name the task, concept, or symptom. Avoid generic headings such as “Overview” when the subject can be named. Link to the specific destination with descriptive text. Keep existing explicit anchors stable even when headings change.

In Mintlify, cards belong at navigation choices, tabs at genuinely alternative implementations, and steps at ordered procedures. Keep prerequisites, required actions, and essential limitations visible. Use callouts for a consequential exception or pitfall, not ordinary paragraphs. Put long explanations below a table instead of forcing them into cells. A diagram should explain ownership, sequence, or data movement; label those relationships and explain its takeaway in text. Do not claim to have checked the rendered layout from source alone.

## Audit and finish

For a requested audit, use [the audit method](references/audit.md). For a small edit, apply only the relevant checks.

1. Confirm that the reader can follow the main path without guessing a missing prerequisite, value, or next action.
2. Compare the rewrite against the original for lost behavior, caveats, examples, and links. Update corresponding RST and MDX content in the same change; preserve their anchors and check navigation when moving content.
3. Run the applicable checks from the Mintlify contributor guide and inspect affected rendered pages when changing site content or layout. Test executable examples when the environment supports them; otherwise state what was checked statically and what still needs hardware, weights, or credentials. A successful docs build does not validate a serving command.
4. Report the changes, checks and results, and any unresolved technical claims. Keep the evidence in the review notes or PR rather than cluttering every user-facing paragraph with internal investigation details.

The completion criterion is a reader who can act or reason correctly, not a target length, a readability score, or a page decorated with more components.
