# Auditing an existing documentation set

Use this for a requested audit or substantial rewrite. A typo fix does not need a site-wide inventory. Separate observed defects from preferences, and do not equate long or technical prose with bad writing.

## Map the reader's path

List the pages in navigation and the canonical source for each. Identify their audiences and primary jobs. Trace a few actual journeys: installation to first response, deployment configuration to verification, symptom to diagnosis, and model authoring to validation.

Look for missing prerequisites, forks that require knowledge the reader has not acquired, duplicate reference material, missing destinations, and sections serving a different audience. Record useful content to retain as well as problems.

## Make separate passes

**Accuracy and completeness.** Check commands, defaults, supported interfaces, model requirements, shapes, timing claims, and lifecycle descriptions against code/config/tests. Compare paired RST and MDX sections: parity means equivalent behavior and caveats, not identical typography. Flag apparent contradictions for verification before rewriting them.

**Task and structure.** Can a reader reach the promised result in order? Are alternatives distinguishable from required steps? Does the page mix a tutorial, a reference dump, and an internal design inventory? Do headings help someone find their task or error? Split only where it helps navigation; preserve the relationships with links.

**Prose and teaching.** Find empty claims, repeated introductions, unexplained terms, missing actors, and paragraphs that bury an action among exceptions. Replace abstractions with examples where they teach more. Retain causal explanations and conditions; do not flatten a subtle mechanism into an unsupported slogan.

**Examples and presentation.** Walk through code with a consistent model, address, file, and setup. Check success criteria and failure handling. Inspect tables, callouts, tabs, diagrams, code wrapping, anchors, and links in the rendered site. Distinguish an observed rendering problem from a concern inferred from MDX.

## Record actionable findings

Each finding should contain:

- File and section or line, with enough context to locate the issue.
- The reader's task and the concrete obstacle.
- Evidence: a quoted fragment, a code/config reference, a broken link, or a preview observation.
- A proposed change that preserves necessary information.
- The corresponding RST/MDX page and any anchors or incoming links affected.
- Verification needed, including hardware or runtime checks that were unavailable.

Prioritize by impact: incorrect or unusable instructions first; missing context and confusing task structure next; wording and visual polish after those. Do not manufacture numeric readability scores or call a claim false merely because it has not yet been verified.

## Rewrite in reviewable groups

Group related reader tasks rather than rewriting an arbitrary number of files at once. Installation and quickstart often need to change together; a large reference and the guides that point into it need an anchor plan. Keep runtime behavior changes separate from editorial work unless the user requests both.

Before moving content, record its facts, examples, caveats, and anchors. After editing, account for each: retained, relocated with a link, corrected with evidence, or removed with a reason. Pay special attention to defaults, frontend differences, memory/latency tradeoffs, and conditions buried in prose.

For an audit-only request, deliver the findings and recommended sequence. For an authorized rewrite, continue through edits and validation. Report commands run and results separately from static inspection and checks that could not run. Do not describe a renderer pass or a syntactically valid snippet as proof of runtime behavior.
