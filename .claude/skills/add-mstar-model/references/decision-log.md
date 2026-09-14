# Porting decision log

Use `progress-artifacts/evidence/model-port-decisions.jsonl` to expose the
choices that determine a port's architecture and to identify gaps in this
skill. This is an append-only record of decisions and evidence, not a transcript
of private reasoning or routine commands.

## Required events

Append events when the corresponding decision is made:

1. `shape_classification`: before implementation, choose exactly `matched`,
   `composed`, or `unmatched-derived`.
2. `reference_assessment`: for each seriously considered reference, record
   `selected` or `rejected` and the matching or conflicting invariants.
3. `resource_mapping`: for each persistent state category, choose `graph-edge`,
   `request-state`, `reuse-resource`, `add-resource`, or `evidence-blocked`.
4. `blocker`: before marking a direction `evidence-blocked`, identify the exact
   generic extension operation that is missing and the mapping already tried.
5. `skill_feedback`: record a reusable gap that is broader than one decision.

The shape classifications mean:

- `matched`: one catalog pattern directly informs both graph topology and state
  behavior;
- `composed`: patterns from multiple references are selected per stage or
  resource; and
- `unmatched-derived`: no catalog pattern determines the design, so it is
  derived directly from checked-out graph and resource contracts.

## JSONL schema

Write one compact JSON object per line with these fields:

```json
{"schema_version":1,"seq":1,"phase":"design","event":"shape_classification","subject":"target model","options":["matched","composed","unmatched-derived"],"decision":"unmatched-derived","evidence":["upstream/config.json","mstar/engine/resources"],"reason":"No catalog pattern covers the target's stage and state invariants.","skill_signal":"missing","skill_source":"references/model-shape-map.md","skill_note":"No example combines these stages.","supersedes":null}
```

`seq` increases monotonically. `phase` is `design`, `implementation`, or
`validation`. `evidence` names inspected files, symbols, tests, or artifacts.
`reason` is a short engineering rationale. Correct an earlier choice by
appending a new event whose `supersedes` contains the earlier sequence number;
do not rewrite history.

For `skill_signal`, use exactly:

- `covered`: the cited guidance clearly supported the decision;
- `ambiguous`: guidance existed but did not distinguish the alternatives;
- `missing`: no guidance covered the observed invariant; or
- `misleading`: following the guidance would have selected the wrong shape.

Set `skill_source` to the skill or reference path that influenced the decision,
or `null` when guidance was absent. When the signal is not `covered`, use
`skill_note` to state the reusable documentation gap rather than model-specific
implementation detail.

## Developer queries

Show how the model was classified:

```bash
jq -c 'select(.event == "shape_classification") | {seq, decision, reason, evidence}' \
  progress-artifacts/evidence/model-port-decisions.jsonl
```

Find guidance that may need revision:

```bash
jq -s '[.[] | select(.skill_signal != "covered") | {seq, skill_signal, skill_source, skill_note}]' \
  progress-artifacts/evidence/model-port-decisions.jsonl
```
