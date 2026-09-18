# fuzzer

This package does property-based fuzzing of the M\* internals. It tests the
engine, not the models.

New models, new engine resources and new accelerators reach code paths that no
released model uses. A hand-written test covers the paths that a person thought
of. A fuzzer covers the space around them. In M\*, the model is data and the
engine is the system under test. For this reason the generators make graphs,
schedules and faults. They do not make model inputs.

## Layout

| directory | what it drives | cost of one case | hardware |
|---|---|---|---|
| `tier0/` | the pure-Python state machines: the page allocator, the refcounts, the graph readiness, the scheduler and the resource lifecycle | approximately 1 ms | CPU |
| `tier1/` `model_run` | the real graph layer, over a generated model, against an interpreter of that model | approximately 2 ms | CPU |
| `tier1/` `kv_run` | the same, plus the real resources: the KV cache, the positions, the step runner, pre-planning, capture buckets and forks | approximately 25 ms | CPU |
| `tier1/` `kv_race` | the real resources under several threads, with the interleaving inside the case | approximately 2 ms | CPU |
| `tier1/` lane 1a | the real conductor and the real worker, over a generated synthetic model | approximately 1 s | CPU |
| `tier1/` lane 1b | lane 1a on a real device: the behavior of a resource, the pre-plan threads, CUDA graphs, eviction | seconds | GPU, on `team1` |
| `tier2/` | tier 1, plus injected faults: out-of-memory errors, aborts, evictions and changed messages | seconds | GPU, on `team1` |
| `common/` | the shared driver: cases, generation, shrinking and the corpus | — | — |

Tier 0 is complete. Tier 1 has two machines. Both generate the model, run it,
and compare every value that reaches the client against an interpreter of the
same model. `model_run` runs it over the real graph layer. `kv_run` runs it
over the real resources of `mstar/engine/resources/` as well, and the value of
a step covers the cache pages that the step reads. The rest of lane 1a, lane
1b and tier 2 have design notes only. All of them use `common/` without
changes.

Tier 0 drives the resource lifecycle against stubs, so it covers the order and
the scope of the calls into a resource and no behavior of one. `kv_run` is
where that behavior is covered, including the three that tier 0 can never
reach: pre-planning a step ahead, the padded addressing of a capture bucket,
and forks between cache streams.

Nothing runs the fuzzer automatically yet. `.github/workflows/ci.yml` does not
call `pytest fuzzer/`, so tier 0 and `tier1/model_run` are cheap enough for
each pull request but no job runs them. Tier 1 lane 1b and tier 2 need a GPU,
and the CI runners are `ubuntu-latest`. Those two lanes belong in a nightly job
on the cluster.

## What a case is

A case is a program. It is not a value. A case has two parts:

* a **config**, which gives the shape of the system: the sizes, the topology and
  the limits
* an ordered list of **ops**, which drive one state machine

A failure report is thus short, and a person can read it:

```
machine: micro_scheduler
config:  {'num_requests': 2, 'num_nodes': 3, ...}
ops:
    0  make_ready(0, 0)
    1  make_ready(1, 0)
    2  next_batch(2, 0, -1, 1)
    3  pending_remove(1, 1)
    4  next_batch(1, 0, -1, 0)
```

All of this code uses only the standard library, and all of it is
deterministic. The same case always gives the same result. Each machine
replaces the parts that would prevent this. For example, the scheduler machine
replaces the clock.

## Shrinking and the corpus

A new failure has 80 ops. Almost all of them are not related to the bug. The
shrinker makes the case small in three steps:

1. It removes groups of ops. This is delta debugging.
2. It makes the remaining ops simpler.
3. It makes the config smaller.

Each candidate case must fail with the same **signature**. If the signature
could change, a shrink step could move the search to a different bug.

A signature stays the same while the case becomes small:

* For an invariant of the harness, the signature is the name of the invariant.
* For an error that the system under test raised, the signature is the deepest
  stack frame inside the repository.

The harness writes each small failure to `<tier>/corpus/<machine>/`. Each case
in the corpus has a status:

* `open` — a known bug that nobody corrected yet. The tests replay the case on
  each run. The case must fail again in the same way. If the case starts to
  pass, the tests tell you to change its status.
* `fixed` — a regression guard. The case must pass.

The corpus is the regression suite, and it becomes larger at no cost.

## How to run it

```bash
pytest fuzzer/                                    # the corpus, a short search, and the oracle self-tests
python -m fuzzer.tier0 list
python -m fuzzer.tier0 run --seeds 20000          # all of the machines
python -m fuzzer.tier0 run --machine graph_io --time 120 --all
python -m fuzzer.tier0 replay fuzzer/tier0/corpus/graph_io/<case>.json
python -m fuzzer.tier1 run --seeds 20000 --all    # the generated models
```

Each tier has the same command line. `--save LABEL` writes each small failure
to the corpus.

Three limits are important when you read a report:

* A search stops at the first failure of each machine. Use `--all` to collect
  the other signatures.
* A case stops at its first failed op. Thus a frequent failure can hide an
  invariant that comes later in the same case.
* A tier only finds the faults that its model can represent. Read the
  "Not covered" list in a machine's header before treating a pass as
  evidence.
  `tier0/step_runner.py` is the clearest example: it tests the sequence of the
  calls into a resource, and no behavior of any resource.

## How to write a machine

Write a subclass of `fuzzer.common.StateMachine`. Then add the subclass to
`machines.py` in the tier. The machine must have these two properties:

* **It must be deterministic.** Do not read the clock, the address space or a
  global variable. Replace them.
* **It must be tolerant.** The shrinker removes ops. Thus `execute` receives
  ops that the generator made for a state that no longer exists. If an op does
  not apply, skip it. Only the system under test can make a case fail.

Two further rules:

* **Model only reachable states.** An op that can construct a state the real
  system cannot produce reports false failures. Constrain the op instead: a
  false failure costs more than the extra coverage is worth.
* **Write a self-test for each oracle.** `test_tier0.py` breaks each system
  under test on purpose. The machine must find the damage. An invariant that
  cannot fail looks like coverage, but it gives none.

## How to write the documentation here

The comments and the documents in this package follow the principles of
ASD-STE100 Simplified Technical English. Many readers do not have English as a
first language. These rules keep the text clear:

* Write short sentences. Use a maximum of 20 words for an instruction, and a
  maximum of 25 words for a description.
* Give one idea in each sentence.
* Use the active voice. Write "the shrinker removes the op". Do not write "the
  op is removed".
* Use the simple tenses. Write "the search finds". Do not write "the search has
  found".
* Use the same word for the same thing each time. Do not change the word to
  make the text more interesting.
* Do not use idioms, metaphors or humor.
* Keep the articles ("the", "a") and the relative pronouns ("that", "which").
* Use a maximum of three words in a noun cluster. Write "the limit of the
  batch size", not "the batch size limit value".
* Keep each paragraph to a maximum of six sentences.
* Use a vertical list for more than two related items.
