PersonaPlex Commit 1 reference trace
====================================

T3 uses NVIDIA's actual ``LMGen`` at
``3428dfd95309a7f3c84fd93259ded0f810d1ff91``, with tiny random CPU transformers,
greedy sampling, production vocabulary sizes, delays and 16 depth slots.
The recorder imports no M* implementation. It records complete integer cache
and mask snapshots after every prepare, commit and voice-cache restore, plus
model inputs, targets, sampled tokens, outputs and source/version metadata.
Schema 2 includes ``content_sha256``: SHA256 of the complete fixture excluding
``argv`` and ``content_sha256``, serialized as UTF-8 JSON with sorted keys and compact
separators. Invocation paths stay in ``argv`` for provenance but do not affect this
content hash. Regenerate older traces; T3 verifies the hash before replaying.
No checkpoint, voice preset or GPU is required. ``tqdm`` is needed by the reference
imports but absent from NVIDIA's declared dependencies, so install it explicitly.

Keep the old reference torch separate from M*'s torch. From the repository root,
with Python 3.12 available:

.. code-block:: bash

   git clone https://github.com/NVIDIA/personaplex.git /tmp/personaplex-3428dfd9
   git -C /tmp/personaplex-3428dfd9 checkout 3428dfd95309a7f3c84fd93259ded0f810d1ff91
   python3.12 -m venv /tmp/personaplex-oracle
   /tmp/personaplex-oracle/bin/python -m pip install /tmp/personaplex-3428dfd9/moshi tqdm
   /tmp/personaplex-oracle/bin/python test/scratch/personaplex_reference_oracle.py \
     --source /tmp/personaplex-3428dfd9 --mode trace --out /tmp/personaplex-trace
   PERSONAPLEX_REFERENCE_TRACE=/tmp/personaplex-trace/reference_trace.json \
     python -m pytest test/modular/test_personaplex_delayed_cache.py -v

The recorder verifies the checkout SHA, rejects tracked changes under ``moshi``,
checks the imported ``LMGen`` path, and records hashes of the Python source files.
It calls the reference ``step``, ``step_embeddings`` and voice replay path; its
subclass only observes the methods' inputs and outputs. Coverage includes B=1
and B=2, token priming, synthetic ``.pt`` embedding replay and ``cache.copy_``,
6 silence / 3 text / 6 silence frames, 20 user frames and the final step.
The M* tests replay recorded sampled integers, so this tests cache equivalence,
not M* transformer numerics or real-checkpoint parity.
T3 covers both raw event replay and production ``prepare_embedding()`` replay.
The latter observes every internal prepare call, comparing its supplied dummies,
state and forward input/target/mask directly with NVIDIA's recorded events.

PR1 serves B=1. B=2 traces use synthetic B=2 presets; loading a real B=1 preset
into a B=2 cache via broadcasting is deferred. The cache currently requires
the preset batch to match the live batch, unlike NVIDIA's broadcasting ``copy_``.

Until the fixture policy (worklog D9) is resolved, generate traces outside the
checkout and pass the environment variable above. T3 also discovers a future
``reference_trace.json`` beside this file. Without either it skips with instructions;
an explicitly configured missing or invalid fixture fails. T1, T2 and T4 do not
need a fixture. If macOS cannot install the reference dependencies, run the same
recorder on Linux CPU and transfer the JSON. Never synthesize a replacement
trace from M* or the symbolic plan, or claim T3 parity without executing it.
