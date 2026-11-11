Ministral 3 serving comparison
=============================

Use the same pinned checkpoint and JSONL request manifest on every engine. Each
line contains ``prompt`` and optionally ``image_path`` relative to the manifest.
The runner reuses ``benchmark.runner`` with the same OpenAI client for all arms::

    python -m benchmark.ministral3.run --engine mstar --url http://127.0.0.1:8100 \
      --manifest /path/to/short.jsonl --concurrency 4 --output-tokens 128 \
      --output-dir /path/to/results/mstar/short/c4-r1

To recreate the original H200 corpus, run
``python -m benchmark.ministral3.corpus /tmp/ministral-corpus`` with a new output
directory. ``RESULTS.rst`` and ``per-run.csv`` preserve the original comparison
and distinguish its source snapshot from the rebased integration.

Start servers through ``.claude/skills/benchmarking/scripts/server.sh``. The
output directory must be new. Preserve stdout in a separate launcher log, or
tee it to a sibling file: the runner creates its output directory itself.

Outputs include the usual ``results.json`` and generated text, plus:

* ``run-manifest.json``: checkpoint revision, corpus/image hashes, engine label
  and load settings. The label identifies the engine; the common HTTP adapter
  still appears as ``ours_openai`` in ``results.json.system``.
* ``requests.jsonl``: the selected inputs in their original order.
* ``raw-requests.json``: all measured requests, including failed requests,
  client chunk arrival times relative to HTTP send, counts and latency.

An error or incorrect reported output length makes the command exit nonzero.
The client requests usage counts. Each raw record identifies ``server_usage``
or ``chunk_estimate``; ``token_counts_verified_by_server`` in the aggregate is
false if any request lacked usage. Without usage, the length check re-tokenizes
the complete returned text after timing. This handles Unicode tokens coalesced
into one HTTP chunk, but does not recover the original generated token IDs. Do not
quote those aggregate token-throughput or token-normalized latency fields as
verified token measurements. Chunk arrival intervals measure HTTP delivery,
not device token-generation intervals.
Warmup reuses measured prompts: disable prefix and media reuse when measuring
uncached work, or explicitly label a cached experiment. Record server commands,
versions, source hashes, resource capacities and cache policies outside the cell.

These are performance measurements, not output-quality checks. Establish
correctness and numerical-policy equivalence separately. Repeat across server
boots, interleave engines and preserve every cell; one successful run does not
establish a performance advantage or reliable tail-latency estimate.
