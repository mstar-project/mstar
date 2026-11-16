Ministral 3 3B on one H200: completed pilot
==========================================

Three independent server boots per engine, four workloads, and concurrency
1/4/16: 108 measured cells, 8,640 completed requests, zero measured failures.
The earlier aborted M* counting pilot is retained but excluded from these tables.

Each cell used 80 measured requests after five warmup waves, greedy sampling,
a 128-token output budget and disabled prefix/image reuse. The 31–33-token short
prompts and 1,923–1,962-token long prompts have verified tokenizer parity. Image
inputs use two fixtures; image token layouts match, while normalized pixel
values differ between HF and mistral-common preprocessing. All arms retained
the checkpoint's FP8 text projections, with BF16 KV and surrounding computation.

M* used -O2, torch 2.12.1+cu130 and the native integration snapshot recorded in
the original onboarding benchmark artifacts. Baselines were vLLM 0.33.0 / torch 2.14.0+cu130 and
SGLang 0.5.23 / torch 2.14.1+cu130. This is a comparison of supported deployments,
not identical framework builds. Checkpoint revision:
b35d4dfe56c142746f54dbd64f579faab2744308.

Values below are medians across boots, followed by the observed min–max. Those
ranges are not confidence intervals. Full cells are in ``per-run.csv``.

.. csv-table:: Completed requests/second at concurrency 16; higher is better
   :header: "Workload", "M*", "vLLM", "SGLang"

   "Short text", "32.75 (32.75–32.86)", "50.59 (50.33–52.15)", "42.82 (42.44–42.91)"
   "Long text", "16.02 (15.73–16.03)", "26.87 (26.33–26.91)", "24.35 (24.24–24.45)"
   "Image to text", "15.36 (14.82–15.46)", "19.46 (19.41–19.60)", "16.38 (16.35–16.48)"
   "Mixed traffic", "16.58 (15.90–17.68)", "28.70 (27.94–28.76)", "24.80 (24.63–24.85)"

.. csv-table:: Median HTTP time to first text in milliseconds at concurrency 1; lower is better
   :header: "Workload", "M*", "vLLM", "SGLang"

   "Short text", "9.71 (8.32–10.20)", "10.94 (10.18–10.99)", "29.67 (29.22–31.30)"
   "Long text", "35.71 (34.88–36.06)", "21.27 (21.11–21.37)", "32.57 (32.37–34.26)"
   "Image to text", "77.06 (76.52–77.56)", "104.01 (103.60–108.82)", "89.55 (88.80–90.63)"
   "Mixed traffic", "35.18 (34.91–35.42)", "20.35 (20.05–20.66)", "32.36 (32.24–32.96)"

M* omits streaming usage, so its output-length check re-tokenizes the complete
returned text after timing. This checks returned text, not original generated
token IDs. Do not quote the shared runner's chunk-based text_token_throughput
field as verified generated-token throughput. Competitor usage reports supply
their generated-token counts. Per-run CSV keeps decoded-text throughput separate.

These are closed-loop client measurements at fixed concurrency, not latency at
an equal offered arrival rate. The fastest cells last only a few seconds;
80 requests per cell does not establish a reliable p99 or production SLO.
Full-model GPU logits parity and semantic quality are separate, unfinished
gates. Successful serving and stored-weight checks do not substitute for them.

M* shutdown required SIGTERM escalation after the lifecycle helper's 60-second
SIGINT grace period. The helper reaped remaining processes and released GPU
memory before the next engine ran. This remains a lifecycle issue to investigate.


Scope of the evidence
---------------------

These measurements predate the PR rebase onto newer main. They describe the
original integration on base ``9f3dd66a2842b5687d5d4d2eee2546776d368f59``; they
are not measurements of the rebased PR. The preserved onboarding artifacts
contain source hashes, raw responses and server commands; those research
artifacts are excluded from the production change. The preserved source archive
SHA-256 is ``6665a771a54bc13bbfd349522e450e925ca8c1e5d10e2dedca88b9ce2fb80695``. This summary and the
108 aggregate rows retain the original benchmark, not the Fireworks experiment.

All engines ran on the same H200 with a single GPU and approximately 13 GiB
of BF16 KV capacity (131,072 tokens), an 8,192-token context limit, and disabled
prefix caching. The benchmark's M* YAML used 1,024 pages of 128 tokens; the
shipping example's smaller 128-page default was not the measured configuration.
The client used the same explicit system message and greedy settings for every
engine. The mixed corpus alternated short text, long text and the two image
fixtures. Image cases used ``test/manual/bagel/bagel.png`` and ``bagel_gen.png``.

Generate the original manifests and image inputs with::

    python -m benchmark.ministral3.corpus /tmp/ministral-corpus

The generator was checked byte for byte against all four original manifests
and both image fixtures. Run the common client described in ``README.rst``
against each engine with those inputs and the deployment settings above.
