PersonaPlex reference fixtures
==============================

LMGen integer trace
-------------------

The trace replay tests (``test_personaplex_delayed_cache.py``) use NVIDIA's actual ``LMGen`` at
``3428dfd95309a7f3c84fd93259ded0f810d1ff91``, with tiny random CPU transformers,
greedy sampling, production vocabulary sizes, delays and 16 depth slots.
The recorder imports no M* implementation. It records complete integer cache
and mask snapshots after every prepare, commit and voice-cache restore, plus
model inputs, targets, sampled tokens, outputs and source/version metadata.
Schema 2 includes ``content_sha256``: SHA256 of the complete fixture excluding
``argv`` and ``content_sha256``, serialized as UTF-8 JSON with sorted keys and compact
separators. Invocation paths stay in ``argv`` for provenance but do not affect this
content hash. Regenerate older traces; the replay verifies the hash first.
No checkpoint, voice preset or GPU is required. ``tqdm`` is needed by the reference
imports but absent from NVIDIA's declared dependencies, so install it explicitly.

Keep the old reference torch separate from M*'s torch. From the repository root,
with Python 3.12 available:

.. code-block:: bash

   git clone https://github.com/NVIDIA/personaplex.git /tmp/personaplex-3428dfd9
   git -C /tmp/personaplex-3428dfd9 checkout 3428dfd95309a7f3c84fd93259ded0f810d1ff91
   python3.12 -m venv /tmp/personaplex-oracle
   /tmp/personaplex-oracle/bin/python -m pip install /tmp/personaplex-3428dfd9/moshi tqdm
   /tmp/personaplex-oracle/bin/python test/personaplex/reference_oracle.py \
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
The replay covers both raw events and production ``prepare_embedding()`` replay.
The latter observes every internal prepare call, comparing its supplied dummies,
state and forward input/target/mask directly with NVIDIA's recorded events.

Each request keeps its own B=1 token cache, even when the server batches several
requests into one LM step. B=2 traces use synthetic B=2 presets; loading a real
B=1 preset into a B=2 cache via broadcasting is not supported. The cache currently requires
the preset batch to match the live batch, unlike NVIDIA's broadcasting ``copy_``.

Fixtures are not committed: generate traces outside the checkout and pass the
environment variable above. The replay also discovers a ``reference_trace.json``
beside this file. Without either it skips with instructions; an explicitly
configured missing or invalid fixture fails. The other delayed-cache tests do
not need a fixture. If macOS cannot install the reference dependencies, run the
same recorder on Linux CPU and transfer the JSON. Never synthesize a replacement
trace from M* or the symbolic plan, or claim trace parity without executing it.

LM manifest and tiny-model numerics
-----------------------------------

The same recorder has two more modes, run with the same reference venv:

.. code-block:: bash

   /tmp/personaplex-oracle/bin/python test/personaplex/reference_oracle.py \
     --source /tmp/personaplex-3428dfd9 --mode manifest --out /tmp/personaplex-fixtures
   /tmp/personaplex-oracle/bin/python test/personaplex/reference_oracle.py \
     --source /tmp/personaplex-3428dfd9 --mode lm --out /tmp/personaplex-fixtures
   PERSONAPLEX_REFERENCE_MANIFEST=/tmp/personaplex-fixtures/reference_lm_manifest.json \
   PERSONAPLEX_REFERENCE_LM=/tmp/personaplex-fixtures/reference_lm.pt \
     python -m pytest test/modular/test_personaplex_components.py -v

- ``manifest`` builds NVIDIA's full-size ``LMModel`` on the meta device with
  ``get_moshi_lm``'s ``dep_q=16`` override and records every parameter's key and
  shape plus the total (8,371,408,896). About 30 KB of JSON; no weights.
- ``lm`` builds a tiny fp32 ``LMModel`` (dim 64, 2+2 layers) that keeps the 17
  codebooks, vocabularies, delays and 16 per-step depth slots, randomises its
  RMSNorm scales, and runs NVIDIA's ``LMGen`` greedily with ``return_logits=True``
  over the same phases as the trace (B=1/2; token priming and ``.pt``-style
  embedding replay). For every forward it records the model input, the normed
  hidden state, text logits, depth inputs, all 16 depth logits and tokens, and
  the gathered output, plus the tiny state dict. About 99 MB (``torch.save``,
  loadable with ``weights_only=True``); ``content_sha256`` digests every tensor.

The LM replay test loads that state dict through the strict M* key map and replays the
whole greedy loop with M*'s cache, temporal transformer and depformer, so token
errors cascade instead of being masked. This shows algorithmic equivalence with
random weights only; it is not checkpoint, bf16 or GPU parity.

Mimi streaming reference
------------------------

.. code-block:: bash

   /tmp/personaplex-oracle/bin/python test/personaplex/reference_oracle.py \
     --source /tmp/personaplex-3428dfd9 --mode mimi --out /tmp/personaplex-fixtures
   PERSONAPLEX_REFERENCE_MIMI=/tmp/personaplex-fixtures/reference_mimi.pt \
     python -m pytest test/modular/test_personaplex_mimi.py -v   # needs transformers

``mimi`` builds NVIDIA's ``MimiModel`` with the loader's kwargs shrunk to 32
channels, 8 filters and 2 transformer layers (strides, kernels, causal padding,
RoPE, 250-step context and 32x2048 codebooks unchanged). It spreads the
zero-initialised codebooks over real-ish latents so codes vary, then runs the
model **one 1920-sample frame per call in streaming mode**, as ``offline.py``
does. Recorded: latents, 8-codebook codes, the best-vs-second codebook distance
margin per code (so fp32 ties can be excluded), PCM decoded from the codes and
from random codes, for 7 and 160 frames (160 > 124, where NVIDIA's ring KV
starts to differ from plain causal attention); an ``offline.py``-style run with
warmup, reset and interleaved encode/decode; the tiny state dict; and the
full-size model's key/shape manifest (no weights). About 11 MB.

The M* side converts the state dict with the vendored HF v4.45.0 table and
compares ``MimiCodec`` (HF submodules + the ring mask) against those outputs. The
recorder itself imports no transformers code.

Offline component replay with reference user codes (tiny random weights)
------------------------------------------------------------------------

.. code-block:: bash

   /tmp/personaplex-oracle/bin/python test/personaplex/reference_oracle.py \
     --source /tmp/personaplex-3428dfd9 --mode offline --out /tmp/personaplex-fixtures
   PERSONAPLEX_REFERENCE_OFFLINE=/tmp/personaplex-fixtures/reference_offline.pt \
     python -m pytest test/modular/test_personaplex_submodules.py -v -k offline_components

``offline`` runs NVIDIA's ``offline.py`` steps with its own helpers (``warmup``,
``load_voice_prompt_embeddings``, ``step_system_prompts``, ``_iterate_audio``,
``encode_from_sphn``, ``decode_tokens_to_pcm``) on the tiny LM and Mimi, greedy:
a ``.pt`` preset made by NVIDIA's ``embed_codes``, a 4-token persona given as ids
(no tokenizer), and 130 frames + 777 samples of stereo user audio. Recorded:
both state dicts, the preset, the user codes, the per-frame text tokens and the
trimmed PCM. About 48 MB. The M* test runs MimiEncoder → LM (conditioning,
generate) → MimiDecoder; encoder codes may differ only at fp32 argmin ties,
which it checks, and later stages use NVIDIA's codes so a tie cannot cascade.
This compares each stage against the recorded reference; it does not establish
an uninterrupted M* encoder-to-LM-to-decoder result. A separate deterministic
case with well-separated codebook choices is still needed for that claim.

Runtime controls and fixture portability
---------------------------------------

Record and test on the same machine and PyTorch build when applying tight
numerical gates. A pinned NVIDIA source revision and matching weights do not
make floating-point outputs portable between CPU libraries, operating systems
or PyTorch versions. Keep older fixtures and their failing results when
regenerating a local control; a passing regeneration does not validate the
older environment or the production GPU path.

New fixtures record the platform, architecture, PyTorch build configuration,
float32 matrix-multiplication precision and thread count, in addition to the
source hashes and Python/PyTorch versions. Their content digest covers this
provenance as well as the recorded values.

The recorder defaults to NVIDIA's declared ``torch>=2.2,<2.5`` constraint.
For a supplementary comparison on the serving runtime, explicitly pass
``--allow-unsupported-torch``. These fixtures set
``torch_supported_by_reference`` to false and must be reported separately from
supported-runtime reference results. This flag does not modify NVIDIA's
implementation or make that runtime officially supported.
