Command A+ integration walkthrough
=================================

Status and scope
---------------

The text path runs with the full BF16 checkpoint: TP8 matched all 48 short-prompt
reference token choices, and a separate TP4 run generated complete, coherent
answers. Two TP8 logit rows still fail the original numerical acceptance limits;
the diagnosis below traces their amplification through MoE routing. Synthetic
native HTTP serving passes. Full-checkpoint HTTP serving, longer-context checks,
and production profiling remain pending.

Supported scope: text prompts, the official chat template, prefill, cached
autoregressive decode, mixed-length batches, EOS/output-budget stopping, BF16,
tensor parallelism, and literal thinking markers in streamed text. Vision,
quantization, sequence parallelism, structured reasoning/tool parsing, multi-turn
OpenAI chat adaptation, and model-level CUDA graph capture are outside this first
delivery. Use the native ``/generate`` endpoint or ``MStarClient.chat``.

Why this integration teaches M*
------------------------------

M* separates model mathematics, request execution, and device placement::

    HTTP / SDK
       -> API data worker: prompt -> official template -> token IDs
       -> conductor: choose graph walk and worker group
       -> worker / engine: prepare inputs and plan resources
       -> LLM node: embeddings -> decoder layers -> logits -> sampler
       -> graph: emit token; feed token back for decode
       -> API data worker: token bytes -> client

The model declares *what* runs. The YAML specifies *where* it runs. The engine
owns reusable state and kernels: paged KV memory, attention plans, positions, and
sampling. The conductor coordinates requests rather than implementing attention.
This is why adding another model does not require another scheduler.

The files, in learning order
---------------------------

1. ``mstar/model/command_a_plus/config.py`` translates the outer vision/text
   configuration into a validated text configuration. It rejects unsupported
   architectural flags instead of silently treating them as another model.
   ``shared_intermediate_size`` is a derived read-only property:
   ``intermediate_size * num_shared_experts``. The original intermediate size
   belongs to each routed expert; the shared MLP uses the larger derived size.

2. ``components/language_model.py`` defines the neural network. The custom
   LayerNorm computes mean and variance in FP32, applies its learned scale, and
   casts back to the input dtype. The router computes ``[T, H] @ [E, H].T``, picks
   top-k logits, applies sigmoid, and normalizes the selected scores. The MoE
   combines a weighted sum of selected routed experts with an always-active
   shared SwiGLU MLP: ``(routed + shared) / 2``.

   Each decoder layer computes::

       normalized = LayerNorm(x)
       x = x + Attention(normalized) + MoE(normalized)

   Both branches receive the same normalized input. There is no second norm
   between attention and MoE. Local layers use interleaved GPT-J RoPE; every
   fourth layer uses global attention without RoPE. A final norm produces the
   hidden states used for logits.

   The backbone accepts embeddings rather than token IDs because the node calls
   the embedding module first. The causal-LM wrapper multiplies final hidden
   states by the embedding weight transpose, gathers vocabulary shards, and
   applies ``logit_scale``. There is one embedding/output weight, not a separately
   allocated output-head copy.

3. Shared ``mstar/engine/resources/attn/`` implements the generic sliding-window
   capability. For a window W, query position t can see keys
   ``max(0, t-W+1)..t``. FlashInfer counts preceding positions, so the wrapper
   passes ``window_left=W-1``. Global attention passes ``-1``. M* already had
   attention/KV infrastructure; this work added the window option throughout
   configuration, planning, prefill, and decode. FlashInfer supplies optimized
   GPU attention kernels; M* plans their inputs and maintains the cache.

4. ``weight_loading.py`` bridges checkpoint storage and runtime layout.
   Published Q/K/V tensors become a fused QKV parameter. Gate/up tensors become
   fused gate-up parameters. Each explicit expert number maps to its correct
   slice. Existing parameter loaders handle tensor-parallel slicing. The loader
   checks full source shapes, duplicates, unexpected text weights, and every
   missing source tensor, including individual pieces of a fused destination.
   Non-text tensors are skipped. Loading streams through CPU rather than building
   a second complete GPU checkpoint.

   Similar work exists elsewhere: ``vjepa2/weight_loader.py`` and
   ``wan22/weight_loader.py`` are separate files; other models put mappings in
   their model/component classes. A separate file is an organization choice.

5. ``submodules.py`` adapts the network to M*'s execution contract.
   ``prepare_inputs`` validates one request. ``declare_step`` describes how many
   tokens each request adds. ``preprocess`` packs requests into one tensor.
   ``forward_batched`` computes and samples one token per request. Prefill selects
   the last real prompt token of **each** request. ``postprocess`` supplies the
   decode seed/feedback; ``check_stop`` handles EOS and the total token budget,
   including the token produced during prefill.

6. ``command_a_plus_model.py`` declares prefill and decode graph walks, resource
   specifications, prompt processing, byte-level output, and checkpoint loading.
   Both attention managers share one KV resource. This avoids advancing cache
   lengths twice. The engine's resource cycle is
   ``admit -> plan -> forward -> commit``: reserve pages, calculate layouts,
   run kernels, then record consumed tokens. Request removal releases pages.

7. ``assets.py`` pins metadata to checkpoint revision
   ``5fb6fde5fd12ff89356aae552e11883bc49f069b``. Automatic resolution downloads only
   listed JSON/template/index files, never checkpoint tensors. Workers fail before
   parameter allocation if required local shards are absent. The model registry,
   CLI registry, dependency extra, and ``configs/command_a_plus_tp8.yaml`` make the
   integration discoverable and describe one LLM group shared by both graph walks.

Tensor dimensions and Python reminders
-------------------------------------

``torch.zeros(E, H)`` has shape ``[E, H]``: the first argument is the first axis.
``nn.Linear(H, O)`` stores its weight as ``[O, H]`` and computes ``x @ weight.T``.
``torch.Tensor`` represents tensor data; ``nn.Parameter`` registers trainable
tensor storage on a module. There is no standard ``nn.Tensor`` class.

In a class derived from ``nn.Module``, ``super().__init__()`` initializes the
module machinery that tracks parameters and child modules. Inheritance itself
comes from the class declaration. Calling ``module(x)`` invokes ``forward(x)``
through that machinery; it also runs hooks used in our parity tests.

The production dimensions are H=4096, 128 query heads, 8 KV heads, D=128,
128 routed experts, and top-k=8. Query width is **16384**, not H. On TP8, each
rank has 16 query heads and one KV head. Its fused QKV weight is
``[(16+1+1)*128, 4096] = [2304, 4096]``. ``comm_group`` coordinates ranks:
row-parallel outputs are summed with all-reduce, and output vocabulary slices
are assembled with all-gather. SP stays at one.

Validation recorded on 2026-09-22
-------------------------------

The reference environment used PyTorch 2.11.0+cu129, FlashInfer 0.6.11.post2,
Transformers 5.17.0, tokenizers 0.23.2, CUDA toolkit 12.9, and NVIDIA H200 GPUs.
Transformers/tokenizers were installed in an isolated overlay for this checkout;
the existing vLLM environment was not upgraded.

* 42 ordinary offline tests cover configuration, component math, dense reference
  calculations, strict checkpoint loading, TP1/2/4/8 weight slices, graph flow,
  stopping, registration, and metadata-only downloads.
* With both HF checks and the official tokenizer check enabled, all 45 Command A+
  tests pass. All 9 shared sliding-window tests also pass, including real GPU
  prefill/decode and attention-wrapper CUDA graph capture/replanning.
* HF FP32 prefill plus four cached decode steps: maximum absolute logit error
  ``5.96e-8`` and normalized-hidden error ``2.98e-7``. Tolerance is
  ``atol=rtol=2e-5``. Embeddings are identical.
* Real GPU BF16, two mixed-length requests (19 and 9 tokens), prefill plus four
  decode steps, repeated after cleanup: maximum logit error ``0.003102`` at TP1
  and ``0.004883`` at TP2. Layer/norm tolerance is ``atol=rtol=0.03``; logits use
  ``0.01``. Every request returns its KV pages. Router IDs and weights match
  exactly for identical inputs to each layer's router.
* TP1 greedy tokens match all 20 comparisons. TP2 has 2/20 near-tie differences
  (the same case repeated after cleanup): leaders at ``0.3770`` and ``0.3750``
  exchange places. The test enforces numerical parity, verifies the sampler
  chooses M*'s actual argmax, and permits a reference winner change only when
  its logit gap is at most twice that row's measured maximum logit error.
  Cached reference comparisons follow the same input tokens as M*.
* The official tokenizer passes chat-template checks and byte-by-byte UTF-8
  reconstruction for multilingual text, emoji, and literal thinking markers.
  Transformers 5.17 requires explicit ``return_dict=False`` when the caller
  expects token IDs from ``apply_chat_template``; the adapter now supplies it.
* An actual API server/conductor/GPU worker loaded random tiny weights with the
  official vocabulary/tokenizer. Sequential budgets 1/6 and concurrent budgets
  3/7 produced exactly those token counts, with request cleanup logged.
  Random-weight output is not a test of language quality.

The shared CUDA IPC transfer path emits a shutdown refcount warning in both the
standalone resource harness and server teardown. Per-request page reclamation
passes; clean IPC process shutdown remains a shared-runtime follow-up, not a
claim established by these tests.

Efficiency assessment
---------------------

The implementation reuses packed attention, grouped-query KV storage, fused MoE,
tensor-parallel projections, and streaming weight loaders. Prefill computes
vocabulary logits only for each request's final token. Tied output weights avoid
another vocabulary-sized allocation. Windowed attention limits reads on local
layers, but the current cache still retains their full histories; this is not
yet a sliding/ring cache memory optimization.

The tiny timing probe (697,664 parameters, TP1 BF16, 19-token prompt, 32 warmed
decode steps) measured median planning ``0.585 ms``, forward GPU-event interval
``5.382 ms``, and total step ``6.052 ms``. Peak PyTorch allocated memory was
``322.6 MiB`` including resource workspaces. The measurement excludes HTTP,
worker scheduling, and reference execution. The GPU-event interval includes
idle gaps between Python-launched kernels. These numbers cannot predict
production latency or throughput.

The main optimization opportunities are:

* Model-level CUDA graphs/compilation: the initial node deliberately runs eager.
  Shared attention wrapper graph tests do not establish whole-model graph safety.
* Tensor-parallel communication: attention, routed experts, and shared experts
  currently reduce separately (three reductions per layer). Combining reductions
  would require changes to component contracts and fresh numerical validation.
* LayerNorm/router and residual operations use several small kernels; profile
  production traces before choosing fusion work.
* Local layers keep old KV pages. Reclaiming them requires a cache-layout design
  that preserves global layers and absolute RoPE positions.
* Checkpoint shards are read by every TP rank and sliced on CPU. This bounds
  temporary memory but can make shared-storage startup bandwidth a bottleneck.

The production TP8 meta model has 27,296,665,600 parameters per rank: BF16 storage
is about **50.84 GiB/rank**, plus **4 GiB/rank** for the default KV arena
(2048 pages x 128 tokens x 32 layers x K/V x 1 KV head x 128 dimensions x 2 bytes).
Workspaces, activations, collectives, and allocator overhead are additional.
The roughly 437.5 GB download and eight-GPU validation were authorized on
2026-09-24. Real weights now load successfully in both HF and M*. The short-prompt
comparison and numerical diagnosis below are complete; the original numerical
gate still fails. Longer-context checks, real-checkpoint HTTP concurrency, TTFT,
inter-token latency, throughput, and peak-memory benchmarks remain pending.

Reproducing the checks
---------------------

Use a compatible CUDA/PyTorch environment; install this model's extra with
``pip install -e '.[command_a_plus]'``. No test below fetches real weights::

    python -m unittest discover -s test/command_a_plus -v
    MSTAR_TEST_HF=1 python -m unittest test.command_a_plus.test_reference -v

    # Add COMMAND_A_METADATA_DIR=/local/pinned/metadata for the real tokenizer test.
    CUDA_VISIBLE_DEVICES=0 MSTAR_TEST_HF=1 MSTAR_TEST_CUDA=1 \
      python -m unittest test.command_a_plus.test_reference -v
    CUDA_VISIBLE_DEVICES=0 MSTAR_TEST_CUDA=1 \
      python -m unittest discover -s test/attention -v

    CUDA_VISIBLE_DEVICES=0,1 MSTAR_TEST_HF=1 MSTAR_TEST_CUDA=1 MSTAR_TEST_TP=1 \
      torchrun --standalone --nproc-per-node=2 --module unittest \
      test.command_a_plus.test_reference.ReferenceTests.test_bf16_paged_gpu_batched_prefill_decode_and_cleanup -v

    CUDA_VISIBLE_DEVICES=0 python -m test.command_a_plus.benchmark_tiny

For the HTTP smoke, create a **random** checkpoint in an empty scratch directory::

    python -m test.command_a_plus.create_smoke_checkpoint \
      --metadata-dir /local/pinned/metadata --output-dir /tmp/command-a-smoke
    CUDA_VISIBLE_DEVICES=0 mstar-serve \
      --config /tmp/command-a-smoke/deployment.yaml --host 127.0.0.1 --port 18937 \
      --tensor-comm-protocol SHM --socket-path-prefix /tmp/command-a-smoke/socket
    # In another terminal:
    python -m test.command_a_plus.smoke_http --url http://127.0.0.1:18937

The production launch configuration is ``configs/command_a_plus_tp8.yaml``.
Set ``model_kwargs.checkpoint_dir`` to the approved, complete local checkpoint
before launching ``mstar serve command_a_plus --gpus 0,1,2,3,4,5,6,7``.
Without it, registry resolution fetches pinned metadata only and worker loading
fails with an explicit missing-weights error.

Full-checkpoint runner
----------------------

``test/command_a_plus/validate_checkpoint.py`` saves an eager Hugging Face
reference from the official multimodal wrapper, then compares M* in a separate
``torchrun`` invocation. Artifacts include rendered prompts, token IDs, full
per-step logits, loading diagnostics, greedy output, memory peaks, and timings.
The HF reference runner additionally requires ``accelerate`` for device placement
(the tested environment used version 1.12.0).
The reference explicitly places the rotary-position buffer as well as model
parameters across devices. A small sharded checkpoint with the real vocabulary
has passed this path at TP2 before the full run.

The full-model acceptance limits are declared before running: finite logits,
RMSE at most 0.15, maximum absolute error at most 1.0, and cosine similarity at
least 0.999. The report also records maximum absolute
error, top-10 overlap, and any greedy-token difference with its reference logit gap.
These limits accommodate BF16 accumulation across the larger stack; inspect the
actual errors and generated text before declaring production readiness. Numerical
checks precede timing runs, and each benchmark prompt shape gets its own warmup.

``test/command_a_plus/run_validation.py`` can queue the sequence behind an existing
download. It waits for ``checkpoint_path.txt`` in its run directory and requires
eight GPUs with no compute processes, no utilization, and less than 256 MiB used
for 60 consecutive seconds before each model load. It never kills another user's
process. The queue runs HF reference, TP8 parity plus 128/1024-token timing probes,
then native HTTP smoke tests, stopping its own server afterward. It writes an
atomic ``status.json`` and per-stage logs; a failed stage stops the sequence.
The availability wait expires after 12 hours by default. This checks availability
but does not reserve a shared node against later arrivals.

Full-checkpoint numerical diagnosis (2026-09-24)
-----------------------------------------------

The pinned checkpoint loaded without missing, unexpected, or mismatched weights.
Three prompts, each with 16 cached generation steps, gave 48/48 identical greedy
token choices at TP8. Two logit rows failed the original RMSE/cosine gate. The
pipeline stopped before benchmarks and HTTP tests; this remains a failed
production validation, despite matching the token choices in these short cases.

A separate free-running TP4 smoke test with the real checkpoint and a 256-token
budget produced complete answers: ``4``, ``Paris.``, and
``Welcome aboard—great to have you on the team!`` for the three prompts. All
requests stopped on EOS, after 63, 43, and 48 generated tokens respectively
(including the default reasoning text and structural tokens). Logits remained
finite and each request returned its KV pages. This establishes basic generation
through real M* resources, separately from the still-failing numerical gate;
it is not an HTTP or broad quality evaluation. Reproduce it with::

    torchrun --standalone --nproc-per-node=4 --module \
      test.command_a_plus.generate_checkpoint --checkpoint /local/checkpoint \
      --output /scratch/text-smoke.json --tokens 256

An opt-in ``--trace`` mode records embedding, layer input, LayerNorm, attention,
shared-expert, combined-MoE, layer output, and router activations. It also performs
a separate counterfactual pass with HF expert IDs **and routing weights** held
fixed while M* computes the rest of the network. Counterfactual outputs are
diagnostic artifacts, not a replacement model or an alternative acceptance gate.
Trace mode synchronizes and copies activations to CPU; its timings are unsuitable
for performance claims.

The full trace reproduced both original HF and M* logits bit for bit. Results:

* Case 0, step 15 (zero-based): the first changed expert set is in layer 13.
  HF selects expert 68 for the final slot; M* selects expert 18. The HF gap
  between the eighth and ninth scores is 0.0078125. At that layer, relative L2
  error grows from 1.78% in the layer input to 9.06% in the MoE output.
* Case 2, step 11: the first changed set is in layer 3, replacing expert 120
  with expert 43. The HF boundary gap is 0.03125. Relative L2 error grows from
  1.03% in the input to 5.23% in the MoE output. Additional routing changes
  amplify differences later in the network.
* Holding HF routing fixed reduces the two final-logit RMSE values from
  0.188400 to 0.026241 and from 0.179547 to 0.032375. Across all 48 rows, the
  counterfactual maximum RMSE is 0.058341, minimum cosine is 0.999807, and
  maximum absolute error is 0.25; all original thresholds pass in that
  counterfactual. Token choices still match 48/48.
* An independent replay of all 1,536 trained-router snapshots (32 layers x
  16 steps x 3 cases), using identical HF inputs, matches router logits,
  selected IDs, and routing weights exactly. The router implementation itself
  does not introduce a difference on identical inputs.
* A TP4 diagnostic changes which rows fail; holding reference routing fixed
  again brings all rows within the same limits (maximum RMSE 0.066673).

These observations localize the large-error amplification to MoE routing.
Smaller differences already appear in the first layer's attention and MLP, where
fused kernels and tensor-parallel reductions use different BF16 arithmetic from
the eager HF reference. The evidence is consistent with those small numerical
perturbations changing near-boundary top-k choices; it does not isolate one
particular kernel as the sole source. Fixing router inputs reproduces HF exactly.
Forcing reference routes is not deployable, and the production tolerances and
model code have not been changed to make this experiment pass. Broader prompt,
context, and precision-baseline checks are needed before revising acceptance
criteria or claiming production readiness.

Reproduce activation diagnosis with a complete local checkpoint and a separate
scratch directory (HF must finish before starting M*)::

    python -m test.command_a_plus.validate_checkpoint reference \
      --checkpoint /local/checkpoint --output /scratch/trace --tokens 16 --trace
    torchrun --standalone --nproc-per-node=8 --module \
      test.command_a_plus.validate_checkpoint mstar \
      --checkpoint /local/checkpoint --output /scratch/trace --tokens 16 --trace
    # The M* command still exits nonzero when its unmodified baseline fails.
    python -m test.command_a_plus.compare_traces /scratch/trace
    CUDA_VISIBLE_DEVICES=0 python -m test.command_a_plus.replay_routers \
      --checkpoint /local/checkpoint --traces /scratch/trace

``activation_comparison.json`` reports each component and router-set difference.
``reference_routing.json`` contains the counterfactual logit errors;
``router_replay.json`` records the exact same-input router checks. Model startup
also exposed a shared expert-alignment JIT lock race; some workers used the
existing torch fallback. This did not change the reproduced logits, but is a
separate startup/performance follow-up.
