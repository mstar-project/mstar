# Annotated documentation examples

Reviewed on 2026-11-09. This is a selective, qualitative study of official documentation, not a ranking or a measured usability study. “Observed” describes the inspected pages; “Adopt” and “Limit” are editorial judgments for M*. Linked live pages can change. Examples are paraphrased, and external commands were not executed. Borrow the writing technique, not another engine's defaults, support claims, or benchmark results.

## Inference engines

### vLLM

- [Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/): the online path carries a model from launch through model listing and generation; chat-template and sampling caveats appear beside calls. **Adopt:** consistent values and local caveats. **Limit:** the combined offline/online/platform tour is too broad to copy as one M* first-run path.
- [Online serving](https://docs.vllm.ai/en/stable/serving/online_serving/): endpoint entries identify applicable models and compatibility exceptions. **Adopt:** a searchable contract beside literal endpoint paths, instead of an unqualified compatibility claim.
- [Optimization and tuning](https://docs.vllm.ai/en/stable/configuration/optimization/): startup and memory interventions explain what they save and what they cost; preemption advice starts from a recognizable warning. **Adopt:** condition, mechanism, cost, measurement. **Limit:** heuristics are specific to the engine and workload.
- [Troubleshooting](https://docs.vllm.ai/en/stable/usage/troubleshooting/): download hangs and weight-loading hangs have different investigations; diagnostic settings include overhead and cleanup advice. **Adopt:** distinguish causes before changing settings. **Limit:** a list of logging flags is not itself a diagnosis.

### SGLang

- [Quickstart](https://docs.sglang.io/docs/get-started/quickstart): platform assumptions precede launch, followed by a readiness marker, a second-terminal request, response-field verification, and shutdown. **Adopt:** an observable beginning and end to the task. **Limit:** a floating image tag is not a reproducibility guarantee.
- [Sending requests](https://docs.sglang.io/docs/basic_usage/send_request): client examples progress to streaming consumption and route other workloads separately. **Adopt:** show how to consume the stream, not only how to enable it. **Limit:** keep notebook test helpers and subprocess wrappers out of public shell examples.
- [Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning): log counters lead to conditional interventions; memory discussion separates allocations and execution phases. **Adopt:** teach interpretation before prescribing a knob. **Limit:** do not transfer numeric thresholds to M*.
- [Troubleshooting](https://docs.sglang.io/docs/references/faq): symptom headings distinguish OOM phases, hangs, and output variation. **Adopt:** searchable symptoms and the cost of a remedy. **Limit:** plausible causes still need discriminating checks.
- [Server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments): common recipes precede categorized lookup; file-versus-CLI precedence is explicit. **Adopt:** connect recipes to reference and state override order. **Limit:** exhaustive lookup is not a tutorial.

### TensorRT-LLM

- [Quick start](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html): launch, a matching request, and example JSON close the loop; container context is stated for the second terminal. **Adopt:** request and result together. **Limit:** move quantized-model alternatives after the first success.
- [Serving reference](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html): abbreviated usage, argument reference, endpoint examples, and parser guidance answer different questions. **Adopt:** mark feature maturity and selection conditions beside the feature. **Limit:** long reference material can interrupt task flow.
- [Performance guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/) and [baseline](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html): a concrete case study establishes hardware and request sizes, then dataset preparation, commands, and output. **Adopt:** make the baseline reproducible before comparing trials. **Limit:** these historical engine-build workflows are editorial examples, not current M* instructions.
- [Batch and token tuning](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html): small scheduler diagrams explain why a request fits or waits before introducing result tables. **Adopt:** a worked constraint example before knobs, with simplifications labeled. **Limit:** a schematic is not the real memory layout or a universal tuning recommendation.
- [Version 1.0.0 troubleshooting](https://nvidia.github.io/TensorRT-LLM/1.0.0/reference/troubleshooting.html): diagnostics expose tensor names, devices, types, and shapes. **Adopt:** tell readers which output fields to compare. **Limit:** giant trace dumps hide the relevant evidence; versioned workflows need their scope stated.

### Hugging Face Text Generation Inference

- [Quick tour](https://huggingface.co/docs/text-generation-inference/en/quicktour): a concrete model and pinned container lead to client variants and response JSON; a volume argument gets an inline explanation. **Adopt:** one complete route and explanations for non-obvious arguments. **Limit:** put hardware prerequisites before launch; generated text is illustrative.
- [Architecture](https://huggingface.co/docs/text-generation-inference/en/architecture): a diagram and ownership descriptions identify router, launcher, model server, and their communication. **Adopt:** explain responsibility and boundary crossings. **Limit:** full CLI-help dumps belong in reference.
- [Launcher reference](https://huggingface.co/docs/text-generation-inference/en/reference/launcher): literal arguments have environment bindings, defaults, interactions, and budget examples. **Adopt:** explain consequences and coupled limits. **Limit:** generated help still needs editing and verification.

### llama.cpp

- [Project README](https://github.com/ggml-org/llama.cpp/blob/master/README.md): concrete interactive and server commands appear early, with specialized setup linked separately. **Adopt:** put a first result before the capability inventory. **Limit:** performance adjectives do not replace evidence.
- [Build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md): a backend index routes by hardware; a recognizable architecture-detection warning leads to scoped build choices and tradeoffs. **Adopt:** organize prerequisites and remedies by the actual environment. **Limit:** an ever-growing platform-exception catalog can overwhelm installation.
- [HTTP server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): flags include defaults and environment equivalents; health responses distinguish loading and readiness. **Adopt:** explicit status/body meanings and precedence. **Limit:** do not put the entire flag table before the first serving example.

### Ollama

- [Quickstart](https://docs.ollama.com/quickstart): requests are followed by the response field readers should inspect; local and cloud paths differ in credentials. **Adopt:** finish with interpretation. **Limit:** an audience/platform/API matrix makes a poor single first-run path.
- [FAQ](https://docs.ollama.com/faq): a GPU-placement answer pairs an inspection command with output and column interpretation; configuration guidance accounts for launch environment and restart. **Adopt:** include scope and activation, not only a value. **Limit:** core configuration should have a canonical guide or reference, not live only in FAQ.
- [Troubleshooting](https://docs.ollama.com/troubleshooting): environment-specific log locations precede condition-specific remedies; backend overrides follow an identifiable log observation. **Adopt:** evidence before intervention. **Limit:** broad host changes need precise scope, not generic reinstall advice.

## Product and developer documentation

### Stripe

- [Accept a payment, hosted Checkout](https://docs.stripe.com/payments/accept-a-payment?payment-ui=checkout&ui=stripe-hosted): steps identify client/server ownership, introduce the session when needed, and verify intermediate progress; test inputs cover distinct outcomes. **Adopt:** name the owner and observable completion state. **Limit:** the inspected Ruby redirect and verification prose use different status expectations; polished examples still need internal consistency checks.
- [Error handling, Python](https://docs.stripe.com/error-handling?lang=python): an action-oriented error table precedes the taxonomy; examples show how to trigger failures and identify requests. **Adopt:** explain what each diagnostic can and cannot prove, with a reproducible failure where useful.

### Fly.io

- [Deploy an app](https://docs.fly.io/launch/deploy): the command has a working directory; build selection has explicit precedence; deployment strategies describe machine and traffic behavior. **Adopt:** defaults, effect, and scope together. **Limit:** advanced strategies should not bury the primary path.
- [Troubleshoot deployment](https://docs.fly.io/getting-started/troubleshooting): visible failures route to state checks, health results, logs, and conditional remedies. **Adopt:** explain which evidence source answers the question. **Limit:** logs do not necessarily contain a failure observed by an external check.

### Cloudflare Workers

- [CLI getting started](https://developers.cloudflare.com/workers/get-started/guide/): prerequisites, interactive choices, generated files, local URL, and a visible code change make each stage inspectable. **Adopt:** file context and checkpoints; package-manager tabs represent alternatives. **Limit:** repeated code blocks need a teaching purpose.
- [Errors and exceptions](https://developers.cloudflare.com/workers/observability/errors/): a code lookup table leads to deeper explanations and minimal broken examples; a compatibility-date caveat sits beside the affected case. **Adopt:** a small failing example plus a precise repair. **Limit:** runtime mechanisms require M* evidence before adapting the prose.

### uv

- [Working on projects](https://docs.astral.sh/uv/guides/projects/): initialization leads to a file tree and a command with visible output, then explains the files readers now have. Dependency operations describe affected artifacts. **Adopt:** action, resulting state, then explanation. **Limit:** the whole project lifecycle is useful as a guide, but too much for M*'s first inference request.

### Requests

- [Quickstart](https://docs.python-requests.org/en/latest/user/quickstart/): short request/response examples introduce one operation at a time. Timeout documentation explains the boundary of the timeout, and error examples show recognizable exceptions. **Adopt:** concrete behavior and the limit of a familiar-sounding term. **Limit:** skip the repeated assurances of simplicity; distinguish REPL transcripts from copyable scripts.

## Teaching technical mechanisms

### The Rust Book

- [Ownership](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html): one string example progresses through scope, assignment, move, clone, and calls; diagrams and compiler errors expose the changing relationships. **Adopt:** follow one object through a mechanism and explicitly defer irrelevant detail. **Limit:** general performance statements are not M* benchmark evidence.
- [References and borrowing](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html): an inconvenience in the preceding example motivates references, followed by an invalid variation, diagnostic, and repair. **Adopt:** teach a restriction through a visible consequence. **Limit:** full repeated listings are more useful in tutorials than focused how-to pages.

### Django

- [Tutorial, part 1 (5.2)](https://docs.djangoproject.com/en/5.2/intro/tutorial01/): one poll application supplies continuity; version checks, directories, named files, expected output, and a browser check support each step. Expected warnings are explained locally. **Adopt:** introduce vocabulary at first use and acknowledge likely confusion. **Limit:** a platform alternative is a support commitment, not decoration.
- [Tutorial, part 5 (5.2)](https://docs.djangoproject.com/en/5.2/intro/tutorial05/): a real defect in the same application becomes a reproduction, failing test, and repair. **Adopt:** prove the need for the change, then verify it. **Limit:** experienced contributors may not need a long introduction to why testing matters.

### PostgreSQL

- [Transactions](https://www.postgresql.org/docs/current/tutorial-transactions.html): a single account-transfer example motivates atomicity and isolation, then extends to rollback and savepoints; client-library exceptions qualify the rule. **Adopt:** show the partial state a guarantee prevents. **Limit:** simplified fragments must not masquerade as runnable procedures.
- [Using EXPLAIN](https://www.postgresql.org/docs/current/using-explain.html): query/output pairs build from a disclosed dataset; estimates, execution, instrumentation overhead, and excluded work are distinguished. **Adopt:** define measurement boundaries before interpreting numbers. **Limit:** an introductory diagnostic path does not need every advanced variant.

### SQLite

- [Atomic commit](https://www.sqlite.org/atomiccommit.html): the rollback-mode scope and hardware assumptions come before a sequence of state changes, diagrams, and recovery cases. **Adopt:** enough causal detail to reason about failure, with assumptions visible. **Limit:** historical implementation details are not current M* facts; substantial length is justified by the explanation, not a template for every page.

## Guides used to organize the observations

- [Diátaxis](https://diataxis.fr/): distinguishes tutorial, how-to, reference, and explanation by reader need. Use it to choose a page's center of gravity, not to ban short explanations beside commands or force a site-wide reorganization.
- [Google developer style highlights](https://developers.google.com/style/highlights) and [clear sentences](https://developers.google.com/tech-writing/one/clear-sentences): active actors, specific verbs, condition-first instructions, descriptive links, and sentence-case headings. Apply these to meaning; a word blacklist cannot establish clarity.
- [Write the Docs: writing software documentation](https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/): distinguish user and contributor needs, show a small use case, and avoid using FAQ as a substitute for organized documentation. Its broad README template is not a required shape for an M* guide.

## What the synthesis keeps

The recurring pattern is observable progress: a task with a result, an explanation with a concrete example, a reference with exact behavior, or a diagnostic with evidence. Good pages differ in length and voice. M* should retain technical depth while removing repeated setup prose, unsupported promises, hidden prerequisites, and unnecessary choices. These editorial conclusions inform the skill; their effectiveness still needs checking on actual M* pages.
