"""Plain Qwen3 served from real weights against transformers' Qwen3ForCausalLM (GPU).

Qwen3-8B, the registered size, serves (its head is untied). Each prompt's greedy reply,
as token ids from the server, is teacher-forced through transformers in full float32
(no TF32). At each reply token the deficit is how far its fp32 log-prob sits below the
fp32 argmax's: 0 where they agree, else the fp32 margin the server's choice gave up.

A greedy implementation in bf16 picks another token than fp32's only where its error on
the difference of the two logits exceeds that margin. transformers' own bf16 error on
that difference is the spread: the largest ``|(b_top - b_other) - (f_top - f_other)|``
over the positions of a sequence, ``other`` the token compared (the reply's, or fp32's
second). It is taken on the served replies and on transformers' own fp32 greedy replies,
which no server bug can widen, and the smaller is the tolerance; the spread itself must
stay under ``SPREAD_CEILING``. The served replies' largest deficit must be within the
tolerance. Each control (``CONTROLS``, a checkpoint edited the way a port bug would
read it) must miss it by more than ``CONTROL_FACTOR`` times.

The bound and ceiling are first estimates, to be set from the GPU runs' printed
deficits and spreads.

Qwen3-0.6B ties its head to the embedding (its file also ships the head, equal to
the embedding); it is not a registered size, so the loader's tie path is checked
on the model class itself, with the file's head left out of the stream.

Skips unless the checkpoint is fully cached, CUDA and transformers are present.

    pytest test/integration/test_qwen3_real_weights.py -q -s
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import requests
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
E2E = REPO / "test" / "e2e"

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May? Think step by step.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in three sentences why the sky is blue.",
    "List the first ten prime numbers, separated by commas, then sum them.",
    "Translate into French: The library opens at nine and closes at five on weekdays.",
    "A train leaves at 3:15 pm and arrives at 6:40 pm. How long is the trip? Show your work.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "What is 17 * 23? Answer with the computation.",
]
TOKENS = 128
# nats: a spread past this is a reference too noisy to bound anything
SPREAD_CEILING = 1.0
# a control counts only past this many tolerances
CONTROL_FACTOR = 4.0


def _scale_rope_theta(config: dict) -> None:
    # transformers 5 writes it under rope_parameters, and reads it over the top-level one
    for holder in (config, config.get("rope_parameters") or {}):
        if "rope_theta" in holder:
            holder["rope_theta"] *= 1.1


def _llama_norm_eps(config: dict) -> None:
    # Llama's RMSNorm eps (1e-5) in place of Qwen3's (1e-6): a default carried over from another port
    config["rms_norm_eps"] = 1e-5


CONTROLS = [("rope_theta x1.1", _scale_rope_theta), ("rms_norm_eps 1e-5", _llama_norm_eps)]


def _snapshot(repo: str) -> Path | None:
    """A cached snapshot holding every shard its index names (or its one shard)."""
    roots = [Path(os.environ["HF_HUB_CACHE"])] if os.environ.get("HF_HUB_CACHE") else []
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        snapshots = root / f"models--{repo.replace('/', '--')}" / "snapshots"
        for snap in sorted(snapshots.iterdir()) if snapshots.is_dir() else ():
            index = snap / "model.safetensors.index.json"
            if index.is_file():
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if all((snap / shard).is_file() for shard in shards):
                    return snap
            elif (snap / "model.safetensors").is_file() and (snap / "config.json").is_file():
                return snap
    return None


def _server_module():
    key = "_e2e_server"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, E2E / "server.py")
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


def _ids(url: str, prompt: str) -> list[int]:
    """One streamed greedy reply's token ids."""
    kwargs = {"max_output_tokens": TOKENS, "enable_thinking": False, "return_token_ids": True, "temperature": 0}
    data = {"text": prompt, "output_modalities": "text", "streaming": "true", "model_kwargs": json.dumps(kwargs)}
    r = requests.post(url + "/generate", data=data, stream=True, timeout=900)
    r.raise_for_status()
    ids = []
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            if chunk["modality"] == "error":
                raise AssertionError(base64.b64decode(chunk["data"]).decode(errors="replace"))
            ids += (chunk.get("metadata") or {}).get("token_ids", [])
    return ids


class _Reference:
    """The prompts' ids as the model renders them, and transformers in fp32 and bf16 over a
    sequence: each reply position's log-probs, and its own fp32 greedy replies. Loaded only
    once the server it is compared against has stopped."""

    def __init__(self, transformers, snap: Path):
        from mstar.model.qwen3.qwen3_model import Qwen3Model

        self.transformers, self.snap = transformers, snap
        model = Qwen3Model(model_path_hf=str(snap))
        self.prompts = [
            model.process_prompt(p, ["text"], ["text"], enable_thinking=False)
            .new_input_tensors["text_inputs"][0].tolist()
            for p in PROMPTS
        ]
        # encode_ids over the rendered text gives the ids apply_chat_template's own tokenize does
        for p, ids in zip(PROMPTS, self.prompts, strict=True):
            want = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True,
                return_dict=False, enable_thinking=False,
            )
            assert ids == list(want), f"the rendered prompt's ids differ from the template's own: {p[:40]!r}"

    def _load(self, dtype: torch.dtype):
        return self.transformers.AutoModelForCausalLM.from_pretrained(self.snap, dtype=dtype).cuda().eval()

    def _scope(self, dtype: torch.dtype):
        from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope

        # importing mstar sets TF32 matmuls process-wide; the fp32 reference runs without them
        return fp32_precision_scope(Fp32Precision("highest", False) if dtype == torch.float32 else None)

    def log_probs(self, dtype: torch.dtype, replies: list[list[int]]) -> list[torch.Tensor]:
        """Per prompt, ``[len(reply), vocab]`` float32 log-probs of each reply token's position."""
        lm, out = self._load(dtype), []
        try:
            with torch.no_grad(), self._scope(dtype):
                for prompt, reply in zip(self.prompts, replies, strict=True):
                    seq = torch.tensor([prompt + reply], device="cuda")
                    out.append(lm(seq).logits[0, len(prompt) - 1:-1].float().log_softmax(-1).cpu())
        finally:
            del lm
            torch.cuda.empty_cache()
        return out

    def greedy(self) -> list[list[int]]:
        """transformers' own fp32 greedy replies, a sequence no server touches."""
        lm, out = self._load(torch.float32), []
        try:
            with torch.no_grad(), self._scope(torch.float32):
                for prompt in self.prompts:
                    ids = torch.tensor([prompt], device="cuda")
                    out.append(lm.generate(ids, max_new_tokens=TOKENS, do_sample=False)[0, len(prompt):].tolist())
        finally:
            del lm
            torch.cuda.empty_cache()
        return out


class _Measure(NamedTuple):
    deficits: list[float]  # each prompt's largest fp32 deficit
    spreads: list[float]  # each prompt's bf16 spread

    @property
    def deficit(self) -> float:
        return max(self.deficits)

    @property
    def spread(self) -> float:
        return max(self.spreads)


def _measure(ref: _Reference, replies: list[list[int]]) -> _Measure:
    """Each prompt's largest fp32 deficit over ``replies``, and transformers' bf16 spread over
    the same positions: the error of the logit difference between fp32's top token and the
    compared one (the reply's where it differs from the top, else fp32's second)."""
    fp32 = ref.log_probs(torch.float32, replies)
    bf16 = ref.log_probs(torch.bfloat16, replies)
    deficits, spreads = [], []
    for f, b, reply in zip(fp32, bf16, replies, strict=True):
        if not reply:
            deficits.append(0.0)
            spreads.append(0.0)
            continue
        rows = torch.arange(len(reply))
        served = torch.tensor(reply)
        top2 = f.topk(2, dim=-1).indices
        top = top2[:, 0]
        other = torch.where(served == top, top2[:, 1], served)
        deficits.append(float((f[rows, top] - f[rows, served]).max()))
        error = (b[rows, top] - b[rows, other]) - (f[rows, top] - f[rows, other])
        spreads.append(float(error.abs().max()))
    return _Measure(deficits, spreads)


def _report(label: str, served: _Measure, own: _Measure) -> float:
    """Print each prompt's numbers (the PR quotes them) and return the tolerance."""
    tolerance = min(served.spread, own.spread)
    for i, (d, s, o) in enumerate(zip(served.deficits, served.spreads, own.spreads, strict=True)):
        print(f"{label}: prompt {i} deficit {d:.4f} spread served {s:.4f} transformers-greedy {o:.4f}")
    print(f"{label}: largest deficit {served.deficit:.4f}, tolerance {tolerance:.4f}")
    return tolerance


CASES = [("qwen3_8b", "Qwen/Qwen3-8B")]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _served_ids(tmp_path, name: str, model_path: Path | None = None) -> list[list[int]]:
    # max_seq_len is required and unread; the checkpoint's context bounds a request
    config = {"model": name, "max_seq_len": 4096, "resources": {"kv_cache": {"max_num_pages": 1024}},
              "node_groups": [{"node_names": ["LLM"], "ranks": [0]}]}
    if model_path is not None:
        config["model_kwargs"] = {"model_path_hf": str(model_path)}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    server = _server_module()
    command = [
        sys.executable, "-u", "-m", "mstar.api_server.entrypoint", "--config", str(path),
        "--host", server.HOST, "--port", "{port}", "--socket-path-prefix", str(tmp_path / "ipc"),
        "--upload-dir", str(tmp_path / "upload"), "--tensor-comm-protocol", "SHM",
    ]
    with server.Server(REPO, path, tmp_path / "server.log", command=command, env={"HF_HUB_OFFLINE": "1"}) as url:
        return [_ids(url, p) for p in PROMPTS]


def _snapshot_or_skip(repo: str) -> Path:
    snap = _snapshot(repo)
    if snap is None:
        pytest.skip(f"{repo} is not fully cached")
    return snap


def _edited(tmp_path, snap: Path, edit) -> Path:
    """``snap`` with its config.json edited by ``edit``, the weights linked."""
    out = tmp_path / "edited"
    out.mkdir()
    for f in snap.iterdir():
        if f.name != "config.json":
            (out / f.name).symlink_to(f.resolve())
    config = json.loads((snap / "config.json").read_text())
    edit(config)
    (out / "config.json").write_text(json.dumps(config))
    return out


@pytest.fixture(scope="module")
def reference():
    """transformers' fp32 greedy replies and their spread, once for the module."""
    transformers = pytest.importorskip("transformers")
    ref = _Reference(transformers, _snapshot_or_skip(CASES[0][1]))
    own = _measure(ref, ref.greedy())
    assert own.spread < SPREAD_CEILING, f"transformers' own bf16 spread {own.spread:.4f} is past {SPREAD_CEILING}"
    return ref, own


@pytest.mark.parametrize("name, repo", CASES, ids=[c[0] for c in CASES])
def test_plain_greedy_tokens_are_within_transformers_bf16_spread(tmp_path, name, repo, reference):
    ref, own = reference
    served = _measure(ref, _served_ids(tmp_path, name))
    tolerance = _report(name, served, own)
    assert served.spread < SPREAD_CEILING, f"bf16 spread over the served replies {served.spread:.4f}"
    assert served.deficit <= tolerance, (
        f"{name}: a served token sits {served.deficit:.4f} below fp32's top log-prob, past transformers' "
        f"bf16 spread ({tolerance:.4f})"
    )


@pytest.mark.parametrize("control, edit", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_a_port_bug_fails_the_check(tmp_path, control, edit, reference):
    """The negative controls: each must miss the tolerance by ``CONTROL_FACTOR``."""
    ref, own = reference
    name, repo = CASES[0]
    served = _measure(ref, _served_ids(tmp_path, name, model_path=_edited(tmp_path, _snapshot_or_skip(repo), edit)))
    tolerance = _report(f"{name} ({control})", served, own)
    assert served.deficit > CONTROL_FACTOR * tolerance, (
        f"{control} is not caught by a clear margin: deficit {served.deficit:.4f}, tolerance {tolerance:.4f}"
    )


def test_a_tied_head_checkpoint_loads_its_head_from_the_embedding():
    repo = "Qwen/Qwen3-0.6B"
    snap = _snapshot(repo)
    if snap is None:
        pytest.skip(f"{repo} is not fully cached")
    from safetensors import safe_open

    from mstar.model.qwen3.qwen3_model import Qwen3Model

    model = Qwen3Model(model_path_hf=str(snap))
    assert model.config.tie_word_embeddings
    lm = model.build_lm(torch.device("cuda"))
    names = {}
    for shard in snap.glob("*.safetensors"):
        with safe_open(shard, "pt", device="cuda") as f:
            names.update({key: f.get_tensor(key) for key in f.keys() if key.startswith(("model.", "lm_head."))})
    head = names.pop("lm_head.weight")
    assert torch.equal(head, names["model.embed_tokens.weight"])
    # no head in the stream: the loader ties it to the embedding
    lm.load_weights((key.removeprefix("model."), tensor) for key, tensor in names.items())
    assert torch.equal(lm.lm_head.weight, head.to(lm.lm_head.weight.dtype))
