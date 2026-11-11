"""Plain Qwen3 served from real weights against transformers' Qwen3ForCausalLM (GPU).

Qwen3-8B, the registered size, serves (its head is untied). Each prompt's greedy reply
comes back with the server's own log-prob of each token (the sampler's raw log-softmax,
T = 1). The reply is teacher-forced through transformers in full float32 (no TF32) and
in bf16. At every reply position t, with s_t the served token and f, b the fp32 and bf16
log-probs:

    e_srv(t) = |served(s_t) - f[t, s_t]|      e_bf(t) = |b[t, s_t] - f[t, s_t]|

The gate is, per prompt set, R = trim(e_srv).mean() / trim(e_bf).mean(), each error's
positions under its own 0.98 quantile: a bf16 port errs from fp32 as transformers' bf16
does. Trimmed, because one served greedy flip in a reply of ~100 positions moves a plain
mean by most of the margin (Alice: 1.8 with it, 1.1 without). The real model needs
R <= R_MAX on every set; the control (``CONTROLS``, a checkpoint edited the way a port bug
would read it) must fail the same gate on the long set. Calibrated on an H100 (Qwen3-8B,
three long prompts): the real model at most 1.10 per prompt and 0.93 pooled, rope_theta
x1.1 at least 1.95 and 2.42. rms_norm_eps 1e-5 isn't a control: its served log-probs match
the real model's at every length. The plain-mean and p90 ratios, the greedy-flip counts
and the deficit (fp32's margin given up at a flip) are printed only.

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
DATA = Path(__file__).parent / "data"
# Fixed public-domain prose, committed so it never moves (transcribed, not a given edition), and
# questions that ask for a full reply: Lewis Carroll's Alice's Adventures in Wonderland (1865), chapter I
# into II, 3.4k rendered tokens; Poe's The Cask of Amontillado (1846), 3.2k; Austen's Pride and Prejudice
# (1813), chapter I into II, 1.3k.
LONG_PROMPTS = [
    (DATA / "alice_chapters_1_2.txt").read_text()
    + "\n\nRetell, in order and in detail, everything that happens to Alice in the passage above, from the "
    "riverbank to the end of the passage.",
    (DATA / "poe_cask_of_amontillado.txt").read_text()
    + "\n\nRetell the story above in order and in detail: why Montresor wants revenge, how he lures Fortunato "
    "into the vaults, and how it ends.",
    (DATA / "austen_pride_and_prejudice_1_2.txt").read_text()
    + "\n\nDescribe in detail the conversations in the passage above: what Mrs. Bennet wants, how Mr. Bennet "
    "answers her, and what the narrator says of each of their characters.",
]
SETS = {"short": PROMPTS, "long": LONG_PROMPTS}
# every prompt, in the order the server and the reference run them, with its set
ALL_PROMPTS = [(name, prompt) for name, prompts in SETS.items() for prompt in prompts]
TOKENS = 128
# the real model at 0.93 and the control at 2.42 pooled over the long set: about 1.6x each side
R_MAX = 1.5
# each error's positions past this quantile are left out of its mean
TRIM = 0.98


def _scale_rope_theta(config: dict) -> None:
    # transformers 5 writes it under rope_parameters, and reads it over the top-level one
    for holder in (config, config.get("rope_parameters") or {}):
        if "rope_theta" in holder:
            holder["rope_theta"] *= 1.1


CONTROLS = [("rope_theta x1.1", _scale_rope_theta)]


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


class _Served(NamedTuple):
    ids: list[int]
    logprobs: list[float]  # the server's log-prob of each of ``ids``


def _ids(url: str, prompt: str) -> _Served:
    """One streamed greedy reply's token ids and the server's log-prob of each."""
    kwargs = {"max_output_tokens": TOKENS, "enable_thinking": False, "return_token_ids": True, "temperature": 0,
              "return_logprobs": True}
    data = {"text": prompt, "output_modalities": "text", "streaming": "true", "model_kwargs": json.dumps(kwargs)}
    r = requests.post(url + "/generate", data=data, stream=True, timeout=900)
    r.raise_for_status()
    ids, logprobs = [], []
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            if chunk["modality"] == "error":
                raise AssertionError(base64.b64decode(chunk["data"]).decode(errors="replace"))
            metadata = chunk.get("metadata") or {}
            ids += metadata.get("token_ids", [])
            logprobs += metadata.get("logprobs", [])
    assert len(logprobs) == len(ids), f"{len(ids)} token ids and {len(logprobs)} log-probs"
    return _Served(ids, logprobs)


class _Reference:
    """The prompts' ids as the model renders them, and transformers in fp32 and bf16 over a
    sequence: each reply position's log-probs. Loaded only once the server it is compared
    against has stopped."""

    def __init__(self, transformers, snap: Path):
        from mstar.model.qwen3.qwen3_model import Qwen3Model

        self.transformers, self.snap = transformers, snap
        model = Qwen3Model(model_path_hf=str(snap))
        self.prompts = [
            model.process_prompt(p, ["text"], ["text"], enable_thinking=False)
            .new_input_tensors["text_inputs"][0].tolist()
            for _, p in ALL_PROMPTS
        ]
        # encode_ids over the rendered text gives the ids apply_chat_template's own tokenize does
        for (_, p), ids in zip(ALL_PROMPTS, self.prompts, strict=True):
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


class _Errors(NamedTuple):
    """One prompt's per-position errors against fp32, at the served tokens."""
    served: torch.Tensor  # e_srv
    bf16: torch.Tensor  # e_bf
    served_flips: int  # served token != fp32's argmax
    bf16_flips: int  # bf16's argmax != fp32's
    deficit: float  # fp32's margin given up at a served flip, the largest


def _errors(ref: _Reference, served: list[_Served]) -> list[_Errors]:
    replies = [s.ids for s in served]
    fp32 = ref.log_probs(torch.float32, replies)
    bf16 = ref.log_probs(torch.bfloat16, replies)
    out = []
    for f, b, s in zip(fp32, bf16, served, strict=True):
        rows, tokens = torch.arange(len(s.ids)), torch.tensor(s.ids, dtype=torch.long)
        f_s, top = f[rows, tokens], f.argmax(-1)
        out.append(_Errors(
            served=(torch.tensor(s.logprobs, dtype=torch.float32) - f_s).abs(),
            bf16=(b[rows, tokens] - f_s).abs(),
            served_flips=int((tokens != top).sum()),
            bf16_flips=int((b.argmax(-1) != top).sum()),
            deficit=float((f[rows, top] - f_s).max()) if len(s.ids) else 0.0,
        ))
    return out


def _trimmed_mean(x: torch.Tensor) -> float:
    return float(x[x <= x.quantile(TRIM)].mean())


class _Ratios(NamedTuple):
    trimmed: float  # R, the gate's
    mean: float
    p90: float


def _ratios(errors: list[_Errors]) -> _Ratios:
    srv, bf = torch.cat([e.served for e in errors]), torch.cat([e.bf16 for e in errors])
    assert bf.numel() and float(bf.mean()) > 0, "no positions to compare"
    return _Ratios(
        _trimmed_mean(srv) / _trimmed_mean(bf), float(srv.mean() / bf.mean()),
        float(srv.quantile(0.9) / bf.quantile(0.9)),
    )


def _report(label: str, errors: list[_Errors]) -> dict[str, float]:
    """Print what the calibration reads (the PR quotes it) and return each set's R."""
    def line(group: list[_Errors]) -> str:
        r = _ratios(group)
        srv, bf = torch.cat([e.served for e in group]), torch.cat([e.bf16 for e in group])
        return (f"R {r.trimmed:.3f} (mean ratio {r.mean:.3f}, p90 ratio {r.p90:.3f}); e_srv mean "
                f"{float(srv.mean()):.4f} e_bf mean {float(bf.mean()):.4f}; greedy flips served "
                f"{sum(e.served_flips for e in group)} bf16 {sum(e.bf16_flips for e in group)}; "
                f"largest deficit {max(e.deficit for e in group):.4f}")

    for i, ((name, _), e) in enumerate(zip(ALL_PROMPTS, errors, strict=True)):
        print(f"{label}: {name} prompt {i} ({len(e.served)} tokens): {line([e])}")
    ratios = {}
    for name in SETS:
        group = [e for (n, _), e in zip(ALL_PROMPTS, errors, strict=True) if n == name]
        ratios[name] = _ratios(group).trimmed
        print(f"{label}: {name} set: {line(group)}")
    return ratios


CASES = [("qwen3_8b", "Qwen/Qwen3-8B")]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _served_ids(tmp_path, name: str, model_path: Path | None = None) -> list[_Served]:
    # max_seq_len is required and unread; the checkpoint's context bounds a request
    config = {"model": name, "max_seq_len": 4096,
              "resources": {"kv_cache": {"max_num_pages": 1024}, "sampler": {"enable_logprobs": True}},
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
        return [_ids(url, p) for _, p in ALL_PROMPTS]


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
    """The rendered prompts and transformers, once for the module."""
    transformers = pytest.importorskip("transformers")
    return _Reference(transformers, _snapshot_or_skip(CASES[0][1]))


@pytest.mark.parametrize("name, repo", CASES, ids=[c[0] for c in CASES])
def test_served_log_probs_err_from_fp32_as_transformers_bf16_does(tmp_path, name, repo, reference):
    ratios = _report(name, _errors(reference, _served_ids(tmp_path, name)))
    past = {k: round(r, 3) for k, r in ratios.items() if r > R_MAX}
    assert not past, f"{name}: served log-probs err from fp32 past {R_MAX}x transformers' bf16 on {past}"


@pytest.mark.parametrize("control, edit", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_a_port_bug_fails_the_check(tmp_path, control, edit, reference):
    """The negative control: the real model's gate fails it on the long set."""
    name, repo = CASES[0]
    served = _served_ids(tmp_path, name, model_path=_edited(tmp_path, _snapshot_or_skip(repo), edit))
    ratios = _report(f"{name} ({control})", _errors(reference, served))
    assert ratios["long"] > R_MAX, f"{control} passes the gate: R {ratios['long']:.3f} on the long set"


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
