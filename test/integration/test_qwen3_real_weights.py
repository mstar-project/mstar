"""Plain Qwen3 served from real weights against transformers' Qwen3ForCausalLM (GPU).

Qwen3-8B, the registered size, serves (its head is untied). Each prompt's greedy reply, as token ids from the server, is
compared with transformers' greedy generation on the same prompt ids by the
margin rule: where they first differ, transformers' top-2 log-prob margin there
must be at most 0.25 (a bf16 near-tie two implementations may break apart).

Qwen3-0.6B ties its head to the embedding (its file also ships the head, equal to
the embedding); it is not a registered size, so the loader's tie path is checked
on the model class itself, with the file's head left out of the stream.

Skips unless the checkpoint is fully cached, CUDA and transformers are present.

    pytest test/integration/test_qwen3_real_weights.py -q
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import requests
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
E2E = REPO / "test" / "e2e"
MARGIN = 0.25

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


class _Margins:
    """``margins(prompt_index, ids)``: plain Qwen3-8B's top-2 log-prob margin for
    the token after ``ids`` of that prompt's reply, from transformers in float32 (a bf16 reference
    disagrees with itself by more than the margin rule allows).
    Loaded only once the server it is compared against has stopped."""

    def __init__(self, transformers, repo: str):
        from mstar.model.qwen3.qwen3_model import Qwen3Model

        self.transformers, self.snap, self.lm = transformers, _snapshot(repo), None
        model = Qwen3Model(model_path_hf=str(self.snap))
        self.stop_ids = model.stop_token_ids
        self.prompts = [
            model.process_prompt(p, ["text"], ["text"], enable_thinking=False)
            .new_input_tensors["text_inputs"][0].tolist()
            for p in PROMPTS
        ]

    def load(self):
        if self.lm is None:
            self.lm = self.transformers.AutoModelForCausalLM.from_pretrained(
                self.snap, dtype=torch.float32).cuda().eval()
        return self.lm

    def __call__(self, i: int, ids: list[int]) -> float:
        self.load()
        with torch.no_grad():
            seq = torch.tensor([self.prompts[i] + ids], device="cuda")
            top = self.lm(seq).logits[0, -1].float().log_softmax(-1).topk(2).values
        return float(top[0] - top[1])

    def release(self) -> None:
        self.lm = None
        torch.cuda.empty_cache()


CASES = [("qwen3_8b", "Qwen/Qwen3-8B")]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _served_ids(tmp_path, name: str) -> list[list[int]]:
    config = {"model": name, "max_seq_len": 4096, "model_kwargs": {"max_seq_len": 4096},
              "node_groups": [{"node_names": ["LLM"], "ranks": [0]}]}
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


@pytest.mark.parametrize("name, repo", CASES, ids=[c[0] for c in CASES])
def test_plain_greedy_tokens_are_transformers_greedy_tokens(tmp_path, name, repo):
    if _snapshot(repo) is None:
        pytest.skip(f"{repo} is not fully cached")
    margins = _Margins(pytest.importorskip("transformers"), repo)
    served = _served_ids(tmp_path, name)
    lm = margins.load()
    try:
        for i, ids in enumerate(served):
            prompt = torch.tensor([margins.prompts[i]], device="cuda")
            with torch.no_grad():
                ref = lm.generate(prompt, max_new_tokens=TOKENS, do_sample=False)[0, prompt.shape[1]:].tolist()
            # generate keeps the closing stop token, which the text stream may not
            while ref and ref[-1] in margins.stop_ids:
                ref.pop()
            while ids and ids[-1] in margins.stop_ids:
                ids = ids[:-1]
            n = min(len(ref), len(ids))
            at = next((j for j in range(n) if ref[j] != ids[j]), None)
            if at is None and len(ref) != len(ids):
                at = n
            if at is not None:
                m = margins(i, ref[:at])
                assert m <= MARGIN, f"{name}: prompt {i} first differs at token {at}, margin {m:.3f}"
    finally:
        margins.release()


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
