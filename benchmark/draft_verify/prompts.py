"""Prompt sets for draft-verify (speculative decoding) benchmarks, written as JSONL.

Standard sets, the ones speculative decoding is usually reported on. Each is pinned to a dataset
revision and the sha256 of the file read (``SOURCES``), fetched into the Hugging Face cache
(Spec-Bench, which is on GitHub, into ``--cache``), and turned into chat prompts the same way
every time:

- ``mt_bench``: MT-Bench's 80 questions (``HuggingFaceH4/mt_bench_prompts``), the first turn.
- ``humaneval``: HumanEval's 164 problems (``openai/openai_humaneval``), in EvalPlus's
  instruction for chat models.
- ``gsm8k``: GSM8K's test questions (``openai/gsm8k``, ``main``), the first ``--count``.
- ``sharegpt``: the first human turn of ShareGPT V3 conversations
  (``ShareGPT_V3_unfiltered_cleaned_split.json``), a seeded sample of ``--count``.
- ``specbench_<subset>``: Spec-Bench's six subsets of 80 (``mt_bench``, ``translation``,
  ``summarization``, ``qa``, ``math_reasoning``, ``rag``), the first turn.

Ours, kept as secondary checks:

- ``code_edit``: a chunk of one of this repo's Python files and an edit to make, answered with
  the whole edited chunk. The reply copies most of its prompt, which flatters drafting
  (prompt lookup above all).
- ``chat``: ``benchmark/assets/simple_text_queries.txt``, open questions.

Deterministic for a given tree.

    python -m benchmark.draft_verify.prompts mt_bench out/mt_bench.jsonl
    python -m benchmark.draft_verify.prompts code_edit out/code_edit.jsonl --count 100
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]


class Source(NamedTuple):
    """A pinned file: a Hugging Face dataset's (``host="hf"``) or a GitHub repo's."""
    host: str
    repo: str
    revision: str
    path: str
    sha256: str


SOURCES = {
    "mt_bench": Source(
        "hf", "HuggingFaceH4/mt_bench_prompts", "e3a795c5e9a82ee40611c416b8a7786c73198991",
        "raw/question.jsonl",
        "420ae1d63cbf0f499d8afc3b28d86714e77f13e44c133eedf942ed16c544dd34",
    ),
    "humaneval": Source(
        "hf", "openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "openai_humaneval/test-00000-of-00001.parquet",
        "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
    ),
    "gsm8k": Source(
        "hf", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
        "main/test-00000-of-00001.parquet",
        "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    ),
    "sharegpt": Source(
        "hf", "anon8231489123/ShareGPT_Vicuna_unfiltered", "192ab2185289094fc556ec8ce5ce1e8e587154ca",
        "ShareGPT_V3_unfiltered_cleaned_split.json",
        "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4",
    ),
    "specbench": Source(
        "github", "hemingkx/Spec-Bench", "66230f10cb0a02aced5ef3ce1e85163c16160454",
        "data/spec_bench/question.jsonl",
        "4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04",
    ),
}
SPECBENCH_SUBSETS = ("mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag")
STANDARD = ("mt_bench", "humaneval", "gsm8k", "sharegpt", *(f"specbench_{s}" for s in SPECBENCH_SUBSETS))

# EvalPlus's instruction for chat models (evalplus/provider/utility.py, make_raw_chat_prompt)
_HUMANEVAL = (
    "Please provide a self-contained Python script that solves the following problem in a "
    "markdown code block:\n```\n{prompt}\n```"
)


def fetch(source: Source, cache: Path | None = None) -> Path:
    """The pinned file, downloaded once; refuses one whose hash isn't the pinned one."""
    if source.host == "hf":
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(source.repo, source.path, repo_type="dataset", revision=source.revision))
    else:
        cache = cache or Path.home() / ".cache/mstar/prompts"
        path = cache / source.repo.replace("/", "--") / source.revision / source.path
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{source.repo}/{source.revision}/{source.path}"
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as resp:
                tmp.write_bytes(resp.read())
            tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source.sha256:
        raise ValueError(f"{source.repo}@{source.revision}:{source.path} hashes {digest}, not {source.sha256}")
    return path


def _parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mt_bench(count: int | None = None, cache: Path | None = None) -> list[dict]:
    rows = _jsonl(fetch(SOURCES["mt_bench"], cache))
    return [
        {"id": f"mt_bench/{row['question_id']}", "category": row["category"], "prompt": row["prompt"][0],
         "max_tokens": 1024}
        for row in rows[:count]
    ]


def humaneval(count: int | None = None, cache: Path | None = None) -> list[dict]:
    rows = _parquet(fetch(SOURCES["humaneval"], cache))
    return [
        {"id": row["task_id"], "prompt": _HUMANEVAL.format(prompt=row["prompt"].strip()), "max_tokens": 512}
        for row in rows[:count]
    ]


def gsm8k(count: int | None = 256, cache: Path | None = None) -> list[dict]:
    rows = _parquet(fetch(SOURCES["gsm8k"], cache))
    return [{"id": f"gsm8k/{i}", "prompt": row["question"], "max_tokens": 512} for i, row in enumerate(rows[:count])]


def sharegpt(count: int | None = 256, seed: int = 0, cache: Path | None = None) -> list[dict]:
    """First human turns of conversations that a model answered, 16 to 4,000 characters (about
    1,000 tokens at most), a seeded sample in the file's order."""
    conversations = json.loads(fetch(SOURCES["sharegpt"], cache).read_text())
    firsts = []
    for conv in conversations:
        turns = conv.get("conversations") or []
        if len(turns) < 2 or turns[0].get("from") != "human" or turns[1].get("from") != "gpt":
            continue
        text = (turns[0].get("value") or "").strip()
        if 16 <= len(text) <= 4000:
            firsts.append((conv.get("id", str(len(firsts))), text))
    picked = sorted(random.Random(seed).sample(range(len(firsts)), min(count or len(firsts), len(firsts))))
    return [{"id": f"sharegpt/{firsts[i][0]}", "prompt": firsts[i][1], "max_tokens": 1024} for i in picked]


def specbench(subset: str, count: int | None = None, cache: Path | None = None) -> list[dict]:
    if subset not in SPECBENCH_SUBSETS:
        raise ValueError(f"Spec-Bench has no subset {subset!r}; its subsets are {SPECBENCH_SUBSETS}")
    rows = [row for row in _jsonl(fetch(SOURCES["specbench"], cache)) if _specbench_subset(row) == subset]
    return [
        {"id": f"specbench/{subset}/{row['question_id']}", "prompt": row["turns"][0], "max_tokens": 1024}
        for row in rows[:count]
    ]


def _specbench_subset(row: dict) -> str:
    # MT-Bench's own categories (writing, roleplay, ...) make up Spec-Bench's mt_bench subset
    category = row["category"]
    return category if category in SPECBENCH_SUBSETS else "mt_bench"


_EDITS = [
    "Rename `{name}` to `{name}_v2` everywhere it appears.",
    "Add a short comment above every `for` and `while` loop saying what it iterates over.",
    "Add type hints to every function parameter and return value that lacks one.",
    "Replace every double-quoted string literal with a single-quoted one, unless that needs escaping.",
    "Remove every comment and docstring.",
    "Add a one-line docstring to every function that has none.",
    "Rename every local variable called `{local}` to `{local}_value`.",
    "Wrap every line longer than 80 characters.",
]

_INSTRUCTION = (
    "Here is a chunk of a Python file:\n\n```python\n{code}\n```\n\n{edit} "
    "Keep everything else exactly as it is. Reply with the complete updated chunk in one "
    "```python block and nothing else."
)


def _chunks(path: Path, rng: random.Random, min_lines: int, max_lines: int) -> list[str]:
    """Runs of whole top-level statements, ``min_lines`` to ``max_lines`` long."""
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    starts = [node.lineno - 1 for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    starts = [s for s in starts if s < len(lines)]
    out = []
    for i, start in enumerate(starts):
        end = start
        for nxt in starts[i + 1:] + [len(lines)]:
            if nxt - start > max_lines:
                break
            end = nxt
        if end - start >= min_lines:
            out.append("\n".join(lines[start:end]).rstrip())
    rng.shuffle(out)
    return out


def _edit(code: str, rng: random.Random) -> str:
    names = sorted({m for m in re.findall(r"\bdef (\w+)", code)} | {m for m in re.findall(r"\bclass (\w+)", code)})
    locals_ = sorted({m for m in re.findall(r"^\s+(\w+) = ", code, re.M) if m not in ("self", "_")})
    template = rng.choice(_EDITS)
    if "{name}" in template and not names:
        template = _EDITS[1]
    if "{local}" in template and not locals_:
        template = _EDITS[2]
    return template.format(name=rng.choice(names) if names else "", local=rng.choice(locals_) if locals_ else "")


def code_edit(count: int, seed: int = 0, min_lines: int = 40, max_lines: int = 90) -> list[dict]:
    rng = random.Random(seed)
    files = sorted(p for p in (ROOT / "mstar").rglob("*.py") if "tests" not in p.parts)
    rng.shuffle(files)
    out = []
    for path in files:
        chunks = _chunks(path, rng, min_lines, max_lines)
        if not chunks:
            continue
        code = chunks[0]
        out.append({
            "id": f"code_edit/{len(out)}",
            "source": str(path.relative_to(ROOT)),
            "prompt": _INSTRUCTION.format(code=code, edit=_edit(code, rng)),
            "max_tokens": 2048,
        })
        if len(out) == count:
            break
    return out


def chat(count: int | None = None) -> list[dict]:
    lines = [line.strip() for line in (ROOT / "benchmark/assets/simple_text_queries.txt").read_text().splitlines()]
    lines = [line for line in lines if line]
    return [{"id": f"chat/{i}", "prompt": line, "max_tokens": 512} for i, line in enumerate(lines[:count])]


def build(name: str, count: int | None = None, seed: int = 0, cache: Path | None = None) -> list[dict]:
    """The named set's prompts; ``count`` None takes a set's default size."""
    if name.startswith("specbench_"):
        return specbench(name.removeprefix("specbench_"), count, cache)
    if name == "mt_bench":
        return mt_bench(count, cache)
    if name == "humaneval":
        return humaneval(count, cache)
    if name == "gsm8k":
        return gsm8k(256 if count is None else count, cache)
    if name == "sharegpt":
        return sharegpt(256 if count is None else count, seed, cache)
    if name == "code_edit":
        return code_edit(100 if count is None else count, seed)
    if name == "chat":
        return chat(count)
    raise ValueError(f"no prompt set {name!r}; the sets are {(*STANDARD, 'code_edit', 'chat')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("set", choices=[*STANDARD, "code_edit", "chat"])
    parser.add_argument("out")
    parser.add_argument("--count", type=int, default=None, help="default: the set's own size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=None, help="where GitHub-hosted sets are kept")
    args = parser.parse_args()
    rows = build(args.set, args.count, args.seed, args.cache)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
