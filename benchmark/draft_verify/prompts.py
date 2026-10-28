"""Prompt sets for draft-verify (speculative decoding) benchmarks, written as JSONL.

``code_edit``: a chunk of one of this repo's Python files and an edit to make, answered with
the whole edited chunk. The reply copies most of its prompt, which is where prompt-lookup
drafting pays: the workload of coding assistants applying an edit, a rename or a refactor.

``chat``: ``benchmark/assets/simple_text_queries.txt``, open questions whose answers repeat
little of anything.

Deterministic for a given tree: files and chunks are picked by a seeded generator over the
sorted file list.

    python -m benchmark.draft_verify.prompts code_edit out/code_edit.jsonl --count 100
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("set", choices=["code_edit", "chat"])
    parser.add_argument("out")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = code_edit(args.count, args.seed) if args.set == "code_edit" else chat(args.count)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
