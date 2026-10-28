"""Tool-call accuracy of a chat model behind an OpenAI- or Anthropic-compatible route.

The same requests go to any server (M*, vLLM, SGLang), greedy. Two sets:

* ``bfcl``: single-turn cases sampled (``--per-category``, seeded) from the Berkeley
  Function-Calling Leaderboard's v4 data (``--bfcl-dir``: ``BFCL_v4_<category>.json`` and
  ``possible_answer/``, from github.com/ShishirPatil/gorilla), checked as its AST checker
  does: the function's name, every value among the possible answers (strings compared
  without case, spaces or punctuation), every required argument present, parallel calls in
  any order; an ``irrelevance`` case passes when no call is made.
* ``agent``: multi-turn coding-agent tasks over a small in-memory repository (read, edit,
  list, grep, run the tests); a task passes when the files and the final answer are right
  within ``--max-turns`` model turns.

Each request runs whole and streamed (``--stream both``); the two must agree, call for call.

    python -m benchmark.text_lm.tool_eval --url http://127.0.0.1:8000 --model qwen3_coder_30b_a3b \\
        --bfcl-dir data/bfcl --sets bfcl agent --api chat --out run.json
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from pathlib import Path

import requests

BFCL_CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance")
# BFCL's type names onto JSON schema's (its GORILLA_TO_OPENAPI)
_TYPES = {"integer": "integer", "number": "number", "float": "number", "string": "string", "boolean": "boolean",
          "bool": "boolean", "array": "array", "list": "array", "dict": "object", "object": "object",
          "tuple": "array", "any": "string"}


# --------------------------------------------------------------------------
# requests
# --------------------------------------------------------------------------

class Reply:
    """A reply as both APIs give it: text, calls as (name, arguments object or the raw string), why it ended."""

    def __init__(self, content="", reasoning="", calls=None, finish=None, error=None, seconds=0.0):
        self.content, self.reasoning, self.calls = content, reasoning, calls or []
        self.finish, self.error, self.seconds = finish, error, seconds

    def key(self):
        return self.content.strip(), [(n, a) for n, a in self.calls], self.finish

    def as_dict(self):
        return {"content": self.content, "reasoning": self.reasoning, "calls": self.calls, "finish": self.finish,
                "error": self.error, "seconds": round(self.seconds, 3)}


def _args(raw: str):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def chat(url, model, messages, tools, *, stream, tool_choice="auto", max_tokens=1024, extra=None) -> Reply:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": stream, **(extra or {})}
    if tools:
        body["tools"], body["tool_choice"] = tools, tool_choice
    started = time.perf_counter()
    try:
        with requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, stream=stream, timeout=600) as r:
            if r.status_code != 200:
                return Reply(error=f"{r.status_code}: {r.text[:300]}", seconds=time.perf_counter() - started)
            if not stream:
                choice = r.json()["choices"][0]
                msg = choice["message"]
                calls = [(c["function"]["name"], _args(c["function"]["arguments"]))
                         for c in msg.get("tool_calls") or []]
                return Reply(msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning") or "",
                             calls, choice["finish_reason"], seconds=time.perf_counter() - started)
            content, reasoning, calls, finish = [], [], {}, None
            for line in r.iter_lines():
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if "error" in chunk:
                    return Reply(error=str(chunk["error"]), seconds=time.perf_counter() - started)
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                    for c in delta.get("tool_calls") or []:
                        call = calls.setdefault(c["index"], {"name": "", "arguments": ""})
                        fn = c.get("function") or {}
                        call["name"] += fn.get("name") or ""
                        call["arguments"] += fn.get("arguments") or ""
                    finish = choice.get("finish_reason") or finish
            return Reply("".join(content), "".join(reasoning),
                         [(c["name"], _args(c["arguments"])) for _, c in sorted(calls.items())], finish,
                         seconds=time.perf_counter() - started)
    except requests.RequestException as exc:
        return Reply(error=repr(exc), seconds=time.perf_counter() - started)


def _object(raw: str) -> dict:
    value = _args(raw)
    return value if isinstance(value, dict) else {}


def _to_anthropic(messages, tools):
    """An OpenAI chat as an Anthropic Messages body's system, messages and tools."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    out = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                        "input": _object(c["function"]["arguments"])} for c in m["tool_calls"]]
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    anthropic_tools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                        "input_schema": t["function"]["parameters"]} for t in tools or []]
    return system, out, anthropic_tools


def messages_api(url, model, messages, tools, *, stream, tool_choice="auto", max_tokens=1024, extra=None) -> Reply:
    system, msgs, atools = _to_anthropic(messages, tools)
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.0, "stream": stream,
            **(extra or {})}
    if system:
        body["system"] = system
    if atools:
        body["tools"] = atools
        body["tool_choice"] = {"type": {"auto": "auto", "required": "any"}[tool_choice]}
    started = time.perf_counter()
    try:
        with requests.post(url.rstrip("/") + "/v1/messages", json=body, stream=stream, timeout=600,
                           headers={"anthropic-version": "2023-06-01"}) as r:
            if r.status_code != 200:
                return Reply(error=f"{r.status_code}: {r.text[:300]}", seconds=time.perf_counter() - started)
            blocks, finish = [], None
            if not stream:
                msg = r.json()
                blocks, finish = msg["content"], msg["stop_reason"]
            else:
                for line in r.iter_lines():
                    if not line.startswith(b"data:"):
                        continue
                    event = json.loads(line[5:])
                    if event["type"] == "content_block_start":
                        blocks.append(dict(event["content_block"], partial=""))
                    elif event["type"] == "content_block_delta":
                        d = event["delta"]
                        block = blocks[-1]
                        if d["type"] == "text_delta":
                            block["text"] = block.get("text", "") + d["text"]
                        elif d["type"] == "thinking_delta":
                            block["thinking"] = block.get("thinking", "") + d["thinking"]
                        elif d["type"] == "input_json_delta":
                            block["partial"] += d["partial_json"]
                    elif event["type"] == "message_delta":
                        finish = event["delta"]["stop_reason"]
                    elif event["type"] == "error":
                        return Reply(error=str(event["error"]), seconds=time.perf_counter() - started)
                for block in blocks:
                    if block["type"] == "tool_use":
                        block["input"] = _args(block.pop("partial") or "{}")
            text = "".join(b.get("text", "") for b in blocks if b["type"] == "text")
            thinking = "".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking")
            calls = [(b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"]
            finish = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length"}.get(finish, finish)
            return Reply(text, thinking, calls, finish, seconds=time.perf_counter() - started)
    except requests.RequestException as exc:
        return Reply(error=repr(exc), seconds=time.perf_counter() - started)


# --------------------------------------------------------------------------
# BFCL single-turn cases
# --------------------------------------------------------------------------

def _schema(param: dict) -> dict:
    out = copy.deepcopy(param)
    out["type"] = _TYPES.get(out.get("type", "string"), "string")
    if param.get("type") == "float":
        out["description"] = out.get("description", "") + " This is a float type value."
    if "properties" in out:
        out["properties"] = {k: _schema(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = _schema(out["items"])
    return out


def _tool(fn: dict) -> dict:
    """A BFCL function as an OpenAI tool; dots in names become underscores, as its OpenAI handlers send them."""
    return {"type": "function", "function": {"name": fn["name"].replace(".", "_"),
                                             "description": fn.get("description", ""),
                                             "parameters": _schema(fn["parameters"])}}


def load_bfcl(root: Path, per_category: dict[str, int], seed: int) -> list[dict]:
    cases = []
    for category, n in per_category.items():
        rows = [json.loads(line) for line in (root / f"BFCL_v4_{category}.json").read_text().splitlines()]
        answers = {}
        if category != "irrelevance":  # passes when no call is made: it has no possible answers
            answer_file = root / "possible_answer" / f"BFCL_v4_{category}.json"
            answers = {a["id"]: a["ground_truth"] for a in map(json.loads, answer_file.read_text().splitlines())}
        for row in random.Random(f"{seed}-{category}").sample(rows, n):
            cases.append({"id": row["id"], "category": category, "messages": row["question"][0],
                          "functions": row["function"], "tools": [_tool(f) for f in row["function"]],
                          "answer": answers.get(row["id"])})
    return cases


def _match(value, want) -> bool:
    """``value`` against one possible answer: strings without case, spaces or punctuation, an
    integer where a float is wanted, lists element by element, and a dict key by key over the
    possible answers each key lists (BFCL's nested form)."""
    if isinstance(want, dict):
        return isinstance(value, dict) and set(value) <= set(want) and all(
            _value_ok(value[k], allowed) if k in value else "" in allowed for k, allowed in want.items())
    if isinstance(want, (list, tuple)):
        return isinstance(value, (list, tuple)) and len(value) == len(want) and all(
            map(_match, value, want))
    if isinstance(want, bool) or isinstance(value, bool):
        return type(value) is type(want) and value == want
    if isinstance(want, (int, float)):
        return isinstance(value, (int, float)) and float(value) == float(want)
    if isinstance(want, str):
        return isinstance(value, str) and _standard(value) == _standard(want)
    return value == want


def _standard(text: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", text).lower().replace("'", '"')


def _value_ok(value, allowed: list) -> bool:
    return any(_match(value, a) for a in allowed if a != "")


def _call_ok(name, args, gt: dict, functions: dict) -> str | None:
    """None if (name, args) is the ground-truth call ``gt`` ({name: {param: [allowed]}}), else why not."""
    ((gt_name, gt_args),) = gt.items()
    if name != gt_name.replace(".", "_"):
        return f"name {name!r} != {gt_name!r}"
    if not isinstance(args, dict):
        return f"arguments are not an object: {args!r}"
    fn = functions[gt_name]
    for req in fn["parameters"].get("required", []):
        if req not in args:
            return f"missing required {req!r}"
    for key, value in args.items():
        if key not in gt_args:
            return f"unexpected argument {key!r}"
        if not _value_ok(value, gt_args[key]):
            return f"{key}={value!r} not in {gt_args[key]!r}"
    for key, allowed in gt_args.items():
        if key not in args and "" not in allowed:
            return f"missing {key!r}"
    return None


def check_bfcl(case: dict, reply: Reply) -> str | None:
    """None if the reply passes, else why it fails."""
    if reply.error:
        return f"error {reply.error}"
    if case["category"] == "irrelevance":
        return None if not reply.calls else f"called {[n for n, _ in reply.calls]}"
    functions = {f["name"]: f for f in case["functions"]}
    gt = case["answer"]
    if len(reply.calls) != len(gt):
        return f"{len(reply.calls)} calls, want {len(gt)}"
    unmatched = list(range(len(gt)))
    for name, args in reply.calls:
        for i in unmatched:
            if _call_ok(name, args, gt[i], functions) is None:
                unmatched.remove(i)
                break
        else:
            return _call_ok(name, args, gt[unmatched[0]], functions) or "no ground truth matches"
    return None


# --------------------------------------------------------------------------
# agent tasks: a small repository in memory
# --------------------------------------------------------------------------

AGENT_SYSTEM = ("You are a coding agent working in a repository. Use the tools to inspect and change files; "
                "do not guess file contents. When the task is done, reply with a short final answer.")
AGENT_TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List the entries of a directory.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory."}},
                    "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file's text.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit_file",
     "description": "Replace the one occurrence of old_text in a file with new_text.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                                     "new_text": {"type": "string"}},
                    "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file, replacing its text.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Lines matching a regular expression, by file.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the test suite.",
     "parameters": {"type": "object", "properties": {}}}},
]

REPO = {
    "README.md": "# shop\nA tiny web shop.\n",
    "config/server.yaml": "host: 0.0.0.0\nport: 8443\nworkers: 4\n",
    "src/app.py": ("from src.utils import slugify\n\n\ndef product_url(name):\n"
                   "    return '/p/' + slugify(name)\n"),
    "src/utils.py": ("import re\n\n\ndef slugify(text):\n"
                     "    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"),
    "src/mathx.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n",
    "src/limits.py": ("class TokenBucket:\n    def __init__(self, rate, burst):\n"
                      "        self.rate, self.burst = rate, burst\n"),
    "src/data.json": '{"items": []}\n',
    "tests/test_mathx.py": ("from src.mathx import add, mul\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
                            "def test_mul():\n    assert mul(2, 3) == 6\n"),
}


class Repo:
    def __init__(self):
        self.files = dict(REPO)

    def call(self, name, args) -> str:
        if not isinstance(args, dict):
            return "error: arguments must be a JSON object"
        try:
            if name == "list_dir":
                prefix = args["path"].strip("/").removeprefix("./")
                prefix = "" if prefix in ("", ".") else prefix + "/"
                names = sorted({p[len(prefix):].split("/")[0] + ("/" if "/" in p[len(prefix):] else "")
                                for p in self.files if p.startswith(prefix)})
                return "\n".join(names) or "error: no such directory"
            if name == "read_file":
                return self.files.get(args["path"].removeprefix("./"), "error: no such file")
            if name == "write_file":
                self.files[args["path"].removeprefix("./")] = args["content"]
                return "ok"
            if name == "edit_file":
                path = args["path"].removeprefix("./")
                text = self.files.get(path)
                if text is None:
                    return "error: no such file"
                if text.count(args["old_text"]) != 1:
                    return f"error: old_text occurs {text.count(args['old_text'])} times"
                self.files[path] = text.replace(args["old_text"], args["new_text"])
                return "ok"
            if name == "grep":
                pattern = re.compile(args["pattern"])
                hits = [f"{p}:{i + 1}:{line}" for p, t in sorted(self.files.items())
                        for i, line in enumerate(t.splitlines()) if pattern.search(line)]
                return "\n".join(hits) or "no matches"
            if name == "run_tests":
                ok = "return a + b" in self.files["src/mathx.py"]
                return ("2 passed" if ok else
                        "FAILED tests/test_mathx.py::test_add - assert -1 == 5\n 1 failed, 1 passed")
        except (KeyError, re.error) as exc:
            return f"error: {exc!r}"
        return f"error: no tool {name!r}"


AGENT_TASKS = [
    {"id": "port", "prompt": "What port does the server listen on? The setting is in config/server.yaml.",
     "check": lambda repo, answer: "8443" in answer},
    {"id": "rename", "prompt": ("In src/utils.py rename the function `slugify` to `make_slug`, and update its "
                                "call sites (src/app.py uses it)."),
     "check": lambda repo, answer: all("make_slug" in repo.files[p] and "slugify" not in repo.files[p]
                                       for p in ("src/utils.py", "src/app.py"))},
    {"id": "count", "prompt": "How many Python files are directly in src/? Answer with the number.",
     "check": lambda repo, answer: re.search(r"\b(4|four)\b", answer.lower()) is not None},
    {"id": "fix", "prompt": "The test suite fails. Run the tests, find the bug, fix it, and run the tests again.",
     "check": lambda repo, answer: "return a + b" in repo.files["src/mathx.py"]
     and "return a * b" in repo.files["src/mathx.py"]},
    {"id": "find", "prompt": "Which file defines the class TokenBucket? Answer with its path.",
     "check": lambda repo, answer: "src/limits.py" in answer},
]


def run_agent(send, task, max_turns) -> dict:
    """The task's loop: the model calls tools until it answers; every turn is one request."""
    repo, messages = Repo(), [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": task["prompt"]}]
    turns, transcript = 0, []
    for turns in range(1, max_turns + 1):
        reply = send(messages, AGENT_TOOLS)
        transcript.append(reply.as_dict())
        if reply.error:
            return {"ok": False, "why": f"error {reply.error}", "turns": turns, "transcript": transcript}
        if not reply.calls:
            ok = task["check"](repo, reply.content)
            return {"ok": ok, "why": None if ok else "wrong final state or answer", "turns": turns,
                    "transcript": transcript}
        calls = [{"id": f"call_{turns}_{i}", "type": "function",
                  "function": {"name": n, "arguments": json.dumps(a) if isinstance(a, dict) else str(a)}}
                 for i, (n, a) in enumerate(reply.calls)]
        messages.append({"role": "assistant", "content": reply.content or None, "tool_calls": calls})
        for call, (name, args) in zip(calls, reply.calls, strict=True):
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": repo.call(name, args)})
    return {"ok": False, "why": "out of turns", "turns": turns, "transcript": transcript}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api", choices=("chat", "messages"), default="chat")
    ap.add_argument("--sets", nargs="+", choices=("bfcl", "agent"), default=["bfcl", "agent"])
    ap.add_argument("--bfcl-dir", type=Path)
    ap.add_argument("--per-category", default="simple_python=15,multiple=10,parallel=10,parallel_multiple=5,"
                                               "irrelevance=5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tool-choice", choices=("auto", "required"), default="auto")
    ap.add_argument("--stream", choices=("off", "on", "both"), default="both")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request body")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    extra = json.loads(args.extra)
    request = chat if args.api == "chat" else messages_api
    streams = {"off": [False], "on": [True], "both": [False, True]}[args.stream]

    def send(stream, tool_choice="auto"):
        return lambda messages, tools: request(args.url, args.model, messages, tools, stream=stream,
                                               tool_choice=tool_choice, max_tokens=args.max_tokens, extra=extra)

    out = {"model": args.model, "url": args.url, "api": args.api, "tool_choice": args.tool_choice,
           "seed": args.seed, "bfcl": [], "agent": []}
    if "bfcl" in args.sets:
        per = {k: int(v) for k, v in (kv.split("=") for kv in args.per_category.split(","))}
        for case in load_bfcl(args.bfcl_dir, per, args.seed):
            choice = "auto" if case["category"] == "irrelevance" else args.tool_choice
            row = {"id": case["id"], "category": case["category"]}
            for stream in streams:
                reply = send(stream, choice)(case["messages"], case["tools"])
                row["stream" if stream else "whole"] = {**reply.as_dict(), "why": check_bfcl(case, reply)}
            if len(streams) == 2:
                row["same"] = Reply(**{k: v for k, v in row["whole"].items() if k not in ("why", "seconds")}).key() \
                    == Reply(**{k: v for k, v in row["stream"].items() if k not in ("why", "seconds")}).key()
            out["bfcl"].append(row)
            print(case["id"], {k: v["why"] for k, v in row.items() if isinstance(v, dict)}, row.get("same"))
    if "agent" in args.sets:
        for task in AGENT_TASKS:
            row = {"id": task["id"]}
            for stream in streams:
                row["stream" if stream else "whole"] = run_agent(send(stream), task, args.max_turns)
            if len(streams) == 2:
                row["same"] = [(t["calls"], t["content"].strip()) for t in row["whole"]["transcript"]] == \
                    [(t["calls"], t["content"].strip()) for t in row["stream"]["transcript"]]
            out["agent"].append(row)
            print(task["id"], {k: (v["ok"], v["turns"], v["why"]) for k, v in row.items() if isinstance(v, dict)},
                  row.get("same"))
    summary = {}
    for name in ("bfcl", "agent"):
        for stream in ("whole", "stream"):
            rows = [r for r in out[name] if stream in r]
            if not rows:
                continue
            ok = [r for r in rows if (r[stream]["why"] is None if name == "bfcl" else r[stream]["ok"])]
            summary[f"{name}/{stream}"] = f"{len(ok)}/{len(rows)}"
            if name == "bfcl":
                for category in BFCL_CATEGORIES:
                    cat = [r for r in rows if r["category"] == category]
                    if cat:
                        summary[f"{name}/{stream}/{category}"] = \
                            f"{sum(r[stream]['why'] is None for r in cat)}/{len(cat)}"
        same = [r["same"] for r in out[name] if "same" in r]
        if same:
            summary[f"{name}/same"] = f"{sum(same)}/{len(same)}"
    out["summary"] = summary
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
