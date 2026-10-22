"""Qwen3.5 with per-request LoRA adapters, served by mstar, against HF transformers + PEFT.

GPU only; nothing here runs in CI. Phases, each writing JSON under ``--out`` so one can
be rerun on its own:

``reference``
    HF transformers + PEFT, greedy, per (adapter, prompt): the generated ids and, at
    every step, the top-k ids of the reference's own distribution. Each adapter is
    checked to have reached every LoRA module its file names (a PEFT key that matches
    no module is otherwise only a warning, and the "reference" would be the base
    model). An adapter PEFT cannot load into the class the prompt needs (Tinker's split
    delta-net ``in_proj_q/k/v`` and ``unembed_tokens``, a layout other than the class's,
    or a text-only layout on an image prompt) is merged into the HF weights instead, by
    this script's own name map (``_merge``; the head untied first, as PEFT adapts it
    apart from the embedding); ``reference.json`` records which, per adapter and kind.
    Runs in a child process, so the GPU is free before mstar starts.
``serve``
    mstar with the stock serving config plus a ``resources: lora:`` block naming the
    adapters. ``single``: each (adapter, prompt) alone. ``mixed``: all of them at once,
    so rows on different adapters and on the base model share batches. Also
    ``/v1/models`` and the 400 for an unknown adapter.
``serve --eager``
    The same, with ``benchmark/lora/eager_site`` on the server's ``PYTHONPATH``: the LLM
    node runs without CUDA graphs and ``torch.compile``. Writes ``mstar_eager.json``.
``compare``
    Per (adapter, prompt), vLLM's rule for greedy parity in bf16: the outputs agree up
    to the first differing token, and there mstar's token is in the reference's top-k.
    ``mixed`` must pass the same rule. A control checks each adapter changes the
    output of at least one prompt, so passing does not mean "the adapter never ran".
    Run for the captured server, and for the eager one when its output exists.
``graphs``
    Captured against eager, row by row: identical, or equal up to a first difference
    where both tokens are in the reference's top-k (a near tie the two kernel paths
    may break differently). A difference after the shared prefix has already left the
    reference is reported, not judged.
``tp``
    Needs ``--tp 2`` or more GPUs and the captured single-GPU ``serve`` output. For each
    ``MSTAR_TP_ASYNC_SCHED`` in ``--async-sched`` (default both 0 and 1): the same
    server with the LLM node tensor-parallel, the ``serve`` requests checked against
    the single-GPU run by the ``graphs`` rule, then an eviction-churn load on the same
    server (every adapter and the base model round-robin, more adapters than slots,
    concurrency 64) that must finish with no failed request and must have evicted.

The served LoRA path runs the Triton kernels (``backend: auto`` on CUDA) in bf16, inside
the captured decode and prefill graphs unless ``--eager``.

    python -m benchmark.lora.qwen3_5_parity --model qwen3_5_0.8b --out /tmp/lora_parity

Prompts: six text prompts, and unless ``--no-images`` two image prompts (``img0``,
``img1``), so a vision-tower adapter's delta (and an embedding adapter's on a prompt's
spliced text) is in the comparison.

Default adapters (found on the HF Hub on 2026-10-20; third-party). For 0.8B: three are rank 16,
alpha 32, attention + MLP only. Three also touch the gated delta-net layers
(``linear_attn.*``): ``transaction`` and ``atemoko`` are PEFT ``all-linear`` (rank 16,
alpha 32: ``in_proj_qkv``, ``in_proj_z``, ``in_proj_a``, ``in_proj_b``, ``out_proj``),
and ``medvqa`` is rank 8, alpha 16, on ``in_proj_qkv``, ``in_proj_z`` and ``out_proj``
only, so the fused in-projection runs with some slices off the adapter. ``router`` and
``transaction`` are saved in the text-only causal LM's naming (``model.layers.*``), the
others in the released checkpoint's (``model.language_model.layers.*``), so both layouts
are exercised. ``agnews`` (unsloth, rank 16) also targets the vision tower's blocks and
merger. For 2B, ``comma`` also adapts the conv patch embed. For 4B: ``imajev`` (the most
downloaded), ``oneinbox`` (Tinker: split ``in_proj_q/k/v`` and ``unembed_tokens``, rank 32),
``pirate`` (PEFT ``lm_head`` and ``embed_tokens``, rank 8) and ``clausetagger`` (``lm_head``,
rank 32). For 9B: ``nimble``, ``rlvr`` (Tinker) and ``mkd2world`` (vision, rank 32). The
slots are sized to the largest rank among the adapters. Pass ``--adapter
name=hf_repo_or_dir`` (repeatable) to use others; an adapter for a different size needs a
matching ``--model``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

DEFAULT_ADAPTERS = {
    "qwen3_5_0.8b": {
        "proofreader": "americansquid/Qwen3.5-0.8B-Proofreader-Light",
        "router": "Lance1573/acrouter-qwen35-08b-router-lora",
        "pinocchio": "KevinDavidHayes/pinocchio-0.8b",
    },
}
# also on the delta-net projections; parity only (`phase_plan` and `runtime_load` keep the
# three above)
DELTA_NET_ADAPTERS = {
    "qwen3_5_0.8b": {
        "transaction": "indrapurnayasa/transaction-qwen3.5-0.8b-lora",
        "atemoko": "BurnyCoder/qwen3.5-0.8b-atemokoloporos-positive-expanded",
        "medvqa": "thealper2/qwen3.5-0.8b-slake-medvqa-lora",
    },
}
# the rest of the model: the vision tower, the LM head, the embedding, other key layouts
WHOLE_MODEL_ADAPTERS = {
    "qwen3_5_0.8b": {"agnews": "hammamwahab/jev-agnews-lora"},
    "qwen3_5_2b": {
        "imajev": "mohit67890/imajev-2b",
        "comma": "wjbmattingly/comma-qwen-3.5-2b",
    },
    "qwen3_5_4b": {
        "imajev": "mohit67890/imajev-4b",
        "oneinbox": "dKhRr/oneinbox-qwen3.5-4b-lora",
        "pirate": "William-Gao1/qwen3.5-4b-pirate-lora",
        "clausetagger": "litillabs/litil-clausetagger-4b",
    },
    "qwen3_5_9b": {
        "nimble": "bespokelabs/Bespoke-Nimble-9B",
        "rlvr": "lucabaroni/qwen3.5-9b-rlvr-reward-hacking",
        "mkd2world": "ZhihaoNan/MKD2World-9B",
    },
}
# the stock configs' warmup images
IMAGES = [
    "hf://nvidia/Cosmos3-Edge/assets/example_action_fd_umi_first_frame.png",
    "hf://nvidia/Cosmos3-Edge/assets/edge_action_id_av_0_output.png",
]
IMAGE_PROMPTS = ["Describe this image in one sentence.", "What objects are in this picture? Answer briefly."]
HF_REPOS = {
    "qwen3_5_0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3_5_2b": "Qwen/Qwen3.5-2B",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
    "qwen3_5_9b": "Qwen/Qwen3.5-9B",
}
PROMPTS = [
    "Fix the grammar: 'their going to the store tomorow, and he dont know why.'",
    "Write a SQL query that returns the ten most recent orders per customer.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 100.",
    "Translate to French: 'The meeting is moved to Thursday afternoon.'",
    "What is 17 * 23? Answer with the number only.",
]
BASE = "<base>"  # the run with no adapter


def _adapter_dir(spec: str) -> Path:
    """A local directory holding ``adapter_model.safetensors`` (and its config)."""
    if os.path.isdir(spec):
        return Path(spec)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(spec, allow_patterns=["adapter_model.safetensors", "adapter_config.json"]))


def _messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _prompts(args) -> dict[str, tuple[str, str | None]]:
    """Prompt key -> (text, ``hf://`` image or None): the text prompts by index, then
    the image prompts as ``img<i>``."""
    out = {str(i): (text, None) for i, text in enumerate(PROMPTS)}
    if not args.no_images:
        out.update({f"img{i}": (text, IMAGES[i]) for i, text in enumerate(IMAGE_PROMPTS)})
    return out


def _image_path(ref: str) -> str:
    from huggingface_hub import hf_hub_download

    owner, repo, path = ref[len("hf://"):].split("/", 2)
    return hf_hub_download(f"{owner}/{repo}", path)


def _keys(spec: str) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(_adapter_dir(spec) / "adapter_model.safetensors"), "pt") as f:
        return list(f.keys())


def _max_rank(args) -> int:
    """The largest rank among the adapters, as the slots must hold."""
    from safetensors import safe_open

    rank = 16
    for spec in args.adapters.values():
        with safe_open(str(_adapter_dir(spec) / "adapter_model.safetensors"), "pt") as f:
            for key in f.keys():
                if key.endswith(("lora_A.weight", "lora_embedding_A")):
                    rank = max(rank, f.get_slice(key).get_shape()[0])
    return rank


# --- reference -----------------------------------------------------------------------

def reference(args) -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    repo = HF_REPOS[args.model]
    processor = AutoProcessor.from_pretrained(repo)
    tok = processor.tokenizer
    prompts = _prompts(args)
    out: dict[str, dict] = {"_how": {}}
    for name, spec in [(BASE, None), *args.adapters.items()]:
        rows = {}
        for multimodal in (False, True):
            keys = [k for k, (_, image) in prompts.items() if (image is not None) == multimodal]
            if not keys:
                continue
            model, how = _reference_model(repo, name, spec, multimodal)
            out["_how"][f"{name}/{'image' if multimodal else 'text'}"] = how
            for key in keys:
                text, image = prompts[key]
                if image is None:
                    chat = tok.apply_chat_template(
                        _messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                    inputs = tok(chat, return_tensors="pt", add_special_tokens=False).to("cuda")
                else:
                    chat = processor.apply_chat_template(
                        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                    pixels = [Image.open(_image_path(image)).convert("RGB")]
                    inputs = processor(text=[chat], images=pixels, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    gen = model.generate(
                        **inputs, max_new_tokens=args.max_tokens, do_sample=False,
                        output_scores=True, return_dict_in_generate=True,
                        eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id],
                    )
                new = gen.sequences[0, inputs["input_ids"].shape[1]:].tolist()
                topk = [s[0].float().topk(args.topk).indices.tolist() for s in gen.scores]
                rows[key] = {"ids": new, "topk": topk}
            del model
            torch.cuda.empty_cache()
        out[name] = rows
        print(f"reference {name}: {len(rows)} prompts ({out['_how']})", flush=True)
    _write(args.out / "reference.json", out)


# keys PEFT has no module for in transformers' Qwen3.5
_TINKER = (".in_proj_q.", ".in_proj_k.", ".in_proj_v.", "unembed_tokens")


def _reference_model(repo: str, name: str, spec: str | None, multimodal: bool):
    """The HF model for one adapter: ``(model, "base" | "peft" | "merge")``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    keys = [] if spec is None else _keys(spec)
    released = any(".language_model." in k or ".visual." in k for k in keys)
    text_only = any(k.startswith("base_model.model.model.layers.") for k in keys)
    cls = AutoModelForCausalLM if (text_only and not multimodal) else AutoModelForImageTextToText
    model = cls.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda")
    if spec is None:
        return model.eval(), "base"
    tinker = any(marker in k for k in keys for marker in _TINKER) or any(
        "embed_tokens.lora_A" in k for k in keys
    )
    if not tinker and (released or (text_only and cls is AutoModelForCausalLM)):
        model = PeftModel.from_pretrained(model, str(_adapter_dir(spec)), adapter_name=name)
        reached = sum(
            1 for _, m in model.named_modules()
            if (name in getattr(m, "lora_A", {})) or (name in getattr(m, "lora_embedding_A", {}))
        )
        wanted = sum(1 for k in keys if k.endswith(("lora_A.weight", "lora_embedding_A")))
        if reached != wanted:
            raise SystemExit(f"{name}: PEFT reached {reached} LoRA modules, the file names {wanted}")
        return model.eval(), "peft"
    _merge(model, _adapter_dir(spec))
    return model.eval(), "merge"


def _merge(model, directory: Path) -> None:
    """``W += scaling * B @ A`` for every LoRA module of the adapter in ``directory``,
    placed by name on transformers' Qwen3.5 (any of the key layouts, Tinker's split
    delta-net q/k/v into ``in_proj_qkv``'s rows, ``unembed_tokens`` on the head). The
    head is untied first, so a tied checkpoint's embedding and head take their own."""
    import math

    import torch
    from safetensors.torch import load_file

    config = json.loads((directory / "adapter_config.json").read_text())
    sd = load_file(str(directory / "adapter_model.safetensors"))
    text = getattr(model.config, "text_config", model.config)
    key_dim = text.linear_num_key_heads * text.linear_key_head_dim
    head = model.get_output_embeddings()
    if head.weight is model.get_input_embeddings().weight:
        head.weight = torch.nn.Parameter(head.weight.detach().clone())
    params = dict(model.named_parameters())
    stack = next(p for p in ("model.language_model.", "model.") if any(n.startswith(p + "layers.") for n in params))
    pairs: dict[str, dict] = {}
    for key, value in sd.items():
        module = key.removeprefix("base_model.model.")
        for suffix, kind in ((".lora_A.weight", "A"), (".lora_B.weight", "B"),
                             (".lora_embedding_A", "EA"), (".lora_embedding_B", "EB")):
            if module.endswith(suffix):
                pairs.setdefault(module[:-len(suffix)], {})[kind] = value.float().cuda()
    for module, ab in pairs.items():
        embedding = "EA" in ab
        a, b = (ab["EA"], ab["EB"]) if embedding else (ab["A"], ab["B"])
        r = a.shape[0]
        alpha = config.get("alpha_pattern", {}).get(module.rsplit(".", 1)[-1], config["lora_alpha"])
        scaling = alpha / (math.sqrt(r) if config.get("use_rslora") else r)
        delta = scaling * (b.reshape(b.shape[0], -1) @ a.reshape(r, -1))
        rows = slice(None)
        if module == "lm_head" or module.endswith("unembed_tokens"):
            param = "lm_head.weight"
        elif ".visual." in f".{module}":
            param = "model.visual." + module.split("visual.", 1)[1] + ".weight"
        else:
            rest = module[module.index("layers."):] if "layers." in module else "embed_tokens"
            if rest == "embed_tokens" and not embedding:
                raise SystemExit(f"{module}: a linear-shaped LoRA on the embedding is ambiguous; mstar refuses it too")
            for shard, at in (("q", 0), ("k", key_dim), ("v", 2 * key_dim)):
                if rest.endswith(f".in_proj_{shard}"):
                    rest = rest[:-len(shard)] + "qkv"
                    rows = slice(at, at + delta.shape[0])
            param = stack + rest + ".weight"
        weight = params[param]
        if embedding:
            delta = delta.t()
        with torch.no_grad():
            weight[rows] += delta.view(weight[rows].shape).to(weight.dtype)


# --- mstar -----------------------------------------------------------------------------

def _serving_config(args, workdir: Path, tp: int = 1) -> Path:
    stock = Path(__file__).resolve().parents[2] / "configs" / f"{args.model}.yaml"
    config = yaml.safe_load(stock.read_text())
    if tp > 1:
        # the LLM node over ranks 0..tp-1; the vision encoder stays on rank 0
        config["node_groups"] = [
            {"node_names": ["vision_encoder"], "ranks": [0]},
            {"node_names": ["LLM"], "ranks": list(range(tp)), "tp_size": tp},
        ]
    config.setdefault("resources", {})["lora"] = {
        "max_slots": args.max_slots,
        "max_rank": _max_rank(args),
        "adapters": {
            name: str(_adapter_dir(spec) / "adapter_model.safetensors")
            for name, spec in args.adapters.items()
        },
    }
    if args.no_images:
        # the warmup's images compile the vision tower, which a text-only check skips
        config.pop("warmup_requests", None)
    path = workdir / f"{args.model}_lora_tp{tp}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _post(url: str, body: dict, timeout: float = 600) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _chat(url: str, model: str, prompt: str, max_tokens: int, image: str | None = None) -> str:
    content = prompt
    if image is not None:
        import base64

        data = base64.b64encode(Path(_image_path(image)).read_bytes()).decode()
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
            {"type": "text", "text": prompt},
        ]
    status, body = _post(f"{url}/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
        "max_tokens": max_tokens, "enable_thinking": False,
    })
    if status != 200:
        raise RuntimeError(f"{model}: HTTP {status}: {body}")
    return body["choices"][0]["message"]["content"]


EAGER_SITE = Path(__file__).resolve().parent / "eager_site"
EAGER_MARK = "torch.compile disabled for LLM"


def _outputs(args, eager: bool) -> Path:
    return args.out / ("mstar_eager.json" if eager else "mstar.json")


def _tp_tag(tp: int, async_sched: str) -> str:
    return f"tp{tp}_async{async_sched}"


def serve(args, eager: bool = False, tp: int = 1, async_sched: str | None = None) -> None:
    """``tp`` > 1 serves the LLM node tensor-parallel with ``MSTAR_TP_ASYNC_SCHED`` set
    to ``async_sched``, writes ``mstar_<tag>.json`` and then runs the churn load."""
    from benchmark.eval.server import ManagedServer

    env = {}
    if eager:
        repo = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(EAGER_SITE), str(repo), os.environ.get("PYTHONPATH")) if p
        )
    tag = _tp_tag(tp, async_sched) if tp > 1 else None
    if tag is not None:
        env["MSTAR_TP_ASYNC_SCHED"] = async_sched
    log = args.out / (
        f"server_{tag}.log" if tag else "server_eager.log" if eager else "server.log"
    )
    churn = None
    with tempfile.TemporaryDirectory(prefix="lora-parity-") as tmp:
        config = _serving_config(args, Path(tmp), tp=tp)
        server = ManagedServer(args.model, config=str(config), env=env, log_path=log)
        with server:
            url = server.url
            base_name = args.model
            names = [BASE, *args.adapters]

            prompts = _prompts(args)

            def run(name: str, i: str) -> str:
                text, image = prompts[i]
                return _chat(url, base_name if name == BASE else name, text, args.max_tokens, image)

            jobs = [(n, i) for n in names for i in prompts]
            single = {}
            for n, i in jobs:
                single.setdefault(n, {})[i] = run(n, i)
            mixed = {}
            with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
                for (n, i), text in zip(jobs, pool.map(lambda j: run(*j), jobs), strict=True):
                    mixed.setdefault(n, {})[i] = text
            with urllib.request.urlopen(f"{url}/v1/models") as r:
                models = json.loads(r.read())
            unknown, _ = _post(f"{url}/v1/chat/completions", {
                "model": "no-such-adapter", "messages": _messages("hi"), "max_tokens": 4,
            })
            if tag is not None:
                from benchmark.lora.load import closed_loop

                # more adapters than slots, all in flight at once: slots churn while
                # requests leave, which is where a rank deciding alone would hang
                churn = closed_loop(
                    url, [base_name, *args.adapters], concurrency=64, requests=512,
                    max_tokens=64, warmup=0, timeout=300.0,
                )
    text = log.read_text(errors="replace")
    if (EAGER_MARK in text) != eager:
        raise SystemExit(f"{log}: expected the LLM node {'eager' if eager else 'compiled'}")
    out = {"single": single, "mixed": mixed, "models": models, "unknown_status": unknown}
    if tag is not None:
        out["churn"] = churn
        out["evictions"] = text.count("LoRA pool: evicting")
    _write(args.out / f"mstar_{tag}.json" if tag else _outputs(args, eager), out)


# --- compare ---------------------------------------------------------------------------

def _agree(ids: list[int], ref: dict) -> dict:
    """vLLM's greedy rule: equal up to the first difference, and there mstar's token is
    in the reference's top-k. Past that the two continue from different prefixes."""
    want = ref["ids"]
    for t, (got, exp) in enumerate(zip(ids, want, strict=False)):
        if got != exp:
            in_topk = t < len(ref["topk"]) and got in ref["topk"][t]
            return {"match": t, "diverged": True, "pass": in_topk}
    # one stopped earlier: fine if the shorter one ended on a stop (lengths are capped alike)
    return {"match": min(len(ids), len(want)), "diverged": False, "pass": True}


def compare(args) -> int:
    code = _compare(args, eager=False)
    if _outputs(args, eager=True).exists():
        code |= _compare(args, eager=True)
    return code


def _compare(args, eager: bool) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_REPOS[args.model])
    ref = json.loads((args.out / "reference.json").read_text())
    got = json.loads(_outputs(args, eager).read_text())
    report, failed = {}, []
    for mode in ("single", "mixed"):
        for name, rows in got[mode].items():
            for i, text in rows.items():
                # mstar returns text: re-tokenized, which round-trips for ordinary output
                ids = tok(text, add_special_tokens=False).input_ids
                result = _agree(ids, ref[name][i])
                report[f"{mode}/{name}/{i}"] = result
                if not result["pass"]:
                    failed.append(f"{mode}/{name}/{i}")
    # the adapters must actually change something, in the reference and in mstar
    for name in args.adapters:
        for source, rows in (("reference", ref), ("mstar", got["single"])):
            same = all(
                (rows[name][i]["ids"] if source == "reference" else rows[name][i])
                == (rows[BASE][i]["ids"] if source == "reference" else rows[BASE][i])
                for i in rows[name]
            )
            if same:
                failed.append(f"control/{source}/{name}: identical to the base model on every prompt")
    # the base model and every adapter, each once; the server's order is its own
    listed = [m["id"] for m in got["models"]["data"]]
    if args.model not in listed or sorted(listed) != sorted([args.model, *args.adapters]):
        failed.append(f"/v1/models lists {listed}")
    if got["unknown_status"] != 400:
        failed.append(f"an unknown adapter got HTTP {got['unknown_status']}, not 400")
    # a row batched beside other adapters against the same row alone. Eager bf16 GEMMs
    # are not batch-invariant, so a late divergence is reported, not failed; the rule
    # above still holds each mixed row to the reference.
    same = [k for k, text in _flat(got["mixed"]).items() if text == _flat(got["single"])[k]]
    exact = sum(1 for r in report.values() if not r["diverged"])
    _write(args.out / ("compare_eager.json" if eager else "compare.json"),
           {"rows": report, "mixed_equal_single": same, "failed": failed})
    print(f"[{'eager' if eager else 'captured'}] {exact}/{len(report)} rows identical to the reference; "
          f"{len(same)}/{len(_flat(got['mixed']))} mixed rows identical to their single run; "
          f"{len(failed)} failure(s)")
    for f in failed:
        print("  FAIL", f)
    return 1 if failed else 0


def graphs(args) -> int:
    """Captured against eager, by the rule in the module docstring."""
    return _versus(
        args, _outputs(args, eager=False), _outputs(args, eager=True),
        args.out / "graphs.json", "captured vs eager",
    )


def tp(args) -> int:
    """Each TP run against the single-GPU captured one, then its churn load."""
    code = 0
    for async_sched in args.async_sched:
        tag = _tp_tag(args.tp, async_sched)
        path = args.out / f"mstar_{tag}.json"
        code |= _versus(args, _outputs(args, eager=False), path,
                        args.out / f"tp_{tag}.json", f"single GPU vs {tag}")
        got = json.loads(path.read_text())
        churn, evictions = got["churn"], got["evictions"]
        print(f"[{tag}] churn: {churn['requests']} requests at c={churn['concurrency']}, "
              f"{churn['failed']} failed, {churn['wall_s']:.0f} s, {evictions} evictions")
        if churn["failed"] or not evictions:
            print("  FAIL", "failed requests" if churn["failed"] else "nothing was evicted",
                  churn["errors"])
            code = 1
    return code


def _versus(args, a_path: Path, b_path: Path, out: Path, label: str) -> int:
    """Two served runs, row by row, by the ``graphs`` rule."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_REPOS[args.model])
    ref = json.loads((args.out / "reference.json").read_text())
    run_a = json.loads(a_path.read_text())
    run_b = json.loads(b_path.read_text())
    rows, failed = {}, []
    for mode in ("single", "mixed"):
        for key, text in _flat(run_a[mode]).items():
            name, i = key.split("/")
            a = tok(text, add_special_tokens=False).input_ids
            b = tok(_flat(run_b[mode])[key], add_special_tokens=False).input_ids
            t = next((t for t, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
            if t is None:
                rows[f"{mode}/{key}"] = {"identical": len(a) == len(b), "diverged_at": None}
                continue
            r = ref[name][i]
            on_ref = r["ids"][:t] == a[:t]
            tie = on_ref and t < len(r["topk"]) and a[t] in r["topk"][t] and b[t] in r["topk"][t]
            verdict = "near_tie" if tie else ("fail" if on_ref else "unjudged")
            rows[f"{mode}/{key}"] = {"identical": False, "diverged_at": t, "verdict": verdict}
            if verdict == "fail":
                failed.append(f"{mode}/{key}: diverged at token {t}, outside the reference's top-k")
    identical = sum(1 for r in rows.values() if r["diverged_at"] is None)
    _write(out, {"rows": rows, "failed": failed})
    print(f"{label}: {identical}/{len(rows)} rows token-identical; {len(failed)} failure(s)")
    for f in failed:
        print("  FAIL", f)
    return 1 if failed else 0


def _flat(rows: dict[str, dict[str, str]]) -> dict[str, str]:
    return {f"{name}/{i}": text for name, by_prompt in rows.items() for i, text in by_prompt.items()}


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3_5_0.8b", choices=sorted(HF_REPOS))
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=REPO_OR_DIR")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("all", "reference", "serve", "compare", "graphs", "tp"),
                   default="all")
    p.add_argument("--eager", action="store_true",
                   help="serve: the LLM node without CUDA graphs or torch.compile")
    p.add_argument("--no-images", action="store_true",
                   help="text prompts only (no vision-tower warmup or image prompts)")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max-slots", type=int, default=2,
                   help="fewer slots than adapters exercises eviction in the mixed phase")
    p.add_argument("--tp", type=int, default=1,
                   help="serve / tp: the LLM node's TP size (needs that many GPUs)")
    p.add_argument("--async-sched", default="0,1",
                   help="tp: the MSTAR_TP_ASYNC_SCHED values to run, comma-separated")
    args = p.parse_args()
    args.async_sched = args.async_sched.split(",")
    args.adapters = dict(a.split("=", 1) for a in args.adapter) or {
        **DEFAULT_ADAPTERS.get(args.model, {}), **DELTA_NET_ADAPTERS.get(args.model, {}),
        **WHOLE_MODEL_ADAPTERS.get(args.model, {}),
    }
    if not args.adapters:
        p.error(f"no default adapters for {args.model}; pass --adapter name=repo")
    if args.phase == "reference":
        reference(args)
        return 0
    if args.phase == "serve":
        if args.tp > 1:
            for async_sched in args.async_sched:
                serve(args, tp=args.tp, async_sched=async_sched)
        else:
            serve(args, eager=args.eager)
        return 0
    if args.phase == "tp":
        if args.tp < 2:
            p.error("--phase tp needs --tp 2 or more")
        for async_sched in args.async_sched:
            if not (args.out / f"mstar_{_tp_tag(args.tp, async_sched)}.json").exists():
                serve(args, tp=args.tp, async_sched=async_sched)
        return tp(args)
    if args.phase == "compare":
        return compare(args)
    if args.phase == "graphs":
        return graphs(args)
    t0 = time.monotonic()
    # the reference in its own process, so its GPU memory is gone before mstar starts
    subprocess.run([sys.executable, "-m", "benchmark.lora.qwen3_5_parity", *sys.argv[1:],
                    "--phase", "reference"], check=True)
    serve(args)
    serve(args, eager=True)
    code = compare(args) | graphs(args)
    print(f"done in {time.monotonic() - t0:.0f} s; results in {args.out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
