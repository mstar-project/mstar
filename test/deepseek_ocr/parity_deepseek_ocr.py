#!/usr/bin/env python3
"""Parity of the M* DeepSeek-OCR port against the checkpoint's remote code (GPU, real weights).

The reference is ``modeling_deepseekocr.py``'s ``infer`` path: its prompt and
views, bf16 weights under bf16 autocast, greedy, with the n-gram ban off
(``infer`` hard-codes ``no_repeat_ngram_size=20``; M* has no n-gram ban). The
remote code needs transformers 4.46.3, which is not the one mstar runs on, so
the reference runs in a second interpreter (``--hf-python``), where this
script calls itself with ``reference``: once with eager attention, the gate,
and once with ``flash_attention_2``, the model card's, whose distance from
eager is the reference's own spread and is printed beside every M* number.

Per page, M* runs its two nodes eagerly, each driven through its resources
the way the engine steps it (declare, admit, plan, forward, commit), and is
compared with the eager reference on:

1. the prompt ids, exactly;
2. DeepEncoder's rows, the embeddings the decoder reads at the image
   positions: every row's cosine to the reference's at least ``ROW_COS_MIN``;
3. the prefill's last-position logits: within ``LOGIT_ATOL`` of the
   reference's everywhere, and the same top-1 where the reference's top-2
   margin is above ``CONFIDENT_MARGIN``;
4. the greedy ids to EOS (or ``--max-new``): the same, or the same up to a
   step where the reference's top two logits are within ``NEAR_TIE``. With
   ``--server`` the served path (CUDA graphs, packed prefills) is held to the
   same.

Exits 1 on any mismatch. The bounds are the reference's own: its eager and
flash-attention runs differ by up to 0.97 in the prefill logits on the nine
pages these were set on, and split on a greedy step whose top two logits
are one bf16 step (0.125) apart.

    python test/deepseek_ocr/parity_deepseek_ocr.py --hf-python <venv with transformers==4.46.3>/bin/python \\
        --pages page1.png page2.png --json parity.json [--server http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = "deepseek-ai/DeepSeek-OCR"
# the model card's prompt; the image renders first, as ``infer`` puts it
INSTRUCTION = "Free OCR."
PROMPT = f"<image>\n{INSTRUCTION}"
IMAGE_TOKEN_ID = 128815

ROW_COS_MIN = 0.999
LOGIT_ATOL = 1.0
CONFIDENT_MARGIN = 1.0
# two bf16 steps at the logits' magnitude (16-32)
NEAR_TIE = 0.25


def resolve_snapshot(repo: str) -> str:
    if Path(repo).is_dir():
        return repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True)


# ---------------------------------------------------------------------------
# Reference (runs under --hf-python; imports nothing from mstar)
# ---------------------------------------------------------------------------


def reference_inputs(mod, tokenizer, image, max_crops: int):
    """``infer``'s inputs for ``PROMPT`` in gundam mode (base 1024, tiles 640)."""
    from PIL import ImageOps

    base, size, patch, down = 1024, 640, 16, 4
    transform = mod.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    before, after = PROMPT.split("<image>")
    ids = mod.text_encode(tokenizer, before, bos=False, eos=False)
    mask = [False] * len(ids)
    if image.size[0] <= size and image.size[1] <= size:
        raw, (cols, rows) = [], (1, 1)
    else:
        raw, (cols, rows) = mod.dynamic_preprocess(image, max_num=max_crops)
    pad = ImageOps.pad(image, (base, base), color=tuple(int(x * 255) for x in transform.mean))
    globals_ = transform(pad).to(torch.bfloat16)[None]
    tiled = cols > 1 or rows > 1
    crops = (torch.stack([transform(c).to(torch.bfloat16) for c in raw]) if tiled
             else torch.zeros((1, 3, base, base), dtype=torch.bfloat16))
    nq, nqb = size // patch // down, base // patch // down
    image_ids = ([IMAGE_TOKEN_ID] * nqb + [IMAGE_TOKEN_ID]) * nqb + [IMAGE_TOKEN_ID]
    if tiled:
        image_ids += ([IMAGE_TOKEN_ID] * (nq * cols) + [IMAGE_TOKEN_ID]) * (nq * rows)
    ids += image_ids
    mask += [True] * len(image_ids)
    tail = mod.text_encode(tokenizer, after, bos=False, eos=False)
    ids = [0] + ids + tail
    mask = [False] + mask + [False] * len(tail)
    return (torch.tensor(ids), torch.tensor(mask), crops, globals_,
            torch.tensor([[cols, rows]], dtype=torch.long))


@torch.no_grad()
def run_reference(args) -> None:
    from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList

    snapshot = resolve_snapshot(args.repo)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModel.from_pretrained(snapshot, _attn_implementation=args.attn, trust_remote_code=True,
                                      use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)
    mod = sys.modules[type(model).__module__]

    class Margins(LogitsProcessor):
        """Each greedy step's top-1 minus top-2 logit; changes nothing."""

        def __init__(self):
            self.margins = []

        def __call__(self, input_ids, scores):
            top = scores.float().topk(2, dim=-1).values[0]
            self.margins.append(float(top[0] - top[1]))
            return scores

    # what the decoder's first layer reads: the prompt embeddings with the image rows in place
    seen = []
    model.model.layers[0].register_forward_pre_hook(
        lambda _m, a, kw: seen.append((a[0] if a else kw["hidden_states"]).detach()), with_kwargs=True)

    records = {}
    for path in args.pages:
        image = mod.load_image(path).convert("RGB")
        ids, mask, crops, globals_, spatial = reference_inputs(mod, tokenizer, image, args.max_crops)
        images = [(crops.cuda(), globals_.cuda())]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            seen.clear()
            out = model(input_ids=ids[None].cuda(), images=images, images_seq_mask=mask[None].cuda(),
                        images_spatial_crop=spatial, use_cache=False)
            rows = seen[0][0][mask.cuda()].float().cpu()
            logits = out.logits[0, -1].float().cpu()
            margins = Margins()
            t0 = time.time()
            gen = model.generate(
                ids[None].cuda(), images=images, images_seq_mask=mask[None].cuda(), images_spatial_crop=spatial,
                do_sample=False, num_beams=1, no_repeat_ngram_size=0, eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new, use_cache=True, logits_processor=LogitsProcessorList([margins]),
            )
        greedy = gen[0, ids.numel():].cpu()
        records[os.path.basename(path)] = {
            "ids": ids, "rows": rows, "logits": logits, "greedy": greedy,
            "margins": torch.tensor(margins.margins), "grid": spatial[0].tolist(), "secs": time.time() - t0,
        }
        print(f"reference {args.attn} {os.path.basename(path)}: {ids.numel()} prompt ids, "
              f"{greedy.numel()} greedy ids in {time.time() - t0:.0f}s", flush=True)
    torch.save(records, args.out)


# ---------------------------------------------------------------------------
# M*: the two nodes, eager, through their resources
# ---------------------------------------------------------------------------


class LLMDriver:
    """Steps the ``LLM`` node one request at a time on resources built from
    the model's own declaration, as ``Engine`` steps it, minus the worker."""

    def __init__(self, model, llm, device: torch.device, max_num_pages: int = 1024):
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.distributed.communication import CommGroup, JointGroups
        from mstar.engine.resources import StepRunner, resolve_spec_dependencies
        from mstar.engine.resources.base import EngineResourceInfo, build_resource
        from mstar.engine.resources.kv.transfer import TransferEngineInfo

        specs = [s for s in model.get_node_resources() if "LLM" in s.nodes]
        by_key = resolve_spec_dependencies(specs)
        for spec in specs:
            config = getattr(spec, "config", None)
            if config is not None and hasattr(config, "max_num_pages"):
                spec.apply_yaml_overrides(max_num_pages=max_num_pages)
        groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        transfer = TransferEngineInfo("h", "h", LocalTransferEngine("h"))
        self.resources = {
            spec.resource_key: build_resource(spec, EngineResourceInfo(
                device=device, joint_comm_group=groups, transfer_engine_info=transfer, kv_dtype=torch.bfloat16,
                dependencies={key: by_key[key] for key in spec.depends_on()},
            ))
            for spec in specs
        }
        llm.bind_node_resources(self.resources)
        self.model, self.llm = model, llm
        self.runner = StepRunner(self.resources)

    def open(self, rid: int, name: str, step_metadata: dict):
        """``rid`` is the request's worker handle, as the engine keys resources."""
        from mstar.conductor.request_info import CurrentForwardPassInfo

        configs = self.model.get_request_resource_configs({}, {})
        for config in configs.values():
            config.apply_conductor_config(seed=0)
        self.runner.ingest_request(rid, configs)
        return CurrentForwardPassInfo(request_id=name, graph_walk="", fwd_index=0, random_seed=0, max_tokens=1 << 16,
                                      rid_handle=rid, resource_configs=configs, step_metadata=step_metadata)

    def step(self, walk: str, fwd, inputs: dict) -> torch.Tensor:
        from mstar.engine.resources import StepContext
        from mstar.model.submodule_base import ModelInputsFromEngine

        rid = fwd.rid_handle
        fwd.graph_walk = walk
        prepared = self.llm.prepare_inputs(walk, fwd, inputs)
        step = self.llm.declare_step(walk, [rid], [prepared])
        step.set_ctx(StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False))
        outcome = self.runner.admit(step)
        assert outcome.ok, f"admit failed: {outcome.reason}"
        self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd},
                                              resources=dict(self.resources), step=step)
        tensors = self.llm.preprocess(walk, engine_inputs, [prepared])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            token = self.llm.forward(walk, engine_inputs, **tensors)["new_token"][0]
        self.runner.commit(step)
        return token.view(-1)

    def close(self, rid: int) -> None:
        self.runner.remove_request(rid)


@torch.no_grad()
def run_mstar(args, snapshot: str) -> dict:
    from mstar.model.deepseek_ocr.config import STEP_GRIDS_KEY
    from mstar.model.deepseek_ocr.deepseek_ocr_model import DeepseekOCRModel
    from mstar.model.multimodal import PromptPart
    from mstar.model.submodule_base import ModelInputsFromEngine

    device = torch.device("cuda")
    model = DeepseekOCRModel(snapshot, max_crops=args.max_crops)
    vision = model.get_submodule("vision_encoder", device="cuda", autocast_dtype=torch.bfloat16)
    llm = model.get_submodule("LLM", device="cuda", autocast_dtype=torch.bfloat16)
    driver = LLMDriver(model, llm, device)
    stop = set(model.text_output().stop_token_ids)
    # the logits of each step, as the node's forward hands them to the sampler
    logits_of_step = []
    head = llm.language_model.logits
    llm.language_model.logits = lambda hidden: logits_of_step.append(head(hidden)) or logits_of_step[-1]

    out = {}
    for rid, path in enumerate(args.pages):
        name = os.path.basename(path)
        # a chat's parts, image first, as the reference's prompt orders them
        parts = [PromptPart("image", index=0), PromptPart("text", text=INSTRUCTION)]
        prompt = model.process_prompt(None, ["image", "text"], ["text"], prompt_parts=parts,
                                      tensors={"image_inputs": [model.load_image(path, "cpu").data]})
        signals, metadata = prompt.new_input_tensors, prompt.metadata
        ids = signals["text_inputs"][0]
        fwd = driver.open(rid, name, {STEP_GRIDS_KEY: metadata["image_grids"]})
        engine_inputs = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd})
        node_inputs = vision.prepare_inputs("prefill_image", fwd, {"image_views": signals["image_views"]})
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (rows,) = vision.forward("prefill_image", engine_inputs, [node_inputs])["image_embeds"]
        logits_of_step.clear()
        # on the device, as the transport hands a node its inputs
        tokens = [driver.step("prefill_image", fwd, {"text_inputs": [ids.to(device)], "image_embeds": [rows]})]
        logits = logits_of_step[0][-1].float().cpu()
        while int(tokens[-1]) not in stop and len(tokens) < args.max_new:
            tokens.append(driver.step("decode", fwd, {"text_inputs": [tokens[-1]]}))
        driver.close(rid)
        out[name] = {"ids": ids, "rows": rows.float().cpu(), "logits": logits, "greedy": torch.cat(tokens).cpu()}
        print(f"M* {name}: {ids.numel()} prompt ids, {len(tokens)} greedy ids", flush=True)
    return out


def served_greedy(url: str, path: str, max_new: int) -> torch.Tensor:
    """The served path's greedy ids for ``path`` over ``/generate``."""
    import requests

    with open(path, "rb") as f:
        r = requests.post(f"{url}/generate", files={"files": (os.path.basename(path), f, "image/png")}, data={
            "text": INSTRUCTION, "output_modalities": "text", "streaming": "true",
            "model_kwargs": json.dumps({"max_output_tokens": max_new, "temperature": 0.0, "return_token_ids": True}),
        }, stream=True, timeout=1800)
    r.raise_for_status()
    ids = []
    for line in r.iter_lines():
        if line:
            ids += (json.loads(line).get("metadata") or {}).get("token_ids", [])
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_rows(ours: torch.Tensor, ref: torch.Tensor) -> dict:
    if ours.shape != ref.shape:
        return {"shape": [list(ours.shape), list(ref.shape)], "ok": False}
    cos = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
    diff = (ours - ref).abs()
    return {"cos_min": round(float(cos.min()), 6), "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()), "exact": float((diff == 0).float().mean()),
            "ok": bool(cos.min() >= ROW_COS_MIN)}


def compare_logits(ours: torch.Tensor, ref: torch.Tensor) -> dict:
    top = ref.topk(2).values
    margin = float(top[0] - top[1])
    diff = (ours - ref).abs()
    same_top1 = int(ours.argmax()) == int(ref.argmax())
    return {"max_abs": round(float(diff.max()), 4), "mean_abs": round(float(diff.mean()), 5),
            "same_top1": same_top1, "ref_margin": round(margin, 3),
            "ok": float(diff.max()) <= LOGIT_ATOL and (same_top1 or margin <= CONFIDENT_MARGIN)}


def compare_greedy(ours: torch.Tensor, ref: torch.Tensor, margins: torch.Tensor | None = None) -> dict:
    n = min(ours.numel(), ref.numel())
    differs = (ours[:n] != ref[:n]).nonzero()
    first = int(differs[0]) if differs.numel() else (None if ours.numel() == ref.numel() else n)
    row = {"n": int(ours.numel()), "n_ref": int(ref.numel()), "first_divergence": first, "ok": first is None}
    if first is not None and margins is not None and first < margins.numel():
        row["ref_margin_there"] = round(float(margins[first]), 3)
        row["ok"] = row["ref_margin_there"] <= NEAR_TIE
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd")
    ref = sub.add_parser("reference", help="run the remote-code reference (under transformers 4.46.3)")
    for q in (p, ref):
        q.add_argument("--repo", default=REPO)
        q.add_argument("--pages", nargs="+", default=[])
        q.add_argument("--max-crops", type=int, default=9, help="gundam's tile cap; the card's infer uses 9")
        q.add_argument("--max-new", type=int, default=4096)
    ref.add_argument("--attn", default="eager", choices=("eager", "flash_attention_2"))
    ref.add_argument("--out", required=True)
    p.add_argument("--hf-python", help="an interpreter with transformers==4.46.3 (and flash-attn for the spread)")
    p.add_argument("--reference", nargs="*", default=[],
                   help="reuse saved reference runs (eager first) instead of running --hf-python")
    p.add_argument("--reference-dir", help="where --hf-python's runs are saved (a temporary directory if unset)")
    p.add_argument("--server", help="also hold a running M* server's greedy ids to the reference")
    p.add_argument("--json", help="write the comparison here")
    args = p.parse_args()
    if args.cmd == "reference":
        run_reference(args)
        return 0
    if not args.pages:
        p.error("--pages is required")
    snapshot = resolve_snapshot(args.repo)

    paths = list(args.reference)
    if not paths:
        if not args.hf_python:
            p.error("--hf-python or --reference is required")
        out_dir = args.reference_dir or tempfile.mkdtemp(prefix="dsocr_parity_")
        os.makedirs(out_dir, exist_ok=True)
        for attn in ("eager", "flash_attention_2"):
            out = os.path.join(out_dir, f"{attn}.pt")
            # the reference must not see this checkout's mstar or its transformers
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            subprocess.run([args.hf_python, os.path.abspath(__file__), "reference", "--repo", snapshot,
                            "--pages", *args.pages, "--max-crops", str(args.max_crops),
                            "--max-new", str(args.max_new), "--attn", attn, "--out", out], check=True, env=env)
            paths.append(out)
        print(f"reference runs saved in {out_dir}", flush=True)
    eager = torch.load(paths[0])
    fa2 = torch.load(paths[1]) if len(paths) > 1 else None
    ours = run_mstar(args, snapshot)

    report, ok = {}, True
    for path in args.pages:
        name = os.path.basename(path)
        r, m = eager[name], ours[name]
        row = {
            "ids_equal": torch.equal(m["ids"], r["ids"]),
            "rows": compare_rows(m["rows"], r["rows"]),
            "logits": compare_logits(m["logits"], r["logits"]),
            "greedy": compare_greedy(m["greedy"], r["greedy"], r["margins"]),
        }
        page_ok = row["ids_equal"] and all(row[k]["ok"] for k in ("rows", "logits", "greedy"))
        if fa2 is not None:
            # the reference's own spread: not gated
            row["fa2_vs_eager"] = {
                "logits": compare_logits(fa2[name]["logits"], r["logits"]),
                "greedy": compare_greedy(fa2[name]["greedy"], r["greedy"], r["margins"]),
            }
            row["mstar_vs_fa2_greedy"] = compare_greedy(m["greedy"], fa2[name]["greedy"], fa2[name]["margins"])
        if args.server:
            row["served"] = compare_greedy(served_greedy(args.server, path, args.max_new), r["greedy"], r["margins"])
            page_ok = page_ok and row["served"]["ok"]
        row["ok"] = page_ok
        ok = ok and page_ok
        report[name] = row
        print(f"{'PASS' if page_ok else 'FAIL'} {name}: {json.dumps(row)}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    print("parity", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
