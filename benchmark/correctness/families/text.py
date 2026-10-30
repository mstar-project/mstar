"""Greedy text (Qwen3.5 / Qwen3.8; the Qwen3-Omni thinker reuses this).

Greedy output changes across server processes and batch mixes, so there are two checks. The set
runs on two server instances and we report how often they agree (the noise floor). And the
reference saves its top-k logprobs per step: where the server first diverges, its token should be
in that top-k within ``near_tie_nats`` of the argmax. Anything else gets flagged.

The server streams text with no ids, usually one token per chunk but several for a speculative
step or a bursting server, so a chunk's tokens are found from its text: the reference's next tokens
where they decode to it, else the tokenizer (see align_chunks).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import NamedTuple

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families.base import Family, failed_items, write_meta
from benchmark.correctness.report import FamilyScore, band
from benchmark.correctness.sets import SampleSet
from benchmark.utils import clearly_multi_token

TOPK = 8


def save_text_chunks(responses: list[Response], out_dir: Path) -> None:
    for r in responses:
        if r.ok:
            (out_dir / f"{r.item_id}.txt").write_text(r.text, encoding="utf-8")
            chunks = [c.decode("utf-8", errors="replace") for c in r.text_chunks]
            (out_dir / f"{r.item_id}.chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))


def load_chunks(run: Path, item_id: str) -> list[str] | None:
    p = run / "outputs" / f"{item_id}.chunks.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def generate_reference(model, tok, prompt_ids: list[int], max_new_tokens: int, device: str, eos: list[int],
                       **extra) -> dict:
    """Greedy HF generation, keeping the top-k logprobs of every step (raw logits)."""
    import torch

    with torch.inference_mode():
        out = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None, output_logits=True,
                             return_dict_in_generate=True, eos_token_id=eos, pad_token_id=eos[0], **extra)
    gen = out.sequences[0, len(prompt_ids):].tolist()
    topk = []
    for step_logits in out.logits:
        lp = torch.log_softmax(step_logits[0].float(), dim=-1)
        v, k = lp.topk(TOPK)
        topk.append([[int(a), float(b)] for a, b in zip(k.tolist(), v.tolist(), strict=True)])
    stopped = bool(gen) and gen[-1] in eos
    body = gen[:-1] if stopped else gen
    return {"prompt_len": len(prompt_ids), "ids": body, "stopped": stopped, "eos_id": gen[-1] if stopped else None,
            "text": tok.decode(body, skip_special_tokens=True), "topk": topk, "max_new_tokens": max_new_tokens}


class AlignedOutput(NamedTuple):
    """A stream's chunks as tokens: reference-tokenizer ids (negative where a chunk has no id) and each one's text."""

    ids: list[int]
    texts: list[str]


def _reference_run(tok, chunk: str, pos: int, ref_ids: list[int], ends: dict[str, int] | None = None) -> int:
    """How far from ``pos`` the reference's tokens run inside ``chunk``: to the first point they decode to all of it or,
    given the offsets ``ends`` a chunk may be cut at, the last that decodes to the start of it. ``pos`` if neither."""
    end = pos
    for j in range(pos + 1, len(ref_ids) + 1):
        text = tok.decode(ref_ids[pos:j])
        if text == chunk:
            return j
        if len(text) > len(chunk) + 4:  # a partial character can decode to one U+FFFD per byte
            break
        if ends and text in ends:
            end = j
    return end


def _split_chunk(tok, chunk: str, pos: int, ref_ids: list[int]) -> list[int]:
    """The ids of a chunk that is not one token of the reference's top-k: the reference's next tokens as far as they
    run, then the tokenizer on the rest. A partial run has to end on one of the tokenizer's boundaries, or " the" would
    match the start of " then". A run to the chunk's end needn't: it may split "\\n\\n" where the tokenizer wouldn't."""
    enc = tok.encode(chunk, add_special_tokens=False)
    ends = {tok.decode(enc[:k]): k for k in range(1, len(enc) + 1)}
    end = _reference_run(tok, chunk, pos, ref_ids, ends)
    if end == pos:
        return enc or [-1 - pos]
    text = tok.decode(ref_ids[pos:end])
    return ref_ids[pos:end] + ([] if text == chunk else enc[ends[text]:])


def _bytes_of_last_chunk(tok, dec, chunk: str, pos: int, ref_ids: list[int]) -> int:
    """Text that ends inside a character doesn't show how many byte-level tokens it took: " \\xf0" and " \\xf0\\x9f"
    both read " \\ufffd". If ``chunk`` only lines up further on, past such tokens, the last chunk held those too."""
    for skip in range(4):  # a UTF-8 character has at most 4 bytes
        p = pos + skip
        if skip and (p > len(ref_ids) or "\ufffd" not in dec(ref_ids[p - 1])):
            break
        if tok.decode(ref_ids[p:_reference_run(tok, chunk, p, ref_ids)]) == chunk:
            return skip
    return 0


def align_chunks(tok, ref: dict | None, chunks: list[str]) -> AlignedOutput:
    """Turn a stream's chunks into tokens lined up with the reference. A chunk that is one token of the reference's
    next step stays one token, whatever else it could be read as, so a one-token-per-chunk stream lines up chunk i with
    step i."""
    dec = functools.lru_cache(maxsize=None)(lambda t: tok.decode([t]))
    topk = [{int(a): b for a, b in step} for step in ref["topk"]] if ref else []
    ref_ids = ref["ids"] if ref else []
    ids, texts = [], []
    for c in chunks:
        i = len(ids)
        if i < len(ref_ids) and dec(ref_ids[i]) == c:
            ids.append(ref_ids[i])
            texts.append(c)
            continue
        cands = [t for t in (topk[i] if i < len(topk) else {}) if dec(t) == c]
        if cands:
            ids.append(cands[0])
            texts.append(c)
            continue
        if texts and texts[-1].endswith("\ufffd"):
            skip = _bytes_of_last_chunk(tok, dec, c, i, ref_ids)
            ids += ref_ids[i:i + skip]
            texts += [dec(t) for t in ref_ids[i:i + skip]]
            i += skip
        got = _split_chunk(tok, c, i, ref_ids)
        ids += got
        texts += [c] if len(got) == 1 else [dec(t) for t in got]
    return AlignedOutput(ids, texts)


def compare_chunks_to_reference(tok, ref: dict, chunks: list[str], near_tie_nats: float) -> tuple:
    """Compare one prompt's server chunks with the reference. Returns the comparison and the aligned tokens."""
    out = align_chunks(tok, ref, chunks)
    topk = [{int(a): b for a, b in step} for step in ref["topk"]]
    n = ref["max_new_tokens"]
    c = M.compare_greedy(ref["ids"], out.ids, horizon=n, ref_topk=topk, eos_id=ref.get("eos_id") or tok.eos_token_id,
                         ref_stopped=ref["stopped"], out_stopped=len(out.ids) < n, near_tie_nats=near_tie_nats)
    return c, out


def score_greedy_text(sc: FamilyScore, tok, items, refs: dict, runs_chunks: list[dict], near_tie_nats: float,
                      label: str = "") -> list[dict]:
    """Reference and noise-floor checks shared by the text and omni families."""
    rows, comps = [], []
    n = len(items)
    for it in items:
        chunks = runs_chunks[0].get(it.id)
        ref = refs.get(it.id)
        c, out = None, None
        if chunks is not None and ref:
            c, out = compare_chunks_to_reference(tok, ref, chunks, near_tie_nats)
        elif chunks is not None:
            out = align_chunks(tok, None, chunks)
        row = {"id": it.id, "out_tokens": None, "out_chunks": None}
        if out:
            # Past a divergence a chunk's tokens come from the tokenizer, so this has the runner's drift.
            row.update(out_tokens=len(out.ids) if clearly_multi_token(len(out.ids), len(chunks)) else len(chunks),
                       out_chunks=len(chunks))
        if c:
            comps.append(c)
            row.update(ref_tokens=len(ref["ids"]), exact=c.exact, first_div=c.first_divergence, gap=c.logprob_gap,
                       near_tie=c.near_tie, in_topk=c.out_in_ref_topk)
            if c.first_divergence is not None:
                d = c.first_divergence
                row["ref_at_div"] = tok.decode(ref["ids"][max(0, d - 4):d + 4])
                row["out_at_div"] = "".join(out.texts[max(0, d - 4):d + 4])
        if len(runs_chunks) > 1:
            other = runs_chunks[1].get(it.id)
            if out and other is not None:
                row["r1_r2_first_div"] = M.first_divergence(out.texts, align_chunks(tok, ref, other).texts)
        row["out"] = "".join(chunks or [])[:160]
        rows.append(row)
    if comps:
        s = M.summarize_greedy(comps)
        sc.metrics[f"vs_reference{label}"] = s
        unexplained = [it.id for it, r in zip(items, rows, strict=True) if "exact" in r and not r["exact"] and
                       r.get("near_tie") is not True]
        allowed = max(1, n // 16)
        sc.check(f"divergences vs reference that are not near ties{label}", len(unexplained),
                 band(len(unexplained), lambda v: v == 0, lambda v: v <= allowed),
                 f"0 PASS, <={allowed} WARN. Near tie means the server token is in the reference top-{TOPK} "
                 f"within {near_tie_nats} nat of its argmax (provisional)", ", ".join(unexplained[:8]))
        sc.metrics[f"exact_vs_reference{label}"] = f"{s['exact']}/{s['n']}"
    elif refs:
        sc.check(f"agreement with reference{label}", None, "FAIL", "the server produced no output to compare")
    else:
        sc.check(f"agreement with reference{label}", None, "NO_REFERENCE", "needs reference outputs")
    if len(runs_chunks) > 1:
        same = [r.get("r1_r2_first_div") for r in rows if "r1_r2_first_div" in r]
        n_same = sum(d is None for d in same)
        sc.metrics[f"noise_floor_r1_vs_r2{label}"] = {"identical": n_same, "n": len(same),
                                                       "divergence_positions": M.distribution(
                                                           [d for d in same if d is not None])}
        sc.notes.append(f"Noise floor{label}: {n_same}/{len(same)} identical between two server instances, "
                        f"{sc.metrics.get(f'exact_vs_reference{label}', 'n/a')} identical to the reference.")
    return rows


class TextFamily(Family):
    name = "text"
    set_name = "text_v1"
    default_restarts = 2
    default_concurrency = 8
    near_tie_nats = 1.0
    hf_id = ""

    @property
    def thinking(self) -> bool:
        return self.opt("thinking", "0") in ("1", "true", "True")

    # --- prompt construction: identical ids on both sides -------------------------------------
    def prompt_ids(self, tok, prompt: str) -> list[int]:
        # Same as process_prompt in mstar/model/qwen3_5/qwen3_5_model.py.
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=self.thinking)
        return list(tok(text).input_ids)

    def model_kwargs(self, i: int, item) -> dict:
        n = int(item.params.get("max_new_tokens", 128))
        return {"temperature": 0.0, "max_output_tokens": n, "max_tokens": n, "seed": i,
                "enable_thinking": self.thinking}

    def requests(self, sset: SampleSet) -> list[Request]:
        return [Request(it.id, text=it.prompt, output_modalities="text", model_kwargs=self.model_kwargs(i, it))
                for i, it in enumerate(sset.items)]

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        save_text_chunks(responses, out_dir)

    # --- reference ------------------------------------------------------------------------------
    def reference_id(self) -> str:
        return f"hf-{self.hf_id.replace('/', '--')}-greedy-bf16" + ("-think" if self.thinking else "")

    def reference_description(self) -> str:
        return (f"HF transformers {self.hf_id} (bf16, sdpa, eager), greedy, enable_thinking={self.thinking}, "
                f"top-{TOPK} logprobs per step")

    def load_reference_model(self, device: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.hf_id)
        model = AutoModelForImageTextToText.from_pretrained(self.hf_id, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").to(device).eval()
        return tok, model

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        tok, model = self.load_reference_model(device)
        eos = sorted({*_as_list(getattr(model.generation_config, "eos_token_id", None)), tok.eos_token_id} - {None})
        mismatched_encode = []
        for it in sset.items:
            ids = self.prompt_ids(tok, it.prompt)
            if not self.same_as_server_encode(tok, it.prompt, ids):
                mismatched_encode.append(it.id)
            rec = generate_reference(model, tok, ids, int(it.params.get("max_new_tokens", 128)), device, eos)
            (ref_dir / "outputs" / f"{it.id}.json").write_text(json.dumps(rec, ensure_ascii=False))
            print(f"  ref {it.id}: {len(ids)} prompt tokens, {len(rec['ids'])} generated, stopped={rec['stopped']}",
                  flush=True)
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name,
                   hf_id=self.hf_id, topk=TOPK, eos=eos, prompt_ids_differ_from_mstar_encode_ids=mismatched_encode)

    def same_as_server_encode(self, tok, prompt: str, ids: list[int]) -> bool:
        """Whether the server's chunked encoder gives the same ids (Qwen3.5)."""
        return _same_as_mstar_encode(tok, prompt, ids, self)

    # --- scoring --------------------------------------------------------------------------------
    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        from transformers import AutoTokenizer

        sc = FamilyScore(self.name)
        tok = AutoTokenizer.from_pretrained(self.hf_id)
        ref_out = self.ref_outputs(ref_dir)
        refs = {it.id: json.loads((ref_out / f"{it.id}.json").read_text()) for it in sset.items
                if ref_out and (ref_out / f"{it.id}.json").is_file()}
        if not refs:
            sc.notes.append(f"No reference outputs in {ref_dir}.")
        runs_chunks = [{it.id: load_chunks(run, it.id) for it in sset.items} for run in runs]
        fails = failed_items(runs[0])
        n = len(sset.items)
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL", f"all {n} requests return text",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(fails.items())[:3]))
        empty = [k for k, v in runs_chunks[0].items() if v is not None and not "".join(v).strip()]
        sc.check("empty outputs", len(empty), "PASS" if not empty else "FAIL", "0", ", ".join(empty))
        rows = score_greedy_text(sc, tok, sset.items, refs, runs_chunks, self.near_tie_nats)
        needles = []
        for it, row in zip(sset.items, rows, strict=True):
            if "needle" in it.expected:
                row["needle_out"] = it.expected["needle"] in "".join(runs_chunks[0].get(it.id) or [])
                row["needle_ref"] = it.expected["needle"] in refs[it.id]["text"] if it.id in refs else None
                needles.append(row)
        if needles:
            missed = [r["id"] for r in needles if not r["needle_out"] and r["needle_ref"] is not False]
            sc.check("long-context needle retrieved (where the reference retrieves it)",
                     f"{len(needles) - len(missed)}/{len(needles)}", "PASS" if not missed else "FAIL", "all",
                     ", ".join(missed))
        sc.rows = rows
        (work / "text_comparison.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str))
        sc.artifacts += [str(work / "text_comparison.json")] + [str(r / "outputs") for r in runs]
        return sc


class TextImageFamily(TextFamily):
    """Image plus a question on the 20 image items of omni_v1 (--family-opt set=image).

    The server decodes the file with torchvision and runs the checkpoint's image processor on the
    uint8 pixels, so the reference does the same and only the model math differs.
    """

    set_name = "omni_v1"
    default_concurrency = 4

    def select_items(self, items: list) -> list:
        return [it for it in items if it.params.get("modality") == "image"]

    def requests(self, sset: SampleSet) -> list[Request]:
        return [Request(it.id, text=it.prompt, files=[sset.path(it.files["image"])], input_modalities="image,text",
                        output_modalities="text", model_kwargs=self.model_kwargs(i, it))
                for i, it in enumerate(sset.items)]

    def reference_id(self) -> str:
        return f"hf-{self.hf_id.replace('/', '--')}-image-greedy-bf16" + ("-think" if self.thinking else "")

    def reference_description(self) -> str:
        return (f"HF transformers {self.hf_id} (bf16, sdpa, eager) with its processor on torchvision-decoded "
                f"images, greedy, enable_thinking={self.thinking}, top-{TOPK} logprobs per step")

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        import torch
        import torchvision
        from transformers import AutoModelForImageTextToText, AutoProcessor

        proc = AutoProcessor.from_pretrained(self.hf_id)
        tok = proc.tokenizer
        model = AutoModelForImageTextToText.from_pretrained(self.hf_id, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").to(device).eval()
        eos = sorted({*_as_list(getattr(model.generation_config, "eos_token_id", None)), tok.eos_token_id} - {None})
        for it in sset.items:
            # Model.load_image and _as_hwc_uint8 on the server: uint8 CHW from torchvision, then HWC.
            img = torchvision.io.decode_image(str(sset.path(it.files["image"]))).permute(1, 2, 0).contiguous().numpy()
            messages = [{"role": "user", "content": [{"type": "image", "image": ""},
                                                     {"type": "text", "text": it.prompt}]}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=self.thinking)
            inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
            ids = inputs.pop("input_ids")[0].tolist()
            inputs.pop("attention_mask", None)
            extra = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in inputs.items()
                     if hasattr(v, "is_floating_point")}
            rec = generate_reference(model, tok, ids, int(it.params.get("max_new_tokens", 128)), device, eos, **extra)
            (ref_dir / "outputs" / f"{it.id}.json").write_text(json.dumps(rec, ensure_ascii=False))
            print(f"  ref {it.id}: {len(ids)} prompt tokens, {len(rec['ids'])} generated, stopped={rec['stopped']}",
                  flush=True)
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name,
                   hf_id=self.hf_id, topk=TOPK, eos=eos)


def _as_list(v) -> list:
    return v if isinstance(v, list) else ([v] if v is not None else [])


def _same_as_mstar_encode(tok, prompt: str, ids: list[int], fam) -> bool:
    """Check mstar's chunked encoder gives the same ids. True if mstar isn't importable."""
    try:
        from mstar.utils.fast_tokenize import encode_ids
    except ImportError:
        return True
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=fam.thinking)
    return list(encode_ids(tok, text)) == ids
