"""Qwen3-Omni: 20 image, 20 audio and 10 video questions, answered in text and speech.

The thinker runs greedy and is compared with the HF thinker using the text family's near-tie check,
split by input modality. The talker samples, so its audio is scored by Whisper against the server's
own text. If the HF processor preprocesses media differently from mstar, that modality will show
early divergences.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families import asr_judge
from benchmark.correctness.families.base import Family, failed_items, write_meta
from benchmark.correctness.families.text import (
    TOPK,
    TextFamily,
    generate_reference,
    load_chunks,
    save_text_chunks,
    score_greedy_text,
)
from benchmark.correctness.report import FamilyScore, band
from benchmark.correctness.sets import SampleSet

SYSTEM = ("You are Qwen, a virtual human developed by the Qwen team, Alibaba Group, capable of perceiving "
          "auditory and visual inputs, as well as generating text and speech.")


class OmniFamily(Family):
    name = "omni"
    set_name = "omni_v1"
    default_restarts = 2
    default_concurrency = 4
    hf_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    max_new_tokens = 128
    near_tie_nats = 1.0

    @property
    def speech(self) -> bool:
        return self.opt("speech", "1") in ("1", "true", "True")

    def requests(self, sset: SampleSet) -> list[Request]:
        reqs = []
        for i, it in enumerate(sset.items):
            mod = it.params["modality"]
            reqs.append(Request(it.id, text=it.prompt, files=[sset.path(it.files[mod])],
                                input_modalities=f"{mod},text",
                                output_modalities="text,audio" if self.speech else "text",
                                model_kwargs={"thinker_temperature": 0.0, "max_output_tokens": self.max_new_tokens,
                                              "seed": i}))
        return reqs

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        save_text_chunks(responses, out_dir)
        for r in responses:
            if r.ok and r.audio:
                M.write_wav(out_dir / f"{r.item_id}.wav", M.pcm16_to_float(bytes(r.audio)), r.audio_sample_rate())

    def reference_id(self) -> str:
        return "hf-qwen3-omni-30b-a3b-thinker-greedy-bf16"

    def reference_description(self) -> str:
        return (f"HF transformers Qwen3OmniMoeForConditionalGeneration {self.hf_id} thinker (bf16, sdpa), "
                f"processor + qwen_omni_utils media, greedy, top-{TOPK} logprobs")

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        proc = Qwen3OmniMoeProcessor.from_pretrained(self.hf_id)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(self.hf_id, dtype=torch.bfloat16,
                                                                     attn_implementation="sdpa").to(device).eval()
        thinker = model.thinker
        tok = proc.tokenizer
        eos = [tok.convert_tokens_to_ids("<|im_end|>")]
        for it in sset.items:
            mod = it.params["modality"]
            path = str(sset.path(it.files[mod]))
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                        {"role": "user", "content": [{"type": mod, mod: path}, {"type": "text", "text": it.prompt}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            inputs = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                          use_audio_in_video=False).to(device)
            ids = inputs.pop("input_ids")[0].tolist()
            extra = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in inputs.items()
                     if hasattr(v, "is_floating_point")}
            rec = generate_reference(thinker, tok, ids, self.max_new_tokens, device, eos, **extra)
            (ref_dir / "outputs" / f"{it.id}.json").write_text(json.dumps(rec, ensure_ascii=False))
            print(f"  ref {it.id}: {len(ids)} prompt tokens, {len(rec['ids'])} generated", flush=True)
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name)

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        from transformers import AutoTokenizer

        sc = FamilyScore(self.name)
        tok = AutoTokenizer.from_pretrained(self.hf_id)
        ref_out = self.ref_outputs(ref_dir)
        refs = {it.id: json.loads((ref_out / f"{it.id}.json").read_text()) for it in sset.items
                if ref_out and (ref_out / f"{it.id}.json").is_file()}
        runs_chunks = [{it.id: load_chunks(run, it.id) for it in sset.items} for run in runs]
        fails = failed_items(runs[0])
        n = len(sset.items)
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL", f"all {n} requests answer",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(fails.items())[:3]))
        rows = []
        for mod in ("image", "audio", "video"):
            items = [it for it in sset.items if it.params["modality"] == mod]
            rows += score_greedy_text(sc, tok, items, refs, runs_chunks, self.near_tie_nats, label=f" [{mod}]")
        if self.speech:
            wavs = [runs[0] / "outputs" / f"{it.id}.wav" for it in sset.items]
            heard = asr_judge.transcribe(wavs, device)
            said = ["".join(runs_chunks[0].get(it.id) or []) for it in sset.items]
            norm = M.make_normalizer("whisper")
            per = []
            bad = []
            for it, w, h, s, row in zip(sset.items, wavs, heard, said, rows, strict=True):
                if not w.is_file():
                    bad.append(it.id)
                    continue
                x, sr = M.read_wav(w)
                if M.audio_problems(M.audio_stats(x, sr)):
                    bad.append(it.id)
                e = M.wer([s], [h], norm)["wer"] if norm(s) else None
                row.update(speech_wer=round(e, 3) if e is not None else None, heard=h[:120])
                if e is not None:
                    per.append(e)
            sc.check("spoken replies failing sanity", len(bad), band(len(bad), lambda v: v == 0, lambda v: v <= 2),
                     "0 PASS, <=2 WARN (finite, voiced, >=0.3 s)", ", ".join(bad[:8]))
            if per:
                med = 100 * statistics.median(per)
                sc.metrics["spoken_reply_wer_vs_own_text"] = M.distribution(per)
                sc.check("median WER of spoken reply vs its own text (%)", round(med, 2),
                         band(med, lambda v: v <= 5, lambda v: v <= 15),
                         "<=5 PASS, <=15 WARN (provisional, from the competitor matrix)")
        sc.rows = rows
        (work / "omni_comparison.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str))
        sc.artifacts += [str(work / "omni_comparison.json")] + [str(r / "outputs") for r in runs]
        return sc


class OmniText(TextFamily):
    """The 32-prompt text set on the Qwen3-Omni thinker (--family-opt set=text)."""

    hf_id = OmniFamily.hf_id
    default_concurrency = 4

    def prompt_ids(self, tok, prompt: str) -> list[int]:
        from transformers import Qwen3OmniMoeProcessor

        proc = Qwen3OmniMoeProcessor.from_pretrained(self.hf_id)
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return list(tok(proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)).input_ids)

    def model_kwargs(self, i: int, item) -> dict:
        n = int(item.params.get("max_new_tokens", 128))
        return {"thinker_temperature": 0.0, "max_output_tokens": n, "seed": i}

    def reference_id(self) -> str:
        return "hf-qwen3-omni-30b-a3b-thinker-text-greedy-bf16"

    def reference_description(self) -> str:
        return f"HF transformers Qwen3OmniMoeForConditionalGeneration {self.hf_id} thinker (bf16, sdpa), greedy"

    def load_reference_model(self, device: str):
        import torch
        from transformers import AutoTokenizer, Qwen3OmniMoeForConditionalGeneration

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(self.hf_id, dtype=torch.bfloat16,
                                                                     attn_implementation="sdpa").to(device).eval()
        return AutoTokenizer.from_pretrained(self.hf_id), model.thinker
