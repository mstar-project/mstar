"""Reference-implementation oracle for PR #318's ASR models. Runs in its own venv
(transformers 4.57.6 + qwen-asr 0.0.6); importing anything from mstar (or this
repo's benchmark package) is a hard error, so the oracle owes nothing to the code
it judges (mstar-correctness rule 4).

  ref  greedy reference transcripts of the 200 test-clean utterances in bf16 (the
       reference as users run it) and fp32 (the yardstick), plus the long-form file
       with segment timestamps, word timestamps, language detection and the gallery
       clips. Whisper: transformers WhisperForConditionalGeneration.generate
       (greedy, single beam). Qwen3-ASR: the qwen-asr SDK (Qwen3ASRModel.transcribe,
       transformers backend).
  tf   teacher-forces every collected transcript (<out>/collect/<model>/<tag>.json)
       through the fp32 reference, TF32 off, and writes report.json in the
       mstar-correctness parity.py format (argmax agreement, first disagreement,
       logit gap and rank there, agreement at confident positions = oracle top-2
       margin > 1), plus eos_ok: the oracle's argmax after the last served token is
       end-of-text (a truncated or run-on transcript fails it).

  python oracle.py ref --model turbo --hf openai/whisper-large-v3-turbo --family whisper \
      --data D --out O --device cuda:0
  python oracle.py tf  --model turbo --hf openai/whisper-large-v3-turbo --family whisper \
      --data D --out O --device cuda:0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

if any(m == "mstar" or m.startswith("mstar.") or m == "benchmark" or m.startswith("benchmark.") for m in sys.modules):
    raise SystemExit("oracle.py must not import mstar or the repo's benchmark package")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SR = 16000
# Qwen3-ASR's 30 languages (sglang-omni cookbook / qwen-asr SUPPORTED_LANGUAGES), name -> ISO code
LANG_CODES = {
    "arabic": "ar", "cantonese": "yue", "chinese": "zh", "czech": "cs", "danish": "da", "dutch": "nl",
    "english": "en", "filipino": "fil", "finnish": "fi", "french": "fr", "german": "de", "greek": "el",
    "hindi": "hi", "hungarian": "hu", "indonesian": "id", "italian": "it", "japanese": "ja", "korean": "ko",
    "macedonian": "mk", "malay": "ms", "persian": "fa", "polish": "pl", "portuguese": "pt", "romanian": "ro",
    "russian": "ru", "spanish": "es", "swedish": "sv", "thai": "th", "turkish": "tr", "vietnamese": "vi",
}


def load_wave(path):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def short_set(data: Path) -> list[tuple[str, Path, str]]:
    d = data / "test_clean_200"
    out = []
    for line in (d / "references.tsv").read_text().splitlines():
        if line.strip():
            uid, text = line.split("\t", 1)
            out.append((uid, d / f"{uid}.wav", text))
    return out


def long_form(data: Path) -> tuple[str, Path]:
    d = data / "long_form"
    uid = (d / "references.tsv").read_text().split("\t", 1)[0]
    return uid, d / f"{uid}.wav"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ----------------------------------------------------------------------------- Whisper


class WhisperRef:
    def __init__(self, hf: str, dtype, device: str):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(hf)
        self.tok = self.processor.tokenizer
        self.model = WhisperForConditionalGeneration.from_pretrained(
            hf, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.dtype, self.device = dtype, device
        gc = self.model.generation_config
        self.suppress = list(getattr(gc, "suppress_tokens", None) or [])
        self.begin_suppress = list(getattr(gc, "begin_suppress_tokens", None) or [])
        self.prefix = self.tok.convert_tokens_to_ids(
            ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"])
        self.eos = {self.tok.convert_tokens_to_ids("<|endoftext|>")}

    def feats(self, wave):
        f = self.processor.feature_extractor(wave, sampling_rate=SR, return_tensors="pt").input_features
        return f.to(self.device, self.dtype)

    def greedy(self, waves, language="en", batch=16):
        out = []
        for i in range(0, len(waves), batch):
            f = self.torch.cat([self.feats(w) for w in waves[i:i + batch]])
            with self.torch.inference_mode():
                ids = self.model.generate(f, language=language, task="transcribe", do_sample=False,
                                          num_beams=1, max_new_tokens=440)
            out += [t.strip() for t in self.tok.batch_decode(ids, skip_special_tokens=True)]
        return out

    def detect(self, wave) -> str:
        with self.torch.inference_mode():
            lang = self.model.detect_language(input_features=self.feats(wave))
        return self.tok.convert_ids_to_tokens(int(lang.flatten()[0])).strip("<|>")

    def transcribe_detect(self, wave) -> str:
        with self.torch.inference_mode():
            ids = self.model.generate(self.feats(wave), task="transcribe", do_sample=False, num_beams=1,
                                      max_new_tokens=440)
        return self.tok.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def long_form(self, wave) -> dict:
        inputs = self.processor(wave, sampling_rate=SR, return_tensors="pt", truncation=False,
                                padding="longest", return_attention_mask=True)
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, language="en", task="transcribe", return_timestamps=True,
                                      return_segments=True, do_sample=False, num_beams=1)
        seqs = out["sequences"] if isinstance(out, dict) or hasattr(out, "keys") else out
        text = self.tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
        segs = []
        for s in (out.get("segments") or [[]])[0] if hasattr(out, "get") else []:
            segs.append({"start": float(s["start"]), "end": float(s["end"]),
                         "text": self.tok.decode(s["tokens"], skip_special_tokens=True).strip()})
        return {"text": text, "segments": segs}

    def words(self, wave) -> list[dict]:
        from transformers import pipeline

        pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.tok,
                        feature_extractor=self.processor.feature_extractor, device=self.model.device)
        res = pipe({"raw": wave, "sampling_rate": SR}, return_timestamps="word",
                   generate_kwargs={"language": "en", "task": "transcribe"})
        return [{"word": c["text"].strip(), "start": float(c["timestamp"][0]),
                 "end": float(c["timestamp"][1] if c["timestamp"][1] is not None else c["timestamp"][0])}
                for c in res.get("chunks", [])]

    # teacher forcing --------------------------------------------------------
    def served_ids(self, text: str) -> list[int]:
        return self.tok.encode(" " + text.strip(), add_special_tokens=False) if text.strip() else []

    def full_text(self, text: str) -> str:
        return " " + text.strip()

    def context(self, wave):
        with self.torch.inference_mode():
            return self.model.model.encoder(self.feats(wave)).last_hidden_state

    def logits(self, ctx, ids: list[int]):
        """Logits at the positions that predict ids[0..n-1] and the token after them."""
        torch = self.torch
        dec = torch.tensor([self.prefix + ids], device=self.device)
        with torch.inference_mode():
            out = self.model(encoder_outputs=(ctx,), decoder_input_ids=dec).logits[0].float()
            lg = out[len(self.prefix) - 1:].clone()
            if self.suppress:
                lg[:, self.suppress] = float("-inf")
            if self.begin_suppress:
                lg[0, self.begin_suppress] = float("-inf")
        return lg


# ----------------------------------------------------------------------------- Qwen3-ASR


class QwenRef:
    def __init__(self, hf: str, dtype, device: str):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.torch = torch
        self.asr = Qwen3ASRModel.from_pretrained(hf, dtype=dtype, device_map=device, attn_implementation="sdpa",
                                                 max_inference_batch_size=16, max_new_tokens=512)
        self.model, self.processor = self.asr.model, self.asr.processor
        for gc in (getattr(self.model, "generation_config", None),
                   getattr(self.model.thinker, "generation_config", None)):
            if gc is not None:  # greedy, single beam, whatever the checkpoint's defaults say
                gc.do_sample, gc.num_beams, gc.temperature, gc.top_p, gc.top_k = False, 1, None, None, None
        self.tok = self.processor.tokenizer
        self.dtype, self.device = dtype, device
        self.eos = {151645, 151643}

    def greedy(self, waves, language="English", batch=16):
        res = self.asr.transcribe(audio=[(w, SR) for w in waves], language=language)
        return [r.text.strip() for r in res]

    def detect(self, wave) -> str:
        r = self.asr.transcribe(audio=[(wave, SR)], language=None)[0]
        return LANG_CODES.get((r.language or "").strip().lower(), r.language or "")

    def transcribe_detect(self, wave) -> str:
        return self.asr.transcribe(audio=[(wave, SR)], language=None)[0].text.strip()

    def long_form(self, wave) -> dict:
        r = self.asr.transcribe(audio=[(wave, SR)], language="English")[0]
        return {"text": r.text.strip(), "segments": []}

    def words(self, wave):
        return []

    def served_ids(self, text: str) -> list[int]:
        return self.tok.encode(text.strip(), add_special_tokens=False) if text.strip() else []

    def full_text(self, text: str) -> str:
        return text.strip()

    def context(self, wave):
        prompt = self.asr._build_text_prompt(context="", force_language="English")
        inputs = self.processor(text=[prompt], audio=[wave], return_tensors="pt", padding=True)
        return inputs.to(self.device).to(self.dtype)

    def logits(self, inputs, ids: list[int]):
        torch = self.torch
        base = inputs["input_ids"]
        L = base.shape[1]
        full = torch.cat([base, torch.tensor([ids], device=base.device, dtype=base.dtype)], dim=1)
        kw = {"input_ids": full, "attention_mask": torch.ones_like(full),
              "input_features": inputs["input_features"], "use_cache": False}
        if "feature_attention_mask" in inputs:
            kw["feature_attention_mask"] = inputs["feature_attention_mask"]
        with torch.inference_mode():
            out = self.model.thinker(**kw).logits[0].float()
        return out[L - 1:]


def build(family: str, hf: str, dtype, device: str):
    return WhisperRef(hf, dtype, device) if family == "whisper" else QwenRef(hf, dtype, device)


# ----------------------------------------------------------------------------- phases


def sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cmd_ref(a) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    data, out = Path(a.data), Path(a.out) / "oracle" / a.model
    path = out / "ref.json"
    rec = json.loads(path.read_text()) if path.exists() else {}
    import transformers

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rec.update({"model": a.model, "hf": a.hf, "family": a.family, "transformers": transformers.__version__,
                "torch": torch.__version__, "device": device})
    items = short_set(data)
    waves = [load_wave(p) for _, p, _ in items]
    audio_s = sum(len(w) for w in waves) / SR
    lang = "en" if a.family == "whisper" else "English"
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        if name in rec.get("short", {}):
            print(f"[{a.model}] ref {name}: cached", flush=True)
            continue
        ref = build(a.family, a.hf, dtype, a.device)
        ref.greedy(waves[:4], language=lang)  # warm kernels before timing
        sync()
        t0 = time.perf_counter()
        texts = ref.greedy(waves, language=lang)
        sync()
        dt = time.perf_counter() - t0
        rec.setdefault("short", {})[name] = {uid: t for (uid, _, _), t in zip(items, texts, strict=True)}
        rec.setdefault("timing", {})[name] = {"seconds": dt, "audio_s": audio_s, "rtfx_offline_batch16": audio_s / dt}
        print(f"[{a.model}] ref {name}: {len(items)} utts in {dt:.1f}s "
              f"(offline batch-16 RTFx {audio_s / dt:.0f})", flush=True)
        if name == "fp32":  # the extras from the yardstick precision
            extras = {}
            uid, lpath = long_form(data)
            for key, fn in (
                ("long_form", lambda r=ref, lp=lpath: r.long_form(load_wave(lp))),
                ("detect", lambda r=ref: {c: r.detect(load_wave(p)) for c, p in detect_clips(data, a.assets).items()}),
                ("detect_text", lambda r=ref: {c: r.transcribe_detect(load_wave(p))
                                               for c, p in detect_clips(data, a.assets).items()}),
                ("words", lambda r=ref: {u: r.words(load_wave(p)) for u, p, _ in items[:8]}),
            ):
                try:
                    t0 = time.perf_counter()
                    extras[key] = fn()
                    print(f"[{a.model}] ref {key}: ok ({time.perf_counter() - t0:.1f}s)", flush=True)
                except Exception as e:  # noqa: BLE001 - recorded; the gate reports it as missing
                    extras[key] = {"error": repr(e)}
                    print(f"[{a.model}] ref {key}: FAILED {e!r}", flush=True)
            rec["extras"] = extras
        del ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_json(path, rec)
    write_json(path, rec)
    print(f"wrote {path}")


def detect_clips(data: Path, assets: str) -> dict[str, Path]:
    clips = {Path(p).stem: Path(p) for p in sorted(Path(assets).glob("*.wav"))}
    for uid, p, _ in short_set(data)[:8]:
        clips[uid] = p
    return clips


def cmd_tf(a) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    data, root = Path(a.data), Path(a.out)
    cdir = root / "collect" / a.model
    collected = {f.stem: json.loads(f.read_text()) for f in sorted(cdir.glob("*.json"))
                 if not f.stem.endswith("_extras")}
    if not collected:
        raise SystemExit(f"no collected transcripts in {cdir}")
    items = short_set(data)
    ref = build(a.family, a.hf, torch.float32, a.device)
    import transformers

    report = {"model": a.model, "hf": a.hf, "dtype": "float32", "attn": "sdpa", "tf32": False,
              "transformers": transformers.__version__, "torch": torch.__version__,
              "oracle": {}, "served": {}, "uids": [u for u, _, _ in items]}
    tok = ref.tok
    # incremental: runs already teacher-forced (collect file older than the report) are kept as they are
    path_rep = root / "oracle" / a.model / "report.json"
    if path_rep.exists():
        try:
            old = json.loads(path_rep.read_text())
            mt = path_rep.stat().st_mtime
            keep = {t for t in old.get("served", {}) if t in collected and (cdir / f"{t}.json").stat().st_mtime < mt}
            report["served"] = {t: old["served"][t] for t in keep}
            collected = {t: v for t, v in collected.items() if t not in keep}
            print(f"[{a.model}] tf: kept {sorted(keep)}, forcing {sorted(collected)}", flush=True)
        except (OSError, ValueError):
            pass
    for i, (uid, path, _) in enumerate(items if collected else []):
        ctx = ref.context(load_wave(path))
        for tag, rows in collected.items():
            row = next((r for r in rows if r.get("uid") == uid), None)
            text = (row or {}).get("text") or ""
            full = ref.full_text(text)
            ids = ref.served_ids(text)[:440]
            if not ids:
                report["served"].setdefault(tag, {})[str(i)] = {
                    "uid": uid, "tokens": 0, "argmax_agreement": 0.0, "first_disagreement": 0,
                    "logit_gap_at_first": None, "rank_at_first": None, "confident_positions": 0,
                    "confident_agreement": None, "confident_disagreements": 0, "eos_ok": False,
                    "empty": True, "error": (row or {}).get("error")}
                continue
            # Re-tokenization: adopt the oracle's split where it spells the same text (template rule).
            artifacts = 0
            for _ in range(32):
                lg = ref.logits(ctx, ids)
                top = lg.argmax(-1)
                served = torch.tensor(ids, device=lg.device)
                agree = top[:len(ids)] == served
                realigned = False
                for k in (~agree).nonzero().flatten().tolist():
                    alt_ids = ids[:k] + [int(top[k])]
                    alt = tok.decode(alt_ids)
                    if alt != tok.decode(ids[:k]) and (full.startswith(alt) or alt.startswith(full)):
                        rest = full[len(alt):]
                        ids = (alt_ids + (tok.encode(rest, add_special_tokens=False) if rest else []))[:440]
                        artifacts += 1
                        realigned = True
                        break
                if not realigned:
                    break
            n = len(ids)
            body = lg[:n]
            first = int((~agree).nonzero()[0]) if (~agree).any() else None
            top2 = body.topk(2, dim=-1).values
            confident = (top2[:, 0] - top2[:, 1]) > a.confident_margin
            n_conf = int(confident.sum())
            conf_dis = int((~agree & confident).sum())
            gap = rank = None
            if first is not None:
                r_ = body[first]
                gap = float(r_[top[first]] - r_[served[first]])
                rank = int((r_ > r_[served[first]]).sum())
            eos_ok = int(top[n]) in ref.eos
            report["served"].setdefault(tag, {})[str(i)] = {
                "uid": uid, "tokens": n, "argmax_agreement": float(agree.float().mean()),
                "first_disagreement": first, "logit_gap_at_first": gap, "rank_at_first": rank,
                "retokenization_artifacts": artifacts, "confident_positions": n_conf,
                "confident_agreement": (1.0 - conf_dis / n_conf) if n_conf else None,
                "confident_disagreements": conf_dis, "eos_ok": eos_ok,
                "eos_margin": float(lg[n].max() - lg[n][list(ref.eos)].max()), "prompt_tokens_delta": None}
        if i % 25 == 0:
            print(f"[{a.model}] tf {i + 1}/{len(items)}", flush=True)
    path = root / "oracle" / a.model / "report.json"
    write_json(path, report)
    print(f"\n[{a.model}] | run | argmax agreement | prompts fully argmax | confident disagreements "
          "| eos ok | max gap | max rank |")
    for tag, rows in report["served"].items():
        v = list(rows.values())
        mean = sum(x["argmax_agreement"] for x in v) / len(v)
        full = sum(x["first_disagreement"] is None and not x.get("empty") for x in v)
        cd = sum(x.get("confident_disagreements") or 0 for x in v)
        eos = sum(bool(x.get("eos_ok")) for x in v)
        gaps = [x["logit_gap_at_first"] for x in v if x.get("logit_gap_at_first") is not None]
        ranks = [x["rank_at_first"] for x in v if x.get("rank_at_first") is not None]
        print(f"| {tag} | {mean:.4f} | {full}/{len(v)} | {cd} | {eos}/{len(v)} "
              f"| {max(gaps, default=0):.2f} | {max(ranks, default=0)} |")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ref", "tf"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--family", required=True, choices=("whisper", "qwen"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", default=str(Path(__file__).parent / "assets"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--confident-margin", type=float, default=1.0)
    a = ap.parse_args()
    {"ref": cmd_ref, "tf": cmd_tf}[a.cmd](a)


if __name__ == "__main__":
    main()
