"""Compare the M* Omnilingual ASR CTC port against ``record_oracle.py``'s outputs.

Per clip: the port's own preprocessing (decode, resample, downmix, normalize)
against the reference's normalized waveform, the logits (max abs diff), the
argmax ids, the transcript, and CER against the manifest's ``transcription``
for both sides. Then the clips in padded batches of 4 against the reference,
and negative controls that must diverge: no waveform normalization, padding
without the attention mask, the v1 tokenizer on v2 weights, and the
positional conv's weight norm left unfolded.

The port's modules are imported without the ``mstar`` package (its
``__init__`` pulls in the engine), so the same script also runs in the
reference's venv, where the same torch build makes the comparison bit exact::

    python test/omnilingual_asr/run_parity.py --ref ref_300m_fp32.pt --size 300m \\
        --checkpoint omniASR-CTC-300M-v2.pt --tokenizer omniASR_tokenizer_written_v2.model \\
        --v1-tokenizer omniASR_tokenizer.model --manifest clips/manifest.json --device cuda --out parity.json
"""

import argparse
import importlib
import json
import re
import sys
import time
import types
import unicodedata
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def import_port():
    pkg = "mstar.model.omnilingual_asr"
    for name in ("mstar", "mstar.model", pkg, f"{pkg}.components"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(REPO / name.replace(".", "/"))]
            sys.modules[name] = module
    return types.SimpleNamespace(
        **{
            short.split(".")[-1]: importlib.import_module(f"{pkg}.{short}")
            for short in ("config", "audio", "tokenizer", "weight_loader", "components.wav2vec2")
        }
    )


def norm_text(s: str) -> str:
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s.lower())
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyps: list[str], refs: list[str]) -> float:
    pairs = [(norm_text(h), norm_text(r)) for h, r in zip(hyps, refs, strict=True)]
    return sum(edit_distance(h, r) for h, r in pairs) / max(1, sum(len(r) for _, r in pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--v1-tokenizer", default=None, help="for the wrong-tokenizer control")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    port = import_port()
    dtype = port.config.parse_dtype(args.dtype)
    device = torch.device(args.device)
    if device.type == "cuda" and dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False  # full fp32, as the reference
    ref = torch.load(args.ref, weights_only=False)
    records = ref["records"]
    clips = [c for c in json.load(open(args.manifest)) if c["path"] in records]

    raw_state = port.weight_loader.read_checkpoint(args.checkpoint)
    with torch.device("meta"):
        model = port.wav2vec2.Wav2Vec2CTC(port.config.ctc_config(args.size, args.version))
    port.weight_loader.load_into(model, port.weight_loader.convert_state_dict(raw_state, dtype, device))
    model.eval()
    tok = port.tokenizer.CharTokenizer.from_file(args.tokenizer)

    def run(waves, mask=True):
        lens = [w.shape[0] for w in waves]
        batch = torch.zeros(len(waves), max(lens))
        for i, w in enumerate(waves):
            batch[i, : w.shape[0]] = w
        with torch.inference_mode():
            logits, frames = model(batch.to(device, dtype), lens if mask else None)
        return logits.float().cpu(), frames

    def transcribe(logits, n, tokenizer=tok):
        return tokenizer.decode(port.tokenizer.ctc_greedy_ids(logits, n).tolist())

    rows, waves = [], {}
    t0 = time.time()
    for clip in clips:
        rec = records[clip["path"]]
        wave = port.audio.preprocess(port.audio.load_waveform(clip["path"]))
        waves[clip["path"]] = wave
        logits, frames = run([wave])
        n = int(frames[0])
        text = transcribe(logits[0], n)
        same_shape = wave.shape == rec["wave"].shape and n == rec["logits"].shape[0]
        rows.append(
            {
                "path": clip["path"],
                "lang": clip.get("lang"),
                "frames": n,
                "wave_max_abs_diff": float((wave - rec["wave"]).abs().max()) if same_shape else None,
                "logits_max_abs_diff": float((logits[0, :n] - rec["logits"]).abs().max()) if same_shape else None,
                "argmax_mismatch_frames": int((logits[0, :n].argmax(-1) != rec["ids"]).sum()) if same_shape else None,
                "text_match": text == rec["text"],
                "text": text,
                "ref_text": rec["text"],
            }
        )
    elapsed = time.time() - t0

    scored = [(r, c["transcription"]) for r, c in zip(rows, clips, strict=True) if c.get("transcription")]
    summary = {
        "size": args.size,
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "reference": ref["meta"],
        "clips": len(rows),
        "languages": sorted({r["lang"] for r in rows if r["lang"]}),
        "text_exact_match": sum(r["text_match"] for r in rows),
        "argmax_identical_clips": sum(r["argmax_mismatch_frames"] == 0 for r in rows),
        "max_logits_abs_diff": max(r["logits_max_abs_diff"] for r in rows),
        "max_wave_abs_diff": max(r["wave_max_abs_diff"] for r in rows),
        "cer_mstar": cer([r["text"] for r, _ in scored], [t for _, t in scored]),
        "cer_reference": cer([r["ref_text"] for r, _ in scored], [t for _, t in scored]),
        "elapsed_s": elapsed,
    }

    paths = [c["path"] for c in clips]
    match, diff = 0, 0.0
    for i in range(0, len(paths), 4):
        group = paths[i : i + 4]
        logits, frames = run([waves[p] for p in group])
        for j, p in enumerate(group):
            n = int(frames[j])
            match += transcribe(logits[j], n) == records[p]["text"]
            diff = max(diff, float((logits[j, :n] - records[p]["logits"]).abs().max()))
    summary["batched4_text_match"] = match
    summary["batched4_max_logits_abs_diff"] = diff

    if not args.no_controls:
        controls, sub = {}, paths[:12]
        m = 0
        for p in sub:
            logits, frames = run([port.audio.load_waveform(p)])
            m += transcribe(logits[0], int(frames[0])) == records[p]["text"]
        controls["skip_normalization"] = f"{m}/{len(sub)} transcripts match"
        m = 0
        for i in range(0, len(sub), 4):
            group = sub[i : i + 4]
            logits, _ = run([waves[p] for p in group], mask=False)
            frames = model.frame_lengths([waves[p].shape[0] for p in group])
            m += sum(transcribe(logits[j], int(frames[j])) == records[p]["text"] for j, p in enumerate(group))
        controls["padding_without_mask"] = f"{m}/{len(sub)} transcripts match"
        if args.v1_tokenizer:
            tok_v1 = port.tokenizer.CharTokenizer.from_file(args.v1_tokenizer)
            m = raised = 0
            for p in sub:
                try:
                    m += transcribe(records[p]["logits"], records[p]["logits"].shape[0], tok_v1) == records[p]["text"]
                except IndexError:  # an id past the 9812-piece v1 vocabulary
                    raised += 1
            controls["v1_tokenizer_on_v2"] = f"{m}/{len(sub)} match ({raised} hit ids past the v1 vocabulary)"
        conv = model.encoder_frontend.pos_encoder.conv
        folded = conv.weight
        conv.weight = torch.nn.Parameter(
            raw_state["encoder_frontend.pos_encoder.conv.weight_v"].to(device, dtype),
            requires_grad=False,
        )
        m, d = 0, 0.0
        for p in sub:
            logits, frames = run([waves[p]])
            n = int(frames[0])
            m += transcribe(logits[0], n) == records[p]["text"]
            d = max(d, float((logits[0, :n] - records[p]["logits"]).abs().max()))
        conv.weight = folded
        controls["weight_norm_not_folded"] = f"{m}/{len(sub)} transcripts match, logits max abs diff {d:.3g}"
        summary["negative_controls"] = controls

    print(json.dumps(summary, indent=1, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), ensure_ascii=False, indent=1)
    for r in rows:
        if not r["text_match"]:
            print("MISMATCH", r["path"], "\n  reference:", r["ref_text"], "\n  M*       :", r["text"])


if __name__ == "__main__":
    main()
