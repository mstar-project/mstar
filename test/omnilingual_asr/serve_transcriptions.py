"""Check a running M* Omnilingual ASR server against offline transcripts.

Start the server first, e.g. ``python -m mstar.cli.main serve omnilingual_asr_ctc_300m --port 8000``,
then::

    python test/omnilingual_asr/serve_transcriptions.py --url http://127.0.0.1:8000 \\
        --model omnilingual_asr_ctc_300m --manifest clips/manifest.json --expected parity.json \\
        --checkpoint omniASR-CTC-300M-v2.pt --long over_45s.wav over_75s.wav --out served.json

``--expected`` is ``run_parity.py``'s output for the same dtype and device as the
server; its per-clip M* transcripts are what ``/v1/audio/transcriptions`` must
return. Checks:

* every clip, one request at a time, equals the offline transcript;
* the same clip twice gives byte-identical responses;
* all clips at once (``--concurrency``), so the encoder batches them, equal the
  one-at-a-time responses;
* ``stream=true`` emits deltas whose concatenation, and the ``done`` event,
  equal the non-streamed text; ``verbose_json`` reports one segment;
* each ``--long`` upload (over 20 s) is served as windows: its text equals
  the offline transcripts of the same windows (cut by the route's own
  ``planner_for``) joined by a space.
"""

import argparse
import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch


def post(url: str, model: str, path: str, **fields) -> requests.Response:
    with open(path, "rb") as f:
        return requests.post(
            f"{url}/v1/audio/transcriptions",
            data={"model": model, **fields},
            files={"file": (Path(path).name, f, "audio/wav")},
            timeout=600,
        )


def transcribe(url: str, model: str, path: str) -> str:
    r = post(url, model, path)
    r.raise_for_status()
    return r.json()["text"]


def offline_windows(paths: list[str], checkpoint: str, tokenizer: str, size: str, dtype: str) -> dict[str, str]:
    """What the server should return for over-40 s uploads: the route's own
    windows, each transcribed alone by the offline model, joined by a space."""
    from mstar.api_server.openai.adapters import OmnilingualASRAdapter
    from mstar.api_server.openai.serving_transcriptions import planner_for
    from mstar.model.omnilingual_asr import audio, config, weight_loader
    from mstar.model.omnilingual_asr import tokenizer as tok_mod
    from mstar.model.omnilingual_asr.components.wav2vec2 import Wav2Vec2CTC

    dt = config.parse_dtype(dtype)
    if dt == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False
    with torch.device("meta"):
        model = Wav2Vec2CTC(config.ctc_config(size, "v2"))
    weight_loader.load_into(
        model, weight_loader.convert_state_dict(weight_loader.read_checkpoint(checkpoint), dt, "cuda")
    )
    tok = tok_mod.CharTokenizer.from_file(tokenizer)
    out = {}
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
        for path in paths:
            # the planner deletes its input on cleanup, as the route deletes the upload
            copy = Path(tmp) / Path(path).name
            shutil.copyfile(path, copy)
            planner = planner_for(OmnilingualASRAdapter(), str(copy), Path(tmp))
            texts, windows = [], planner.fixed()
            for window in windows:
                wave = audio.preprocess(audio.load_waveform(window.path))
                logits, frames = model(wave[None].to("cuda", dt))
                texts.append(tok.decode(tok_mod.ctc_greedy_ids(logits[0], int(frames[0])).tolist()).strip())
            out[path] = {
                "text": " ".join(t for t in texts if t),
                "windows": [round(w.duration, 2) for w in planner.fixed()],
            }
            planner.cleanup()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--long", nargs="*", default=[])
    ap.add_argument("--checkpoint", help="for the --long expectation")
    ap.add_argument("--tokenizer", help="for the --long expectation")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    size = args.model.rsplit("_", 1)[1]
    expected = {r["path"]: r["text"].strip() for r in json.load(open(args.expected))["rows"]}
    paths = [c["path"] for c in json.load(open(args.manifest)) if c["path"] in expected]
    report: dict = {"model": args.model, "clips": len(paths)}

    sequential = {p: transcribe(args.url, args.model, p) for p in paths}
    mism = [p for p in paths if sequential[p] != expected[p]]
    report["sequential_match_offline"] = f"{len(paths) - len(mism)}/{len(paths)}"
    report["sequential_mismatches"] = [{"path": p, "served": sequential[p], "offline": expected[p]} for p in mism]

    a = post(args.url, args.model, paths[0]).content
    b = post(args.url, args.model, paths[0]).content
    report["repeat_byte_identical"] = a == b

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        concurrent = dict(zip(paths, pool.map(lambda p: transcribe(args.url, args.model, p), paths), strict=True))
    report["concurrent_seconds"] = round(time.time() - t0, 3)
    diff = [p for p in paths if concurrent[p] != sequential[p]]
    report[f"concurrent{args.concurrency}_match_sequential"] = f"{len(paths) - len(diff)}/{len(paths)}"
    report["concurrent_mismatches"] = [
        {"path": p, "concurrent": concurrent[p], "sequential": sequential[p]} for p in diff
    ]

    r = post(args.url, args.model, paths[0], stream="true")
    deltas, done = [], None
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data: ") and line != "data: [DONE]":
            event = json.loads(line[6:])
            if event["type"] == "transcript.text.delta":
                deltas.append(event["delta"])
            elif event["type"] == "transcript.text.done":
                done = event["text"]
    report["stream_matches"] = done == sequential[paths[0]] and "".join(deltas).strip() == sequential[paths[0]]
    verbose = post(args.url, args.model, paths[0], response_format="verbose_json").json()
    report["verbose_json_segments"] = len(verbose.get("segments", []))
    report["verbose_json_text_matches"] = verbose.get("text") == sequential[paths[0]]

    if args.long:
        want = offline_windows(args.long, args.checkpoint, args.tokenizer, size, args.dtype)
        report["long"] = []
        for path in args.long:
            served = transcribe(args.url, args.model, path)
            report["long"].append(
                {
                    "path": path,
                    "windows_s": want[path]["windows"],
                    "matches_offline_windows": served == want[path]["text"],
                    "served": served,
                    "offline": want[path]["text"],
                }
            )

    print(json.dumps(report, indent=1, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
