#!/usr/bin/env python3
"""Text parity of the served Nemotron-Duplex path against the standalone
``offline_inference`` (verified against the NeMo reference in fp32).

    # serve configs/nemotron_duplex.yaml, then on the same node (server running):
    python test/nemotron_duplex/parity_compare.py --audio user.wav --url http://127.0.0.1:8019 --precision bf16
    # or compare a saved engine text (from a previous /generate) without a server:
    python test/nemotron_duplex/parity_compare.py --audio user.wav --engine-text engine.txt --precision fp32

``--precision bf16`` runs the standalone path with bf16 weights under autocast,
the engine's serving precision: the two must agree token for token. ``fp32``
is the reference-verified oracle; it can legitimately differ from a bf16 run
where the text channel's decision is a knife-edge, which the printed top-2
logit margins show (e.g. the agent's turn start: <s> vs <SPECIAL_12>).

Loads the full model on the GPU (fp32 ~44 GB, bf16 ~22 GB): with a bf16 server
resident (~31 GB) prefer ``--precision bf16``. offline_inference is O(T^2) in
frames, so keep clips to a few seconds.
"""
import argparse
import json
import sys

import torch

import mstar.model.nemotron_duplex.nemotron_duplex_model as duplex
from mstar.model.registry import HF_MODELS


def load_wav_16k_mono(path: str) -> torch.Tensor:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    a = torch.from_numpy(data)
    if a.dim() == 2:
        a = a.mean(dim=-1)
    if sr != 16000:
        import torchaudio.functional as AF

        a = AF.resample(a, sr, 16000)
    return a


def engine_text_from_server(url: str, audio: str) -> str:
    from mstar.client.client import MStarClient

    r = MStarClient(url, timeout=900).generate(
        audio=audio, input_modalities=("audio",), output_modalities=("text",), temperature=0.0,
    )
    return r.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="16 kHz mono user clip (the same file sent to the server)")
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    ap.add_argument("--url", default=None, help="server to query for the engine text")
    ap.add_argument("--engine-text", default=None, help="file holding a previous /generate text instead of --url")
    ap.add_argument("--margins", default=None, help="write per-frame top-2 logits (JSON) here")
    ap.add_argument("--show", default="0-", help="frame range of margins to print, e.g. 36-58")
    args = ap.parse_args()
    if bool(args.url) == bool(args.engine_text):
        ap.error("give exactly one of --url / --engine-text")

    engine_text = open(args.engine_text).read() if args.engine_text else engine_text_from_server(args.url, args.audio)

    # Record the text channel's top-2 logits per frame around the sampler.
    records: list[dict] = []
    sample = duplex._sample_text_token

    def recording_sampler(logits, generated, step, *a, **kw):
        top = torch.topk(logits[0].float(), 2)
        records.append({"frame": step, "ids": top.indices.tolist(), "logits": top.values.tolist(),
                        "margin": float(top.values[0] - top.values[1])})
        return sample(logits, generated, step, *a, **kw)

    duplex._sample_text_token = recording_sampler
    try:
        model = duplex.NemotronDuplexModel(model_path_hf=HF_MODELS["nemotron_duplex"]["model_path_hf"])
        dtype = torch.bfloat16 if args.precision == "bf16" else None
        for node in ("conformer_encoder", "nano_llm"):
            model.get_submodule(node, device="cuda", autocast_dtype=dtype)
        wav = load_wav_16k_mono(args.audio).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            out = model.offline_inference(
                wav.unsqueeze(0), torch.tensor([wav.shape[0]], device="cuda"),
                device="cuda", temperature=0.0, decode_audio=False,
            )
    finally:
        duplex._sample_text_token = sample

    toks = out["tokens_text"][0].tolist()
    tok = model.tokenizer
    # the engine's text is the concatenation of per-token decodes
    standalone_text = "".join(tok.decode([t]) for t in toks)

    lo, _, hi = args.show.partition("-")
    lo, hi = int(lo or 0), int(hi or 10**9)
    hi = min(hi, len(toks) - 1)
    print(f"standalone {args.precision}: {len(toks)} frames; margins (argmax -> runner-up) frames {lo}..{hi}:")
    for r in records:
        if lo <= r["frame"] <= hi:
            a, b = r["ids"]
            print(f"  {r['frame']:3d}: {tok.decode([a])!r:16s} -> {tok.decode([b])!r:16s} margin={r['margin']:.3f}")
    smallest = sorted(records, key=lambda r: r["margin"])[:5]
    print("5 smallest margins:", [(r["frame"], round(r["margin"], 3)) for r in smallest])
    if args.margins:
        json.dump({"tokens": toks, "text": standalone_text, "margins": records}, open(args.margins, "w"))

    same = engine_text == standalone_text
    print(f"engine text == standalone {args.precision} text: {same}")
    if not same:
        pairs = zip(engine_text, standalone_text, strict=False)
        n = next((i for i, (a, b) in enumerate(pairs) if a != b), min(len(engine_text), len(standalone_text)))
        print(f"first difference at char {n}:")
        print(f"  engine:     {engine_text[n:n + 80]!r}")
        print(f"  standalone: {standalone_text[n:n + 80]!r}")
    print("standalone reply:", standalone_text.replace("<SPECIAL_12>", "").strip())
    print("engine reply:    ", engine_text.replace("<SPECIAL_12>", "").strip())
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
