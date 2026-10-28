"""Record Meta's reference outputs for the Omnilingual ASR parity clips.

Runs in a separate venv with the reference installed (``omnilingual-asr==0.2.0``
pins torch 2.8 and fairseq2 0.6, so it cannot share M*'s)::

    uv venv -p 3.12 refvenv && VIRTUAL_ENV=refvenv uv pip install omnilingual-asr==0.2.0 \\
        torchaudio==2.8.0 "pyarrow>=20,<21" soundfile

fairseq2 needs libsndfile (``apt install libsndfile1``). Point a fairseq2 asset
card at local files with ``FAIRSEQ2_USER_ASSET_DIR`` to skip its download, e.g.
a card ``local_omniASR_CTC_300M_v2`` with ``model_family: wav2vec2_asr``,
``model_arch: 300m_v2``, ``checkpoint: file:///.../omniASR-CTC-300M-v2.pt`` and
``tokenizer_ref`` to a ``char_tokenizer`` card; or pass Meta's card name
(``omniASR_CTC_300M_v2``) and let fairseq2 fetch it.

For every clip in the manifest (a JSON list of ``{"path", "lang", ...}``) it
saves the pipeline's normalized waveform, the CTC logits over the valid frames
(float32), their argmax and the pipeline's own ``transcribe`` text at batch 1;
it also transcribes every clip at batch 2 to check the reference's own batch
invariance::

    python test/omnilingual_asr/record_oracle.py --card omniASR_CTC_300M_v2 \\
        --dtype float32 --device cuda --manifest clips/manifest.json --out ref_300m_fp32.pt
"""

import argparse
import json
import time

import torch
from fairseq2.nn import BatchLayout
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    pipe = ASRInferencePipeline(model_card=args.card, device=device, dtype=dtype)
    clips = json.load(open(args.manifest))
    paths = [c["path"] for c in clips]
    records = {}
    t0 = time.time()
    with torch.inference_mode():
        for clip in clips:
            path = clip["path"]
            wave = next(iter(pipe._build_audio_wavform_pipeline([path]).and_return()))
            batch = pipe._create_batch_simple([(wave, None)])
            layout = BatchLayout(batch.source_seqs.shape, seq_lens=batch.source_seq_lens, device=device)
            logits, out_layout = pipe.model(batch.source_seqs, layout)
            text = pipe.transcribe([path], batch_size=1)[0]
            assert text == pipe._apply_model_wav2vec2asr(batch)[0]
            n = int(out_layout.seq_lens[0])
            records[path] = {
                "wave": wave.float().cpu(),
                "logits": logits[0, :n].float().cpu(),
                "ids": logits[0, :n].argmax(-1).cpu(),
                "text": text,
            }
            print(f"{clip.get('lang', '?')} {n:4d} frames | {text[:80]}", flush=True)
        elapsed = time.time() - t0
        batched = pipe.transcribe(paths, batch_size=2)
    meta = {
        "card": args.card,
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "elapsed_s": elapsed,
        "batch2_mismatches": [p for p, t in zip(paths, batched, strict=True) if t != records[p]["text"]],
    }
    torch.save({"meta": meta, "records": records}, args.out)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
