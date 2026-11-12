"""The media the Phi-4-multimodal recordings use, into ``$PHI4MM_ORACLE/media``.

From the Hugging Face cache: LibriSpeech test-clean (a 14 s utterance, and the first
utterances to reach 46 s, which the conformer unfolds into windows) and a Food-101
image (as is, 384 x 512, and resized to 1500 x 900 for twelve crops), plus the bagel
image in this repo. Run in the serving venv (needs pyarrow); then ``record*.py`` in a
venv with the reference's transformers.
"""

import glob
import io
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from PIL import Image

OUT = Path(os.environ["PHI4MM_ORACLE"]) / "media"
HUB = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"))
REPO = Path(__file__).resolve().parents[2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    speech = glob.glob(f"{HUB}/datasets--openslr--librispeech_asr/snapshots/*/clean/test/0000.parquet")[0]
    rows = pq.read_table(speech, columns=["audio", "text"]).slice(0, 30).to_pylist()
    clips = [sf.read(io.BytesIO(r["audio"]["bytes"])) for r in rows]
    sf.write(OUT / "speech_short.wav", *clips[1])
    (OUT / "speech_short.txt").write_text(rows[1]["text"])
    long, texts = [], []
    for (wav, sr), row in zip(clips, rows, strict=True):
        long.append(wav)
        texts.append(row["text"])
        if sum(len(w) for w in long) / sr > 45:
            break
    sf.write(OUT / "speech_long.wav", np.concatenate(long), sr)
    (OUT / "speech_long.txt").write_text(" ".join(texts))
    food = glob.glob(f"{HUB}/datasets--ethz--food101/snapshots/*/data/train-00000-of-00008.parquet")[0]
    image = Image.open(io.BytesIO(pq.read_table(food).slice(0, 1).to_pylist()[0]["image"]["bytes"])).convert("RGB")
    image.save(OUT / "food_small.png")
    image.resize((1500, 900), Image.BICUBIC).save(OUT / "food_wide.png")
    Image.open(REPO / "test/manual/bagel/bagel.png").convert("RGB").save(OUT / "bagel.png")


if __name__ == "__main__":
    main()
