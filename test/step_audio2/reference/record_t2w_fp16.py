"""The reference's own float16 token2wav (``Token2wav(float16=True)``) streamed over the
recorded windows, as the yardstick for a served path that runs the DiT in float16: same
voice, same flow noise buffer (cast), the CUDA RNG seeded 0 once before the stream.

Usage: python -I record_t2w_fp16.py <github/Step-Audio2> <hf/Step-Audio-2-mini> <oracle/rec>
Writes <rec>/token2wav_stream_fp16.pt: window_mels, window_pcm.
"""
import os
import sys

ref_dir, model_dir, rec_dir = sys.argv[1:4]
sys.path.insert(0, ref_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf_shim  # noqa: E402,F401
import torch  # noqa: E402
from token2wav import Token2wav  # noqa: E402

rec = torch.load(os.path.join(rec_dir, "token2wav_stream.pt"))
t2w = Token2wav(os.path.join(model_dir, "token2wav"), float16=True)
t2w.flow.decoder.rand_noise = rec["rand_noise"].to("cuda", t2w.flow.decoder.rand_noise.dtype)
prompt_wav = rec["prompt_wav"]
with torch.inference_mode():
    # set_stream_cache runs outside the autocast that stream wraps the flow in
    # (token2wav.py:106 vs :129), and float16=True fails there on a float x-vector
    with torch.amp.autocast("cuda", dtype=torch.float16):
        t2w.set_stream_cache(prompt_wav)
    mels = []
    orig = t2w.flow.inference_chunk

    def spy(**kw):
        mel, cache = orig(**kw)
        mels.append(mel.float().cpu())
        return mel, cache

    t2w.flow.inference_chunk = spy
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    pcm = [t2w.stream(list(codes), prompt_wav=prompt_wav, last_chunk=last) for codes, last in rec["windows"]]
print("mel max abs vs fp32", max((a - b).abs().max().item() for a, b in zip(mels, rec["window_mels"], strict=True)))
torch.save({"window_mels": mels, "window_pcm": pcm}, os.path.join(rec_dir, "token2wav_stream_fp16.pt"))
