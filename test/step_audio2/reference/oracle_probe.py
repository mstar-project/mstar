"""Greedy probe of the Step-Audio-2-mini reference: token pattern, timings, wavs.

Usage: python -I oracle_probe.py <github/Step-Audio2> <hf/Step-Audio-2-mini> <oracle>/probe
"""
import json
import os
import sys
import time

ref_dir, model_dir, out_dir = sys.argv[1:4]
sys.path.insert(0, ref_dir)
os.makedirs(out_dir, exist_ok=True)

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf_shim  # noqa: E402,F401
from stepaudio2 import StepAudio2  # noqa: E402
from token2wav import Token2wav  # noqa: E402

A = os.path.join(ref_dir, "assets")
model = StepAudio2(model_dir)
t2w = Token2wav(os.path.join(model_dir, "token2wav"))
sys_msg = {"role": "system", "content": "You are a helpful assistant."}
great_wall = os.path.join(A, "give_me_a_brief_introduction_to_the_great_wall.wav")
cases = {
    "t2t": [sys_msg, {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
            {"role": "assistant", "content": None}],
    "t2s": [sys_msg, {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
            {"role": "assistant", "content": "<tts_start>", "eot": False}],
    "s2t": [sys_msg, {"role": "human", "content": [{"type": "audio", "audio": great_wall}]},
            {"role": "assistant", "content": None}],
    "s2s": [sys_msg, {"role": "human", "content": [{"type": "audio", "audio": great_wall}]},
            {"role": "assistant", "content": "<tts_start>", "eot": False}],
}
report = {}
for name, msgs in cases.items():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    toks, text, audio = model(msgs, max_new_tokens=1024, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    # run-length pattern of text (T) vs audio (A) vs special (S) tokens
    kinds = ["A" if t > 151695 else ("S" if t >= 151688 or t in (151665, 151643) else "T") for t in toks]
    runs = []
    for k in kinds:
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    report[name] = {"n_tokens": len(toks), "seconds": dt, "text": text, "n_audio": len(audio),
                    "runs": runs[:40], "specials": sorted({t for t in toks if 151688 <= t <= 151695}),
                    "tokens": toks}
    print(name, len(toks), f"{dt:.2f}s", text[:200].replace("\n", " "), "runs:", runs[:16])
    if audio:
        t0 = time.perf_counter()
        wav = t2w(audio, prompt_wav=os.path.join(A, "default_female.wav"))
        torch.cuda.synchronize()
        report[name]["t2w_seconds"] = time.perf_counter() - t0
        with open(os.path.join(out_dir, f"{name}.wav"), "wb") as f:
            f.write(wav)
json.dump(report, open(os.path.join(out_dir, "probe.json"), "w"), indent=1, ensure_ascii=False)
