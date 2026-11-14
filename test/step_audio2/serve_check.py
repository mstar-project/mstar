#!/usr/bin/env python3
"""Step-Audio2 served end to end (GPU): a running ``mstar serve step_audio2`` against the oracle.

Run a server first (Python or ``--rust-frontend``; the checks are the same), then
``python test/step_audio2/serve_check.py --port N --out DIR``. Exits nonzero on any failed check.

* **greedy against the oracle**, per recorded case (t2t, s2t, s2s, s2t_long; ``llm_<case>.pt``,
  each the reference's 32 greedy tokens): the request is the case's chat through
  ``POST /generate`` streamed, greedy (temperature 0, no repetition penalty), 32 tokens. A
  streamed text chunk is one decode step's token, so the served reply's chunks line up with
  the oracle's tokens one to one: a text token as its bytes, a speech id or special token as
  nothing. The check is the leading run of steps that agree, against the reference's own
  greedy-run spread (``greedy_spread.pt``), as ``parity_llm.py`` holds the engine-driven LLM.
  A speech id's code is not visible in text, so on s2s only the text steps are compared, and
  the speech ids are counted (each ``<audio_N>`` is a b"" chunk) against the oracle's.
* **text, speech in, speech out**: a sampled request of each kind, with the reference's
  sampling, through ``/generate`` and ``/v1/chat/completions``; a spoken reply's audio is
  saved as WAV, and must be non-silent and last exactly as long as its speech ids say (40 ms
  a code). With ``--asr``, the WAVs are transcribed (``self``: by the served model's own ASR
  prompt; or a local Whisper) next to the reply's text (``--asr self``: the served model's own ASR).
* **streamed and not**: a greedy text reply is the same streamed and not, on ``/generate``
  and on ``/v1/chat/completions``.
* **cancelled spoken replies**: ``--slots`` + 8 spoken replies dropped by the client right
  after their first audio, and a text reply dropped mid-stream; then ``--slots`` spoken
  replies at once must all get audio, so no cancelled reply kept a vocoder slot.
* **edges**: ``max_output_tokens`` 1 and 2 return that many steps; a malformed field and a
  ``ref_audio`` are 400s, and the server serves normally afterwards.
* **two identical seeded spoken requests**: whether their text and audio match (reported).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import ASSETS, CKPT, ORACLE, Verdict  # noqa: E402

CLIP = ASSETS / "give_me_a_brief_introduction_to_the_great_wall.wav"
LONG_CLIP = ASSETS / "paralinguistic_information_understanding.wav"
SYSTEM = "You are a helpful assistant."
GREEDY = {"temperature": 0.0, "repetition_penalty": 1.0}
SAMPLE_RATE = 24000
SECONDS_PER_CODE = 0.04
# (text, audio clip or None, input_modalities, spoken): the oracle's prompts, the system
# message being the model's default
CASES = {
    "t2t": ("Give me a brief introduction to the Great Wall.", None, ["text"], False),
    "s2t": (None, CLIP, ["audio"], False),
    "s2s": (None, CLIP, ["audio"], True),
    "s2t_long": ("Describe this audio.", LONG_CLIP, ["text", "audio"], False),
}


class Reply:
    def __init__(self, chunks: list[dict]):
        self.text_chunks = [base64.b64decode(c["data"]) for c in chunks if c.get("modality") == "text"]
        self.audio = b"".join(base64.b64decode(c["data"]) for c in chunks if c.get("modality") == "audio")
        self.errors = [c for c in chunks if "modality" not in c]

    @property
    def text(self) -> str:
        return b"".join(self.text_chunks).decode("utf-8", "replace")

    @property
    def pcm(self) -> np.ndarray:
        return np.frombuffer(self.audio, dtype="<i2")


def generate_form(text, clip, inputs, spoken, streaming=True, **kwargs) -> tuple[dict, list | None]:
    data = {
        "output_modalities": "text,audio" if spoken else "text",
        "input_modalities": ",".join(inputs),
        "streaming": "true" if streaming else "false",
        "model_kwargs": json.dumps(kwargs),
    }
    if text is not None:
        data["text"] = text
    files = [("files", (clip.name, clip.read_bytes()))] if clip is not None else None
    return data, files


def generate(base: str, text, clip, inputs, spoken, streaming=True, **kwargs) -> tuple[int, Reply | str]:
    data, files = generate_form(text, clip, inputs, spoken, streaming, **kwargs)
    with requests.post(f"{base}/generate", data=data, files=files, stream=streaming, timeout=600) as r:
        if r.status_code != 200:
            return r.status_code, r.text
        if not streaming:
            outputs = r.json()["outputs"]
            return 200, Reply([{"modality": m, **c} for m, chunks in outputs.items() for c in chunks])
        return 200, Reply([json.loads(line) for line in r.iter_lines() if line])


def dropped(base: str, text, clip, inputs, spoken, after: str, **kwargs) -> bool:
    """Drop the connection right after the first chunk of modality ``after``; whether one came."""
    data, files = generate_form(text, clip, inputs, spoken, **kwargs)
    with requests.post(f"{base}/generate", data=data, files=files, stream=True, timeout=600) as r:
        for line in r.iter_lines():
            if line and json.loads(line).get("modality") == after:
                return True
    return False


def chat_streamed(base: str, messages, **kwargs) -> tuple[int, str]:
    body = {"model": "step_audio2", "messages": messages, "stream": True, **kwargs}
    with requests.post(f"{base}/v1/chat/completions", json=body, stream=True, timeout=600) as r:
        if r.status_code != 200:
            return r.status_code, r.text
        parts = []
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices", []):
                parts.append((choice.get("delta") or {}).get("content") or "")
        return 200, "".join(parts)


def chat(base: str, messages, spoken: bool, **kwargs) -> tuple[int, dict | str]:
    body = {"model": "step_audio2", "messages": messages, **kwargs}
    if spoken:
        body["modalities"] = ["text", "audio"]
    r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=600)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


def save_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def agreement(a, b) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def speech_ids(steps: int, stopped: bool) -> int:
    """A spoken reply's speech ids, off the reference's interleave (one text slot, then four
    speech ids): a ``<tts_pad>`` text slot decodes to nothing as a speech id does, so the
    text alone can't tell them apart. A reply that ended on its own streams its stop step
    too, which is neither."""
    steps -= int(stopped)
    return steps - -(-steps // 5)


def check_audio(verdict: Verdict, name: str, pcm: np.ndarray, codes: int) -> None:
    """The vocoder voices every speech id, 40 ms each, to the sample."""
    rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768.0) ** 2))) if len(pcm) else 0.0
    want = round(codes * SECONDS_PER_CODE * SAMPLE_RATE)
    verdict.check(len(pcm) == want and rms > 0.005,
                  f"{name}: {len(pcm)} samples ({len(pcm) / SAMPLE_RATE:.2f} s) for {codes} speech ids "
                  f"({want} samples), RMS {rms:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--asr", default=None,
                    help="'self' to transcribe the WAVs with the server's own ASR, or a local Whisper checkpoint")
    ap.add_argument("--skip-chat", action="store_true")
    ap.add_argument("--slots", type=int, default=32,
                    help="spoken replies sent at once after the cancels: at least the config's vocoder slots")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    verdict = Verdict()

    from transformers import Qwen2TokenizerFast

    from mstar.model.step_audio2.config import is_speech_id
    from mstar.model.step_audio2.step_audio2_model import SpokenTextDetokenizer

    detok = SpokenTextDetokenizer(Qwen2TokenizerFast.from_pretrained(str(CKPT)))
    spread = torch.load(ORACLE / "greedy_spread.pt", weights_only=False)

    # -- greedy against the oracle ---------------------------------------------------------
    report = {}
    for case, (text, clip, inputs, spoken) in CASES.items():
        want = torch.load(ORACLE / f"llm_{case}.pt", weights_only=False)["greedy"].tolist()
        status, reply = generate(base, text, clip, inputs, spoken, max_output_tokens=len(want), **GREEDY)
        if status != 200:
            verdict.check(False, f"{case}: greedy request answered {status}: {reply}")
            continue
        # the reply's last token is a stop id when the oracle's is: dropped from the text
        steps = [detok.token_bytes(t) for t in want]
        got = reply.text_chunks
        if spoken:
            # text steps only: a speech id is b"" on both sides
            text_steps = [i for i, t in enumerate(want) if not is_speech_id(t)]
            same = agreement([got[i] if i < len(got) else None for i in text_steps], [steps[i] for i in text_steps])
            lead = text_steps[same] if same < len(text_steps) else len(want)
            speech = sum(1 for c in got if c == b"")
            print(f"{case}: {len(got)} steps, text steps agree through step {lead} of {len(want)}; "
                  f"{speech} empty (speech) steps vs the oracle's {sum(map(is_speech_id, want))}")
        else:
            lead = agreement(got, steps)
            if lead == len(got) and len(got) == len(want) - 1 and want[-1] in (151665, 151694, 151643):
                lead = len(want)  # stopped on the oracle's stop id
        floor = min(agreement(spread[case][a], spread[case][b]) for a, b in (("math", "fp32"),))
        floor = min(floor, agreement(spread[case]["math"], want), agreement(spread[case]["fp32"], want))
        report[case] = (lead, len(want), floor)
        verdict.check(lead >= min(floor, len(want)),
                      f"{case}: served greedy agrees with the oracle for {lead} of {len(want)} steps (the reference's "
                      f"own greedy runs agree for {floor}); served text {reply.text!r}")

    # -- sampled text, speech in, speech out ---------------------------------------------------
    for name, (text, clip, inputs, spoken) in {
        "t2t_sampled": CASES["t2t"], "s2t_sampled": CASES["s2t"], "s2s_sampled": CASES["s2s"],
        "t2s_sampled": (CASES["t2t"][0], None, ["text"], True),
    }.items():
        cap = 1024
        status, reply = generate(base, text, clip, inputs, spoken, seed=11, max_output_tokens=cap)
        if status != 200:
            verdict.check(False, f"{name}: answered {status}: {reply}")
            continue
        print(f"{name}: {reply.text!r}")
        verdict.check(bool(reply.text.strip()) and not reply.errors, f"{name}: a text reply, no error chunk")
        if spoken:
            save_wav(args.out / f"{name}.wav", reply.audio)
            (args.out / f"{name}.txt").write_text(reply.text)
            steps = len(reply.text_chunks)
            check_audio(verdict, name, reply.pcm, speech_ids(steps, stopped=steps < cap))

    # -- the OpenAI chat route -----------------------------------------------------------------
    if not args.skip_chat:
        audio_b64 = base64.b64encode(CLIP.read_bytes()).decode()
        user_audio = {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
        ]}
        status, body = chat(base, [{"role": "system", "content": SYSTEM}, user_audio], spoken=True,
                            seed=11, max_tokens=512, audio={"voice": "default_male"})
        ok = status == 200 and isinstance(body, dict)
        if ok:
            message = body["choices"][0]["message"]
            audio = (message.get("audio") or {}).get("data")
            print(f"chat s2s: {message.get('content')!r}")
            if audio:
                wav = base64.b64decode(audio)
                (args.out / "chat_s2s_male.wav").write_bytes(wav)
                with wave.open(io.BytesIO(wav)) as w:
                    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                rms = float(np.sqrt(np.mean((pcm / 32768.0) ** 2)))
                ok = len(pcm) > SAMPLE_RATE and rms > 0.005
                print(f"chat s2s audio: {len(pcm) / SAMPLE_RATE:.2f} s, RMS {rms:.3f}")
            else:
                ok = False
        verdict.check(ok, f"chat s2s with voice default_male: {status}")
        status, body = chat(base, [{"role": "user", "content": "Say hello in one word."}], spoken=False, max_tokens=8,
                            temperature=0)
        verdict.check(status == 200 and body["choices"][0]["message"]["content"].strip() != "",
                      f"chat t2t: {status} {body if status != 200 else body['choices'][0]['message']['content']!r}")

    # -- streamed and not -------------------------------------------------------------------------
    prompt = "Give me a brief introduction to the Great Wall."
    replies = [generate(base, prompt, None, ["text"], False, streaming=streaming, max_output_tokens=64, **GREEDY)
               for streaming in (True, False)]
    ok = all(status == 200 for status, _ in replies) and replies[0][1].text == replies[1][1].text
    verdict.check(ok, f"/generate greedy text is the same streamed and not: "
                      f"{[r.text if status == 200 else r for status, r in replies]}")
    if not args.skip_chat:
        messages = [{"role": "user", "content": prompt}]
        status, body = chat(base, messages, spoken=False, max_tokens=64, temperature=0, repetition_penalty=1.0)
        streamed_status, streamed = chat_streamed(base, messages, max_tokens=64, temperature=0,
                                                  repetition_penalty=1.0)
        whole = body["choices"][0]["message"]["content"] if status == 200 else body
        verdict.check(status == 200 and streamed_status == 200 and whole == streamed,
                      f"chat greedy text is the same streamed and not: {whole!r} / {streamed!r}")

    # -- cancelled spoken replies --------------------------------------------------------------------
    cancels = args.slots + 8
    with ThreadPoolExecutor(max_workers=cancels + 1) as pool:
        spoken = [pool.submit(dropped, base, None, CLIP, ["audio"], True, "audio", seed=i, max_output_tokens=400)
                  for i in range(cancels)]
        text = pool.submit(dropped, base, prompt, None, ["text"], False, "text", max_output_tokens=400)
        heard = sum(f.result() for f in spoken)
        verdict.check(heard == cancels and text.result(),
                      f"{heard} of {cancels} spoken replies got audio before the client dropped them")
    time.sleep(2)  # the server sees the disconnects
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.slots) as pool:
        after = list(pool.map(
            lambda i: generate(base, None, CLIP, ["audio"], True, seed=100 + i, max_output_tokens=200),
            range(args.slots),
        ))
    voiced = sum(1 for status, r in after if status == 200 and len(r.pcm) > 0)
    verdict.check(voiced == args.slots, f"after the cancels, {voiced} of {args.slots} spoken replies at once got "
                                        f"audio ({time.monotonic() - start:.1f} s)")

    # -- edges ------------------------------------------------------------------------------------
    for n in (1, 2):
        for spoken in (False, True):
            status, reply = generate(base, "Count to ten.", None, ["text"], spoken, max_output_tokens=n, **GREEDY)
            verdict.check(status == 200 and len(reply.text_chunks) == n,
                          f"max_output_tokens={n}{' spoken' if spoken else ''}: {status}, "
                          f"{len(reply.text_chunks) if status == 200 else reply} steps")
    status, body = generate(base, "hi", None, ["text"], False, temperature="hot")
    verdict.check(status == 400, f"a malformed temperature is a 400: {status} {body}")
    status, body = generate(base, "hi", None, ["text"], True, ref_audio="data:audio/wav;base64,AAAA")
    verdict.check(status == 400 and "ref_audio" in str(body), f"ref_audio is refused: {status} {body}")
    status, body = generate(base, "hi", None, ["text"], True, voice="nobody")
    verdict.check(status == 400, f"an unknown voice is a 400: {status} {body}")
    status, reply = generate(base, "Give me a brief introduction to the Great Wall.", None, ["text"], False,
                             max_output_tokens=8, **GREEDY)
    verdict.check(status == 200 and reply.text.startswith("The Great Wall"), f"serves after the refusals: {status}")

    # -- two identical seeded spoken requests ------------------------------------------------------
    pair = [generate(base, None, CLIP, ["audio"], True, seed=5, max_output_tokens=400) for _ in range(2)]
    if all(s == 200 for s, _ in pair):
        (_, a), (_, b) = pair
        same_text, same_audio = a.text == b.text, a.audio == b.audio
        first = next((i for i, (x, y) in enumerate(zip(a.pcm, b.pcm, strict=False)) if x != y), None)
        print(f"seeded pair: text identical {same_text}, audio identical {same_audio} "
              f"({len(a.pcm)} vs {len(b.pcm)} samples, first differing sample {first})")
        save_wav(args.out / "seeded_a.wav", a.audio)
        save_wav(args.out / "seeded_b.wav", b.audio)
    else:
        verdict.check(False, f"seeded pair: {[s for s, _ in pair]}")

    if args.asr:
        # Step-Audio2's own ASR (the reference's ``asr_test`` prompt), greedy, on the server
        # under test, unless a local Whisper checkpoint is named
        pipe = None
        if args.asr != "self":
            from transformers import pipeline

            pipe = pipeline("automatic-speech-recognition", model=args.asr, device=0)
        for wav in sorted(args.out.glob("*.wav")):
            if pipe is not None:
                heard = pipe(str(wav))["text"]
            else:
                audio = base64.b64encode(wav.read_bytes()).decode()
                status, body = chat(base, [
                    {"role": "system", "content": "请记录下你所听到的语音内容。"},
                    {"role": "user", "content": [
                        {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                    ]},
                ], spoken=False, temperature=0, max_tokens=512, repetition_penalty=1.0)
                heard = body["choices"][0]["message"]["content"] if status == 200 else f"<{status}: {body}>"
            said = (args.out / f"{wav.stem}.txt").read_text() if (args.out / f"{wav.stem}.txt").exists() else ""
            print(f"ASR {wav.name}: {heard!r}\n    reply text: {said!r}")
            if said:
                (args.out / f"{wav.stem}.asr.txt").write_text(heard)
    print("\ngreedy agreement (steps agreeing, of, the reference's own spread):", report)
    verdict.finish()


if __name__ == "__main__":
    main()
