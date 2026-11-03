"""Test-only stand-in for a transcription server, so run.py's plumbing (server.sh, client,
content checks, collect, stress, realtime, cells, summary, gate inputs) can be exercised on a
CPU box: ASR318_SELFTEST=1 makes every engine launch this instead. It answers with the
LibriSpeech reference of the uploaded file (found by file name), so WER is ~0.

  python selftest_server.py --port 8100 --data ~/asr318_data
"""
import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import WSMsgType, web


def refs(data: Path) -> dict[str, str]:
    out = {}
    for tsv in data.glob("*/references.tsv"):
        for line in tsv.read_text().splitlines():
            if line.strip():
                uid, text = line.split("\t", 1)
                out[uid] = text
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    table = refs(Path(a.data))
    order = list(table)

    async def health(_):
        return web.Response(text="ok")

    async def transcribe(request):
        form = await request.post()
        f = form.get("file")
        if f is None:
            return web.json_response({"error": "no file"}, status=400)
        if form.get("max_output_tokens") not in (None, "") and not str(form.get("max_output_tokens")).isdigit():
            return web.json_response({"error": "max_output_tokens must be an int"}, status=400)
        stem = Path(f.filename).stem
        text = table.get(stem, "hello world" if stem != "asr_zh" else "ni hao").capitalize() + "."
        lang = "zh" if stem == "asr_zh" else "en"
        await asyncio.sleep(0.01)
        if form.get("response_format") == "verbose_json":
            words = [{"word": w, "start": 0.3 * i, "end": 0.3 * i + 0.25} for i, w in enumerate(text.split())]
            body = {"text": text, "language": lang, "duration": 0.3 * len(words),
                    "segments": [{"id": 0, "start": 0.0, "end": 0.3 * len(words), "text": text}]}
            if "word" in form.getall("timestamp_granularities[]", []):
                body["words"] = words
            return web.json_response(body)
        return web.json_response({"text": text})

    async def realtime(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        vllm = "intent" not in request.query
        await ws.send_str(json.dumps({"type": "session.created" if vllm else "transcription_session.created"}))
        nbytes, sent = 0, 0
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                break
            ev = json.loads(msg.data)
            if ev["type"] == "input_audio_buffer.append":
                nbytes += len(ev["audio"]) * 3 // 4
                if not vllm and nbytes // 64000 > sent:
                    sent = nbytes // 64000
                    await ws.send_str(json.dumps({"type": "mstar.transcription.partial", "text": "partial",
                                                  "audio_seconds": nbytes / 32000}))
            elif ev["type"] == "input_audio_buffer.commit" and (not vllm or ev.get("final")):
                text = table[order[0]] if order else "hello"
                if vllm:
                    await ws.send_str(json.dumps({"type": "transcription.delta", "delta": text}))
                    await ws.send_str(json.dumps({"type": "transcription.done", "text": text}))
                else:
                    await ws.send_str(json.dumps({"type": "conversation.item.input_audio_transcription.completed",
                                                  "transcript": text}))
                break
        await ws.close()
        return ws

    app = web.Application(client_max_size=1024 ** 3)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    app.router.add_get("/v1/realtime", realtime)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
