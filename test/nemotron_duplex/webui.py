#!/usr/bin/env python3
"""Minimal browser UI to talk to a running Nemotron-Duplex (VoiceChat-11B) server.

    # 1) start the model server (see README / launch_server.sh), e.g. on :8019
    # 2) start this UI proxy and open the printed URL:
    python test/nemotron_duplex/webui.py
    python test/nemotron_duplex/webui.py --mstar-url http://127.0.0.1:8019 --port 8500

Record from the mic (or upload a wav) in the browser; this tiny proxy resamples
to 16 kHz mono, appends a silence "reply window" (the model is frame-synchronous
and replies in the frames AFTER you stop talking), forwards to the model server's
/generate with streaming, and streams the agent's text + audio back to the page.

Browser <-> this proxy is same-origin (no CORS); the proxy <-> model server is a
plain server-to-server call. getUserMedia needs a secure context, which includes
http://localhost / http://127.0.0.1 — for remote access put this behind HTTPS.
"""
import argparse
import base64
import io
import json
import sys

from _env import load_env

from mstar.client.client import MStarClient
from mstar.client.types import AudioChunk, TextChunk

MSTAR_URL = "http://127.0.0.1:8019"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Nemotron-Duplex chat</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#111}
 h1{font-size:1.3rem} .row{display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;margin:.6rem 0}
 button{padding:.5rem 1rem;border-radius:.5rem;border:1px solid #ccc;background:#f6f6f6;cursor:pointer;font-size:1rem}
 button.rec{background:#e23;color:#fff;border-color:#e23} button:disabled{opacity:.5;cursor:default}
 #status{color:#555;min-height:1.4em} #transcript{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;
  border-radius:.5rem;padding:.8rem;min-height:3rem;margin:.6rem 0} .muted{color:#888;font-size:.85rem}
 audio{width:100%;margin-top:.4rem}
</style></head><body>
<h1>Nemotron-Duplex — voice chat</h1>
<p class="muted">Speak a short question (or upload a wav), then the agent replies. The reply window
appends silence so the agent has frames to answer in. The agent replies with text and voiced
speech; very dense late sentences of a long reply may thin out.</p>
<div class="row">
 <button id="rec">● Record</button>
 <input type="file" id="file" accept="audio/*">
 <label>reply window <input type="range" id="win" min="5" max="30" value="20"><span id="winv">20</span>s</label>
 <label><input type="checkbox" id="textonly"> text only</label>
</div>
<div id="status">idle</div>
<div id="transcript"></div>
<audio id="player" controls></audio>
<script>
const $=id=>document.getElementById(id);
$("win").oninput=()=>$("winv").textContent=$("win").value;
let ctx,stream,proc,chunks=[],recording=false;

async function startRec(){
  stream=await navigator.mediaDevices.getUserMedia({audio:true});
  ctx=new (window.AudioContext||window.webkitAudioContext)();
  const src=ctx.createMediaStreamSource(stream);
  proc=ctx.createScriptProcessor(4096,1,1);
  chunks=[];
  proc.onaudioprocess=e=>chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(ctx.destination);
  recording=true; $("rec").textContent="■ Stop"; $("rec").classList.add("rec"); $("status").textContent="recording…";
}
async function stopRec(){
  recording=false; $("rec").textContent="● Record"; $("rec").classList.remove("rec");
  proc.disconnect(); stream.getTracks().forEach(t=>t.stop());
  const rate=ctx.sampleRate; await ctx.close();
  let n=chunks.reduce((a,c)=>a+c.length,0), buf=new Float32Array(n), o=0;
  for(const c of chunks){buf.set(c,o);o+=c.length;}
  send(encodeWav(buf,rate));
}
$("rec").onclick=()=>recording?stopRec():startRec();
$("file").onchange=e=>{const f=e.target.files[0]; if(f) send(f);};

function encodeWav(samples,rate){
  const b=new ArrayBuffer(44+samples.length*2), v=new DataView(b);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  ws(0,"RIFF");v.setUint32(4,36+samples.length*2,true);ws(8,"WAVE");ws(12,"fmt ");
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);
  ws(36,"data");v.setUint32(40,samples.length*2,true);
  let o=44;
  for(let i=0;i<samples.length;i++){let s=Math.max(-1,Math.min(1,samples[i]));v.setInt16(o,s*0x7fff,true);o+=2;}
  return new Blob([b],{type:"audio/wav"});
}
function wavFromPcm(bytes,rate){
  const b=new ArrayBuffer(44+bytes.length), v=new DataView(b);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  ws(0,"RIFF");v.setUint32(4,36+bytes.length,true);ws(8,"WAVE");ws(12,"fmt ");
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);
  ws(36,"data");v.setUint32(40,bytes.length,true);
  new Uint8Array(b,44).set(bytes);
  return new Blob([b],{type:"audio/wav"});
}
async function send(blob){
  $("transcript").textContent=""; $("player").removeAttribute("src");
  $("status").textContent="sending…"; $("rec").disabled=true; $("file").disabled=true;
  const fd=new FormData();
  fd.append("audio",blob,"input.wav"); fd.append("reply_window",$("win").value);
  fd.append("text_only",$("textonly").checked?"true":"false");
  let pcmParts=[], sr=22050, spoke=false;
  try{
    const resp=await fetch("/chat",{method:"POST",body:fd});
    const rd=resp.body.getReader(), dec=new TextDecoder(); let carry="";
    while(true){
      const {value,done}=await rd.read(); if(done)break;
      carry+=dec.decode(value,{stream:true});
      let parts=carry.split("\\n\\n"); carry=parts.pop();
      for(const p of parts){
        const line=p.split("\\n").find(l=>l.startsWith("data:")); if(!line)continue;
        const ev=JSON.parse(line.slice(5));
        if(ev.type==="text"){ if(ev.text && !ev.special){ $("transcript").textContent+=ev.text; spoke=true; } }
        else if(ev.type==="audio"){ sr=ev.sr; const bin=atob(ev.pcm); const u=new Uint8Array(bin.length);
          for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);
          pcmParts.push(u); $("status").textContent="agent replying…"; }
        else if(ev.type==="done"){ $("status").textContent="done"; }
        else if(ev.type==="error"){ $("status").textContent="error: "+ev.msg; }
      }
    }
    if(pcmParts.length){
      let n=pcmParts.reduce((a,c)=>a+c.length,0), all=new Uint8Array(n), o=0;
      for(const c of pcmParts){all.set(c,o);o+=c.length;}
      $("player").src=URL.createObjectURL(wavFromPcm(all,sr)); $("player").play().catch(()=>{});
    }
    if(!spoke && !pcmParts.length) $("status").textContent="no reply (try a longer reply window)";
  }catch(e){ $("status").textContent="error: "+e; }
  $("rec").disabled=false; $("file").disabled=false;
}
</script></body></html>"""


def _prep_16k_mono_silence(wav_bytes: bytes, reply_window_s: float) -> bytes:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    d, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if d.ndim == 2:
        d = d.mean(axis=1)
    t = torch.from_numpy(np.ascontiguousarray(d))
    if sr != 16000:
        t = AF.resample(t, sr, 16000)
    out = torch.cat([t, torch.zeros(int(16000 * reply_window_s))]).numpy()
    buf = io.BytesIO()
    sf.write(buf, out, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _is_special(txt: str) -> bool:
    t = txt.strip()
    return t == "" or t.startswith("<SPECIAL") or t in ("</s>", "<s>", "<pad>", "<unk>")


def build_app():
    from fastapi import FastAPI, Form, UploadFile
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI()

    @app.get("/")
    def index():
        return HTMLResponse(PAGE)

    @app.post("/chat")
    async def chat(audio: UploadFile, reply_window: float = Form(20.0), text_only: str = Form("false")):
        raw = await audio.read()

        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        def stream():
            try:
                wav16 = _prep_16k_mono_silence(raw, reply_window)
            except Exception as e:  # noqa: BLE001
                yield sse({"type": "error", "msg": f"audio prep failed: {e}"})
                return
            mods = ("text",) if text_only == "true" else ("text", "audio")
            try:
                c = MStarClient(MSTAR_URL, timeout=600)
                for ev in c.generate(
                    audio=[("input.wav", wav16)], input_modalities=("audio",),
                    output_modalities=mods, temperature=0.0, stream=True,
                ):
                    if isinstance(ev, TextChunk):
                        yield sse({"type": "text", "text": ev.text, "special": _is_special(ev.text)})
                    elif isinstance(ev, AudioChunk):
                        yield sse({"type": "audio", "sr": ev.sample_rate,
                                   "pcm": base64.b64encode(ev.pcm).decode()})
                yield sse({"type": "done"})
            except Exception as e:  # noqa: BLE001
                yield sse({"type": "error", "msg": str(e)})

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main():
    global MSTAR_URL
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mstar-url", default=MSTAR_URL, help="running model server base URL")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8500)
    args = ap.parse_args()
    MSTAR_URL = args.mstar_url

    import uvicorn

    print(f"Nemotron-Duplex web UI -> model server {MSTAR_URL}")
    print(f"open http://{args.host}:{args.port}/  (mic needs localhost or HTTPS)")
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
