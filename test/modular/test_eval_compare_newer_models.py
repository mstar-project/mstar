"""benchmark.eval compare on CPU: the manifest entries for the newer models (#577) and the ``world`` client kind
(image to world video along a seeded camera path), run against a stdlib server that validates requests the way
LingBot-World's ``process_prompt`` does."""

from __future__ import annotations

import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

pytest.importorskip("aiohttp")

from benchmark.eval.e2e import arm, checks, compare, report  # noqa: E402
from benchmark.eval.e2e import manifest as mf  # noqa: E402
from benchmark.eval.suites import e2e  # noqa: E402

NEW = {"fish_s2_pro": "fish_s2_pro", "cosyvoice3": "cosyvoice3", "voxcpm2": "voxcpm2",
       "omnilingual_asr_ctc_300m": "omnilingual_asr_ctc_300m",
       "lingbot_world_1_3b": "lingbot_world_1_3b", "minicpm_o": "minicpm_o"}


def test_newer_models_are_optional_entries_picked_by_their_own_tag():
    m = mf.load()
    picked = m.select("tag:new-model")
    assert {x.name: x.serve for x in picked} == NEW
    assert all(x.optional for x in picked)
    in_all = {x.name for x in m.select("all")}
    assert not in_all & set(NEW) and len(in_all) == 15
    # a shared tag does not pull them in; a name does
    assert not {x.name for x in m.select("tag:tts")} & set(NEW)
    assert [x.name for x in m.select("voxcpm2")] == ["voxcpm2"]
    cases = {x.name: [c.id for c in x.cases] for x in picked}
    assert cases == {"fish_s2_pro": ["tts", "clone"], "cosyvoice3": ["clone"],
                     "voxcpm2": ["tts", "clone", "design"],
                     "omnilingual_asr_ctc_300m": ["upload"], "lingbot_world_1_3b": ["world"],
                     "minicpm_o": ["text", "image_in", "audio_in", "speech_out"]}
    fish = m.models["fish_s2_pro"].cases
    assert fish[0].data == "tts" and fish[1].data == "clone" and "--clone" in fish[1].args
    assert fish[1].quality == ["mos", "speaker_sim"]
    clone = m.models["cosyvoice3"].cases[0]
    assert clone.data == "clone" and "--clone" in clone.args and clone.check == {"kind": "wer", "max_increase": 2.0}
    world = m.models["lingbot_world_1_3b"].cases[0]
    assert world.kind == "world" and world.check["kind"] == "psnr" and world.conc == [1]
    mini = m.models["minicpm_o"]
    assert mini.config == "configs/minicpm_o.yaml"
    assert all(c.kind == "gen_omni" for c in mini.cases)
    assert mini.cases[-1].quality == ["mos"] and "--audio-out" in mini.cases[-1].args


def test_camera_path_is_a_fixed_rigid_trajectory():
    still = np.asarray(e2e.camera_path(5, 0.05, 0.0))
    assert still.shape == (5, 4, 4)
    assert np.allclose(still[0], np.eye(4))
    assert np.allclose(still[:, 2, 3], [0.0, 0.05, 0.10, 0.15, 0.20])      # forward along +z
    turn = np.asarray(e2e.camera_path(e2e.WORLD_FRAMES, 0.05, 1.5))
    for p in turn:
        r = p[:3, :3]
        assert np.allclose(r @ r.T, np.eye(3)) and math.isclose(np.linalg.det(r), 1.0)
        assert np.allclose(p[3], [0, 0, 0, 1])
    assert turn.tolist() == e2e.camera_path(e2e.WORLD_FRAMES, 0.05, 1.5)   # same request every time


def _suite(tmp_path: Path, kind: str) -> e2e.E2ESuite:
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    (media / "i2v.png").write_bytes(b"\x89PNG fake")
    args = Namespace(kind=kind, media=str(media), max_tokens=8, audio_out=False, size="832x480", extra=[],
                     sample_rate=24000, timeout=30.0, model_name="m", data=None, limit=None)
    return e2e.E2ESuite(args)


def _fields(form) -> dict[str, object]:
    return {f[0]["name"]: f[2] for f in form._fields}


def test_world_request_carries_prompt_image_and_camera_path(tmp_path):
    s = _suite(tmp_path, "world")
    f = _fields(s._form(1, e2e.Item(id="001")))
    assert f["files"] == b"\x89PNG fake" and f["text"] == e2e.WORLD_PROMPT
    assert f["input_modalities"] == "image,text" and f["output_modalities"] == "video_frame"
    kw = json.loads(f["model_kwargs"])
    poses = np.asarray(kw["camera_poses"])
    assert poses.shape == (e2e.WORLD_FRAMES, 4, 4) and len(poses) >= 13          # LingBot's minimum
    assert kw["num_frames"] == e2e.WORLD_FRAMES and len(kw["intrinsics"]) == 4 and kw["seed"] == 7
    assert _fields(s._form(1, e2e.Item(id="001")))["model_kwargs"] == f["model_kwargs"]   # seeded


FAKE_WORLD = r'''
import json, sys, base64
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        msg = BytesParser(policy=default).parsebytes(
            b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
        f = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True) for p in msg.iter_parts()}
        kw = json.loads(f["model_kwargs"])
        poses = kw.get("camera_poses") or []
        ok = (f.get("text") and f.get("files") and f["output_modalities"] == b"video_frame" and len(poses) >= 13
              and all(len(p) == 4 and all(len(r) == 4 for r in p) for p in poses) and len(kw["intrinsics"]) == 4)
        if not ok:   # LingBot-World refuses such a request
            self.send_response(400); self.send_header("Content-Length", "3"); self.end_headers()
            self.wfile.write(b"bad"); return
        self.send_response(200); self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close"); self.end_headers()
        frame = bytes([int(10 * abs(p[0][2]) * 255) % 256 for p in poses]) * 16   # depends on the camera path
        for _ in range(2):
            self.wfile.write(json.dumps({"modality": "video_frame", "data": base64.b64encode(frame).decode()}).encode()
                             + b"\n"); self.wfile.flush()
        self.close_connection = True


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


def test_world_kind_runs_end_to_end_and_compares_frames(tmp_path):
    from benchmark.eval.cli import main
    from benchmark.eval.schema import RunResult

    server = tmp_path / "world.py"
    server.write_text(FAKE_WORLD)
    _suite(tmp_path, "world")   # writes the media
    outs = {}
    for label in ("base", "head"):
        out = tmp_path / label
        main(["run", "e2e", "--kind", "world", "--media", str(tmp_path / "media"), "--serve-cmd",
              f"{sys.executable} {server} {{port}}", "--limit", "4", "--concurrency", "1", "--repeats", "1",
              "--warmup", "0", "--label", label, "--out", str(out)])
        res = RunResult.load(out / f"e2e-{label}-c1.json")
        assert res.summary["errors"]["mean"] == 0, res.repeats[0].requests
        outs[label] = checks.saved_dir(out, 1)
        assert sorted(p.name for p in outs[label].iterdir()) == ["000.u8", "001.u8", "002.u8", "003.u8"]
    facts = checks.saved(outs["base"], outs["head"])
    assert facts["psnr_min_db"] == math.inf and facts["only_base"] == []


def test_world_case_command_and_list(tmp_path, capsys):
    """What compare would run for the LingBot case (no GPU), and --list marks the entries optional."""
    m = mf.load()
    case = m.models["lingbot_world_1_3b"].cases[0].to_dict()
    arm.job.update({"python": "py", "data": str(tmp_path), "datasets": m.datasets})
    try:
        cmds = arm.case_cmds(case, "http://h:1", "lingbot-world-base", tmp_path / "c")
    finally:
        arm.job.clear()
    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd[:5] == ["py", "-m", "benchmark.eval", "run", "e2e"]
    assert cmd[cmd.index("--kind") + 1] == "world" and cmd[cmd.index("--media") + 1] == str(tmp_path / "media")
    assert cmd[cmd.index("--limit") + 1] == "4" and cmd[cmd.index("--concurrency") + 1] == "1"
    args = Namespace(cases=str(mf.DEFAULT), list=True, render=None)
    assert compare.cmd_compare(args) == 0
    listed = {ln.split()[0] for ln in capsys.readouterr().out.splitlines() if ln.strip()}
    assert {f"{n}+" for n in NEW} <= listed


FAKE_MINICPM_GENERATE = r'''
import base64, json, sys
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")

    def do_POST(self):
        if self.path != "/generate":
            self.send_error(404); return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        content_type = self.headers["Content-Type"]
        if content_type.startswith("multipart/form-data"):
            msg = BytesParser(policy=default).parsebytes(
                b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
            fields = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                      for p in msg.iter_parts()}
        else:
            fields = {name: values[0].encode() for name, values in parse_qs(body.decode()).items()}
        layout = fields.get("input_modalities", b"").decode()
        output = fields.get("output_modalities", b"").decode()
        kwargs = json.loads(fields.get("model_kwargs", b"{}"))
        valid = (layout in ("text", "image,text", "audio,text") and output in ("text", "text,audio")
                 and fields.get("text") and kwargs.get("temperature") == 0.0
                 and kwargs.get("max_output_tokens") in (64, 96) and isinstance(kwargs.get("seed"), int))
        if layout == "image,text":
            valid = valid and fields.get("files", b"").startswith(b"\x89PNG")
        elif layout == "audio,text":
            valid = valid and fields.get("files", b"").startswith(b"RIFF")
        else:
            valid = valid and "files" not in fields
        if not valid:
            self.send_error(400, "wrong native generate request"); return
        self.send_response(200); self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close"); self.end_headers()
        reply = b"hi \xf0" if len(sys.argv) > 2 else "hi 🌍".encode()
        for chunk in (reply[:5], reply[5:]):  # split one UTF-8 code point across chunks
            self.wfile.write(json.dumps({"modality": "text", "data": base64.b64encode(chunk).decode()}).encode()
                             + b"\n")
        if output == "text,audio":
            for chunk in (bytes(32), bytes(32)):
                self.wfile.write(json.dumps({"modality": "audio", "data": base64.b64encode(chunk).decode()}).encode()
                                 + b"\n")
        self.wfile.flush(); self.close_connection = True


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


@pytest.mark.parametrize("modality,speech", [("text", False), ("image", False), ("audio", False),
                                             ("text", True)])
def test_minicpm_native_generate_cases_run_and_save_text_and_speech(tmp_path, modality, speech):
    from benchmark.eval.cli import main
    from benchmark.eval.schema import RunResult

    server = tmp_path / "generate.py"
    server.write_text(FAKE_MINICPM_GENERATE)
    media = tmp_path / "media"
    media.mkdir()
    (media / "image.png").write_bytes(b"\x89PNG wire test")
    (media / "audio.wav").write_bytes(b"RIFF wire test")
    out = tmp_path / "out"
    cmd = ["run", "e2e", "--kind", "gen_omni", "--input-modality", modality,
           "--media", str(media), "--serve-cmd", f"{sys.executable} {server} {{port}}",
           "--limit", "2", "--concurrency", "1", "--repeats", "1", "--warmup", "0",
           "--label", "mini", "--save-audio", "last", "--out", str(out)]
    if speech:
        cmd += ["--audio-out", "--max-tokens", "64"]
    main(cmd)
    result = RunResult.load(out / "e2e-mini-c1.json")
    assert result.summary["errors"]["mean"] == 0, result.repeats[0].requests
    assert all(r["text"] == "hi 🌍" and r["n_chunks"] == 2 for r in result.repeats[0].requests)
    saved = checks.saved_dir(out, 1)
    assert (saved / "000.txt").read_text() == "hi 🌍"
    if speech:
        assert (saved / "000.pcm").read_bytes() == bytes(64)
    else:
        assert not (saved / "000.pcm").exists()


def test_minicpm_native_text_replaces_an_unfinished_utf8_tail(tmp_path):
    from benchmark.eval.cli import main
    from benchmark.eval.schema import RunResult

    server = tmp_path / "generate.py"
    server.write_text(FAKE_MINICPM_GENERATE)
    out = tmp_path / "out"
    main(["run", "e2e", "--kind", "gen_omni", "--serve-cmd", f"{sys.executable} {server} {{port}} truncated",
          "--limit", "1", "--concurrency", "1", "--repeats", "1", "--warmup", "0",
          "--label", "mini", "--out", str(out)])
    result = RunResult.load(out / "e2e-mini-c1.json")
    assert result.summary["errors"]["mean"] == 0
    assert result.repeats[0].requests[0]["text"] == "hi \ufffd"


def test_minicpm_latency_compares_replies_of_matching_length(tmp_path):
    case = mf.load().models["minicpm_o"].cases[0]
    files = {}
    for arm_name, seconds, tokens in (("base", 1.0, 96), ("head", 2.0, 192)):
        path = tmp_path / f"{arm_name}.json"
        path.write_text(json.dumps({"repeats": [{"wall_s": 3.0, "requests": [
            {"id": str(i), "ok": True, "e2e_s": seconds, "n_chunks": tokens} for i in range(8)
        ]}]}))
        files[arm_name] = [path]
    latency = report.Perf("lat_p50_s", 1, report.Stat.of([1.0]), report.Stat.of([2.0]), True)
    adjusted, _ = report.length_aware(case, files, {"base": [], "head": []}, latency, None)
    assert adjusted is not None
    assert adjusted.metric == "s_per_tok_p50"
    assert adjusted.base.mean == adjusted.head.mean
