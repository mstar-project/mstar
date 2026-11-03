"""Full-Duplex-Bench v1.5 suite (benchmark/eval) on CPU: the metric definitions on tiny
fixtures (pooled latency, label shares, A/F/U quality and coverage, joint, agreement,
kappa, confusion matrix), the judge rules, and the suite end to end against a stdlib fake
duplex server and two fake judges. The official sources are replaced by test doubles, so
nothing is downloaded and no GPU is needed."""

from __future__ import annotations

import base64
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

pytest.importorskip("aiohttp")

from benchmark.eval import fdb_reference, llm_judge, stats  # noqa: E402
from benchmark.eval.audio import read_wav, speech_segments, write_wav  # noqa: E402
from benchmark.eval.datasets import Item, save_manifest  # noqa: E402
from benchmark.eval.schema import RunResult  # noqa: E402
from benchmark.eval.suites import duplex_fdb  # noqa: E402
from benchmark.eval.suites.duplex_fdb import afu, fdb_metrics, label_shares, pool_intervals  # noqa: E402

R, S, U, K = llm_judge.C_LABELS  # C_RESPOND, C_RESUME, C_UNCERTAIN_HANDLING, C_UNKNOWN


def test_pooled_latency_is_the_mean_over_every_interval():
    sessions = [[[0.0, 1.0], [2.0, 2.5]], [], [[1.0, 4.0]]]
    p = pool_intervals(sessions)
    # Pooled over intervals (1, 0.5, 3), not the mean of per-session means (0.75 and 3).
    assert p["mean"] == pytest.approx(1.5) and p["median"] == pytest.approx(1.0)
    assert (p["intervals"], p["sessions"]) == (3, 3)
    assert pool_intervals([[], []])["mean"] is None and pool_intervals([])["mean"] is None


def test_label_shares_use_valid_labels_only():
    s = label_shares([R, S, None, S])
    assert s["judged"] == 3 and s[R] == pytest.approx(1 / 3) and s[S] == pytest.approx(2 / 3) and s[K] == 0
    assert label_shares([None, None]) == {"judged": 0, R: None, S: None, U: None, K: None}


def test_quality_and_coverage_and_joint():
    res = afu(["accept", "accept", "fail", "unresolved", "unresolved"])
    assert res["quality"] == pytest.approx(2 / 3) and res["coverage"] == pytest.approx(3 / 5)
    assert afu(["unresolved"]) == {"quality": None, "coverage": 0.0} and afu([])["coverage"] is None
    j = llm_judge.joint_status
    assert j({"a": "accept", "b": "accept", "c": "accept"}) == "accept"
    assert j({"a": "accept", "b": "fail", "c": "unresolved"}) == "fail"
    assert j({"a": "accept", "b": "accept", "c": "unresolved"}) == "unresolved"


def test_kappa_and_confusion_matrix_on_four_labels():
    a = [R] * 4 + [S] * 4 + [K] * 2
    b = [R] * 3 + [S] + [S] * 4 + [S, K]
    ag = stats.label_agreement(a, b)
    # p_o = 8/10; p_e = 0.4*0.3 + 0.4*0.6 + 0.2*0.1 = 0.38; kappa = 0.42 / 0.62
    assert ag["agreement"] == pytest.approx(0.8) and ag["kappa"] == pytest.approx(0.42 / 0.62)
    cm = stats.confusion_matrix(a, b, list(llm_judge.C_LABELS))
    assert cm["matrix"] == [[3, 1, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1]]
    assert sum(map(sum, cm["matrix"])) == 10
    # The reference runner's reported pair (84.9%, kappa 0.72) implies chance agreement of about 0.46.
    assert (0.849 - 0.72) / (1 - 0.72) == pytest.approx(0.46, abs=0.01)


def _row(i, cat, stop, resp, labels, sem=None, ok=True):
    r = {"id": f"{cat}-{i}", "ok": ok, "category": cat}
    if ok:
        r["overlap"] = {"timing": {"stop_intervals": stop, "response_intervals": resp}}
        r["labels"] = {j: {"status": "valid" if lab else "invalid_label", "label": lab} for j, lab in labels.items()}
        r["semantic"] = sem
    return r


def test_fdb_metrics_per_category_and_all():
    acc = {"interaction_handling": "accept", "relevance": "accept", "grounding": "accept", "joint": "accept"}
    mixed = {"interaction_handling": "fail", "relevance": "accept", "grounding": "unresolved", "joint": "fail"}
    rows = [
        _row(1, "user_interruption", [[1.0, 2.0]], [[3.0, 3.5]], {"qwen": R, "gpt": R}, acc),
        _row(2, "user_interruption", [], [[2.0, 3.0], [5.0, 5.5]], {"qwen": S, "gpt": K}, mixed),
        _row(1, "user_backchannel", [[4.0, 4.25]], [], {"qwen": S, "gpt": None}, acc),
        _row(2, "user_backchannel", None, None, {}, ok=False),  # failed session: in N, never timed or judged
    ]
    m = fdb_metrics(rows, ["qwen", "gpt"], semantic=True)
    assert m["all.pairs"] == 4 and m["all.timed_overlap_sessions"] == 3
    assert m["user_interruption.stop_latency_s"] == pytest.approx(1.0)
    assert m["user_interruption.response_latency_s"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert m["user_backchannel.response_latency_s"] is None  # no intervals: null, never zero
    assert m["all.stop_latency_s"] == pytest.approx(1.25 / 2)
    assert m["all.qwen.judged_pairs"] == 3 and m["all.gpt.judged_pairs"] == 2
    assert m["user_interruption.qwen.C_RESPOND"] == 0.5 and m["user_backchannel.qwen.C_RESUME"] == 1.0
    assert m["user_backchannel.gpt.C_RESUME"] is None
    assert m["all.semantic.interaction_handling_quality"] == pytest.approx(2 / 3)
    assert m["all.semantic.interaction_handling_coverage"] == pytest.approx(3 / 4)
    assert m["all.semantic.joint_quality"] == pytest.approx(2 / 3)
    assert m["user_backchannel.semantic.grounding_coverage"] == pytest.approx(1 / 2)
    # Agreement only over pairs both judges labeled validly: (R, R) and (S, K).
    assert m["agreement.qwen.gpt.n"] == 2 and m["agreement.qwen.gpt.agreement"] == 0.5


def test_window_cuts_or_pads_to_the_input_length_and_native_text_stops_at_T():
    pcm = (np.ones(22050, dtype="<i2") * 1000).tobytes()  # 1 s at 22.05 kHz
    y, after = duplex_fdb.window(pcm, 22050, 8000)
    assert len(y) == 8000 and after
    y, after = duplex_fdb.window(pcm, 22050, 24000)
    assert len(y) == 24000 and not after and np.all(y[17000:] == 0)
    tail_silent = (np.r_[np.ones(11025), np.zeros(11025)].astype("<i2") * 1000).tobytes()
    assert duplex_fdb.window(tail_silent, 22050, 12000)[1] is False  # only the idle tail of the last frame
    assert duplex_fdb.native_text(["<s>", "Hi", " there", "<pad>", " late"], 0.08, 0.3) == "Hi there"


def test_axis_verdicts_need_an_exact_quote():
    packet = {"output_asr_text": "A dolphin is a mammal.", "input_asr_text": "Name a mammal."}
    v = llm_judge.validate_axis
    ok = {"status": "accept", "reason": "r", "source": "output_asr_text", "quote": "dolphin is a mammal"}
    assert v(ok, packet) == {"status": "accept"}
    assert v(ok | {"quote": "a whale"}, packet)["conversion"].startswith("quote is not an exact substring")
    no_quote = {"status": "unresolved", "conversion": "accept verdict has no evidence quote"}
    assert v(ok | {"quote": ""}, packet) == no_quote
    assert v(ok | {"source": "none", "quote": ""}, packet)["status"] == "unresolved"
    assert v(ok | {"status": "unresolved", "source": "none", "quote": ""}, packet) == {"status": "unresolved"}
    assert v(ok | {"extra": 1}, packet) == {"status": "unresolved", "conversion": "schema"}
    assert v(ok | {"quote": "x" * 221}, packet)["conversion"] == "schema"
    packets = [{"sample_id": "p1", **packet}, {"sample_id": "p2", **packet}]
    got = llm_judge.validate_response({"assessments": [
        {"sample_id": "p1", "axes": {ax: ok for ax in llm_judge.AXES}},
        {"sample_id": "p2", "axes": {}}, {"sample_id": "p2", "axes": {}}, {"sample_id": "zz", "axes": {}}]}, packets)
    assert {ax: got["p1"][ax]["status"] for ax in llm_judge.AXES} == dict.fromkeys(llm_judge.AXES, "accept")
    assert got["p2"]["relevance"]["conversion"] == "duplicate sample_id in response"
    assert llm_judge.validate_response(None, packets)["p1"]["grounding"]["conversion"] == "batch response failed"


def _behavior():
    return fdb_reference.Behavior("INSTRUCTION", lambda a, b, c, d: f"{a}|{b}|{c}|{d}",
                                  lambda t: json.dumps(t, separators=(",", ":")),
                                  json.loads)


def test_behavior_label_follows_the_official_retry_loop():
    replies = []

    def post(url, body, headers, timeout):
        seed = body["seed"]
        replies.append(seed)
        return script[seed]

    def reply(content, model="m", finish="stop"):
        return {"model": model, "choices": [{"message": {"content": content}, "finish_reason": finish}]}

    judge = llm_judge.Judge("j", "local", "http://127.0.0.1:1/v1", "m", post=post, retry_sleep_s=0)
    t = {k: {"text": k, "chunks": []} for k in ("input", "clean_input", "output", "clean_output")}
    script = {1: reply("not json"), 2: reply('{"behaviour": ["C_RESUME"]}')}
    res = llm_judge.behavior_label(judge, _behavior(), t, sleep=lambda s: None)
    assert (res["status"], res["label"], res["seed"], replies) == ("valid", S, 2, [1, 2])
    script = {1: reply('{"behaviour": ["C_RESUME"]}', model="other")}
    assert llm_judge.behavior_label(judge, _behavior(), t)["status"] == "model_mismatch"
    script = {1: reply('{"behaviour": ["C_RESUME", "C_RESPOND"]}')}
    assert llm_judge.behavior_label(judge, _behavior(), t)["status"] == "invalid_label"
    script = {1: reply('{"behaviour": ["C_RESUME"]}', finish="length")}
    assert llm_judge.behavior_label(judge, _behavior(), t)["status"] == "invalid_finish"
    script = {}
    res = llm_judge.behavior_label(judge, _behavior(), t, sleep=lambda s: None)
    assert res["status"] == "failed" and res["attempts"] == 3
    body = llm_judge.behavior_body(judge, _behavior(), t)
    assert body["messages"][1]["content"].split("|")[0] == '{"text":"clean_input","chunks":[]}'
    assert body["temperature"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": False}
    api = llm_judge.Judge("g", "openai", "https://api.example/v1", llm_judge.OPENAI_MODEL)
    assert "temperature" not in llm_judge.behavior_body(api, _behavior(), t)  # official: model defaults


def test_transport_retries_rate_limits_with_backoff_and_counts_usage():
    import email.message
    import urllib.error

    calls, sleeps = [], []

    def http_error(code, retry_after=None):
        headers = email.message.Message()
        if retry_after:
            headers["Retry-After"] = retry_after
        return urllib.error.HTTPError("u", code, "x", headers, None)

    def post(url, body, headers, timeout):
        calls.append(body["seed"])
        if len(calls) == 1:
            raise http_error(429, "7")
        if len(calls) == 2:
            raise urllib.error.URLError("reset")
        return {"model": "m", "usage": {"prompt_tokens": 100, "completion_tokens": 5},
                "choices": [{"message": {"content": '{"behaviour": ["C_RESUME"]}'}, "finish_reason": "stop"}]}

    judge = llm_judge.Judge("j", "local", "http://127.0.0.1:1/v1", "m", post=post, sleep=sleeps.append)
    t = {k: {"text": k, "chunks": []} for k in ("input", "clean_input", "output", "clean_output")}
    res = llm_judge.behavior_label(judge, _behavior(), t, sleep=lambda s: None)
    # Same seed throughout: transport retries do not use up the official loop's attempts.
    assert (res["status"], res["seed"], calls) == ("valid", 1, [1, 1, 1])
    assert sleeps[0] == 7.0 and len(sleeps) == 2
    assert judge.usage == {"requests": 1, "retries": 2, "prompt": 100, "completion": 5}

    def bad(url, body, headers, timeout):
        raise http_error(400)

    no_retry = llm_judge.Judge("j", "local", "http://127.0.0.1:1/v1", "m", post=bad, sleep=sleeps.append)
    with pytest.raises(llm_judge.JudgeError, match="HTTP 400"):
        no_retry.chat({"seed": 1})
    assert no_retry.usage["retries"] == 0

    # Retries that run out end the pair as failed on its first seed, never a label from seed 2.
    seeds, waits = [], []

    def limited(url, body, headers, timeout):
        seeds.append(body["seed"])
        raise http_error(429, "120")

    busy = llm_judge.Judge("j", "local", "http://127.0.0.1:1/v1", "m", post=limited, sleep=waits.append)
    res = llm_judge.behavior_label(busy, _behavior(), t, sleep=lambda s: None)
    assert (res["status"], res["label"], set(seeds)) == ("failed", None, {1})
    assert waits == [120.0] * llm_judge.TRANSPORT_RETRIES  # Retry-After is a minimum, honored in full
    seeds.clear()

    def far(url, body, headers, timeout):
        seeds.append(body["seed"])
        raise http_error(429, "3600")

    gone = llm_judge.Judge("j", "local", "http://127.0.0.1:1/v1", "m", post=far, sleep=waits.append)
    assert llm_judge.behavior_label(gone, _behavior(), t, sleep=lambda s: None)["status"] == "failed"
    assert seeds == [1]


def test_truncated_responses_are_retried_on_the_same_seed():
    import http.client

    for caller in ("behavior", "semantic"):
        seeds = []
        content = ('{"behaviour": ["C_RESUME"]}' if caller == "behavior"
                   else '{"assessments": []}')

        def post(url, body, headers, timeout, seeds=seeds, content=content):
            seeds.append(body.get("seed"))
            if len(seeds) == 1:
                raise http.client.IncompleteRead(b"partial", 10)
            return {"model": "m", "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

        judge = llm_judge.Judge("j", "local", "http://127.0.0.1:1/v1", "m", post=post, sleep=lambda s: None)
        if caller == "behavior":
            t = {k: {"text": k, "chunks": []} for k in ("input", "clean_input", "output", "clean_output")}
            res = llm_judge.behavior_label(judge, _behavior(), t, sleep=lambda s: None)
            assert (res["status"], res["seed"], seeds) == ("valid", 1, [1, 1])
        else:
            assert llm_judge.submit(judge, {"seed": 1}) == {"assessments": []} and seeds == [1, 1]
        assert judge.usage["retries"] == 1


def test_malformed_replies_never_raise_their_content():
    secret = "Bearer SYNTHETIC-VALUE"

    def post(url, body, headers, timeout):
        return {"model": "m", "usage": {"prompt_tokens": secret, "completion_tokens": True},
                "choices": [{"message": None, "finish_reason": "stop"}]}

    judge = llm_judge.Judge("j", "local", "http://127.0.0.1:1/v1", "m", post=post)
    assert llm_judge.submit(judge, {"seed": 1}) is None
    assert judge.usage["prompt"] == 0 and judge.usage["completion"] == 0
    t = {k: {"text": k, "chunks": []} for k in ("input", "clean_input", "output", "clean_output")}
    res = llm_judge.behavior_label(judge, _behavior(), t, sleep=lambda s: None)
    assert res["status"] == "failed" and secret not in json.dumps(res)


def test_retry_failed_matches_a_remote_judge_against_its_saved_identity(tmp_path, monkeypatch):
    from benchmark.eval.suites.duplex_fdb import Judging, rejudge

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    monkeypatch.setattr(fdb_reference, "load_behavior", _behavior)

    def post(url, body, headers, timeout):
        return {"model": llm_judge.OPENAI_MODEL,
                "choices": [{"message": {"content": '{"behaviour": ["C_RESUME"]}'}, "finish_reason": "stop"}]}

    def judging():
        (j,) = llm_judge.parse_judges(["gpt4o=openai"])
        j.post = post
        return Judging([j], "none")

    t = {k: {"text": k, "chunks": []} for k in ("input", "clean_input", "output", "clean_output")}
    rows = [{"id": f"a{i}", "ok": True, "category": "user_backchannel", "asr": t,
             "overlap": {"timing": {"stop_intervals": [], "response_intervals": []}}} for i in range(2)]
    first = rejudge(RunResult("fdb", "x", {"suite": {}}, {}, [duplex_fdb_repeat(rows)]).finalize(), judging())
    first.repeats[0].requests[1]["labels"]["gpt4o"] = {"status": "failed", "label": None}
    first.save(tmp_path / "r.json")
    saved = RunResult.load(tmp_path / "r.json")
    assert saved.config["suite"]["judges"][0]["base_url"] != llm_judge.OPENAI_URL  # redacted when saved
    again = rejudge(saved, judging(), retry_failed=True)
    assert [r["labels"]["gpt4o"]["status"] for r in again.repeats[0].requests] == ["valid", "valid"]
    assert again.config["rejudged"][-1]["usage"]["gpt4o"]["requests"] == 1
    assert "sk-test" not in json.dumps(again.to_dict())


def test_rejudge_keeps_the_stored_labels_of_a_judge_skipped_now(monkeypatch, capsys):
    from benchmark.eval.suites.duplex_fdb import Judging, rejudge

    monkeypatch.setattr(fdb_reference, "load_behavior", _behavior)
    monkeypatch.setattr(llm_judge, "resolve", lambda j: j)

    def post(url, body, headers, timeout):
        model = llm_judge.OPENAI_MODEL if "openai" in url else "m"
        return {"model": model,
                "choices": [{"message": {"content": '{"behaviour": ["C_RESUME"]}'}, "finish_reason": "stop"}]}

    def judging():
        q = llm_judge.Judge("qwen", "local", "http://127.0.0.1:1/v1", "m", post=post)
        (g,) = llm_judge.parse_judges(["gpt4o=openai"])
        g.post = post
        return Judging([q, g], "none")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    t = {k: {"text": k, "chunks": []} for k in ("input", "clean_input", "output", "clean_output")}
    rows = [{"id": f"a{i}", "ok": True, "category": "user_backchannel", "asr": t,
             "overlap": {"timing": {"stop_intervals": [], "response_intervals": []}}} for i in range(3)]
    first = rejudge(RunResult("fdb", "x", {"suite": {}}, {}, [duplex_fdb_repeat(rows)]).finalize(), judging())
    stored = json.loads(json.dumps(first.config["suite"]["judges"][1]))
    assert stored["name"] == "gpt4o" and stored["skipped"] is None

    monkeypatch.delenv("OPENAI_API_KEY")
    again = rejudge(first, judging())
    assert [r["labels"]["gpt4o"]["status"] for r in again.repeats[0].requests] == ["valid"] * 3
    assert again.config["suite"]["judges"][1] == stored
    assert again.repeats[0].metrics["agreement.qwen.gpt4o.agreement"] == 1.0
    assert again.config["rejudged"][-1]["judges"] == ["qwen"]
    assert "judge 'gpt4o' skipped; its stored labels are kept" in capsys.readouterr().err
    retried = rejudge(again, judging(), retry_failed=True)
    assert retried.config["suite"]["judges"][1] == stored
    assert retried.repeats[0].metrics["agreement.qwen.gpt4o.agreement"] == 1.0


def _fdb_zips(zdir, level):
    """The four category zips, one pair each, whose audio is a constant ``level``."""
    import zipfile

    zdir.mkdir(parents=True, exist_ok=True)
    wav = zdir / "a.wav"
    write_wav(wav, np.full(1600, level, dtype=np.float32), 16000)
    for cat in fdb_reference.CATEGORIES:
        with zipfile.ZipFile(zdir / f"{cat}.zip", "w") as zf:
            for name in ("input.wav", "clean_input.wav"):
                zf.write(wav, f"{cat}/1/{name}")
            meta = {"context_text": "c", fdb_reference.EVENT_TEXT_KEYS[cat]: "e", "timestamps": [0.0, 0.1]}
            zf.writestr(f"{cat}/1/metadata.json", json.dumps(meta))


def test_fdb_manifest_audio_follows_the_archives_its_revision_names(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    out = tmp_path / "out"
    _fdb_zips(tmp_path / "z1", 0.25)
    _fdb_zips(tmp_path / "z2", -0.5)
    revisions, levels = [], []
    for zdir in ("z1", "z2", "z1"):
        manifest = fdb_reference.build_manifest(out, None, tmp_path / zdir)
        items = [json.loads(line) for line in manifest.read_text().splitlines()]
        revisions.append({i["extra"]["dataset_revision"] for i in items})
        x, _ = read_wav(manifest.parent / items[0]["audio_path"])
        levels.append(round(float(x.mean()), 2))
    assert levels == [0.25, -0.5, 0.25]
    assert all(len(r) == 1 for r in revisions) and revisions[0] == revisions[2] != revisions[1]


def test_fdb_data_refuses_bad_counts_and_reports_a_failed_download(tmp_path, monkeypatch):
    from benchmark.eval.cli import main

    for bad in ("-1", "0", "abc"):
        with pytest.raises(SystemExit, match="positive count"):
            main(["data", "fdb-v15", "--out", str(tmp_path / "o"), "--per-category", bad])

    def offline(**kw):
        raise ConnectionError("drive.google.com unreachable")

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=offline))
    with pytest.raises(SystemExit, match=r"download from Google Drive failed \(ConnectionError\)"):
        main(["data", "fdb-v15", "--out", str(tmp_path / "o"), "--per-category", "1"])


def test_api_judge_is_skipped_without_a_key_and_keys_never_reach_results(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (j,) = llm_judge.parse_judges(["gpt4o=openai"])
    assert not j.available and "OPENAI_API_KEY" in j.skipped
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    q, g = llm_judge.parse_judges(["qwen=http://127.0.0.1:9/v1#qwen3.8-27b", "gpt4o=openai"], ["qwen=abc123"])
    assert g.available and g.model == llm_judge.OPENAI_MODEL and q.model == "qwen3.8-27b" and q.revision == "abc123"
    assert "sk-test" not in json.dumps([q.identity(), g.identity()])
    for bad in (["x"], ["a=openai", "a=openai"], ["a/b=openai"]):
        with pytest.raises(ValueError):
            llm_judge.parse_judges(bad)
    with pytest.raises(ValueError):
        llm_judge.parse_judges(["a=openai"], ["b=rev"])


def test_token_counts_are_kept_and_token_credentials_are_not():
    from benchmark.eval.schema import scrub, scrub_env

    cfg = scrub({"decoding": {"max_tokens": 512, "max_new_tokens": 8}, "tokens": "hf_abc", "HF_TOKENS": "hf_def",
                 "API_TOKENS": 123456789, "max_tokens_secret": "x", "max_tokens": "hf_ghi"})
    assert cfg["decoding"] == {"max_tokens": 512, "max_new_tokens": 8}
    assert {cfg[k] for k in ("tokens", "HF_TOKENS", "API_TOKENS", "max_tokens_secret", "max_tokens")} == {"<redacted>"}
    assert scrub_env({"HF_TOKENS": "hf_abc", "MSTAR_TOKENS": "1"}) == {"HF_TOKENS": "<redacted>",
                                                                       "MSTAR_TOKENS": "<redacted>"}


def test_semantic_replies_from_another_model_are_rejected():
    assets = llm_judge.SemanticAssets(prompt="p", schema={}, controls=[{"sample_id": "s01", "output_asr_text": "ok"}],
                                      expected={"s01": dict.fromkeys(llm_judge.AXES, "accept")})
    axis = {"status": "accept", "reason": "r", "source": "output_asr_text", "quote": "ok"}
    content = json.dumps({"assessments": [{"sample_id": "s01", "axes": dict.fromkeys(llm_judge.AXES, axis)}]})

    def judge(model):
        def post(url, body, headers, timeout):
            return {"model": model, "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        return llm_judge.Judge("j", "local", "http://127.0.0.1:1/v1", "m", post=post)

    assert llm_judge.run_controls(judge("m"), assets)["passed"]
    assert not llm_judge.run_controls(judge("other"), assets)["passed"]
    graded = llm_judge.grade(judge("other"), assets, [{"sample_id": "p1", "output_asr_text": "ok"}])
    assert graded["p1"]["relevance"] == {"status": "unresolved", "conversion": "batch response failed"}


def test_load_behavior_reads_the_prompt_without_importing_the_file(tmp_path):
    src = tmp_path / "eval_behavior.py"
    src.write_text('''
import json
raise RuntimeError("importing this file would make an API client")
def json_dict_to_compact_text(json_list):
    return json.dumps(json_list, separators=(",", ":"), ensure_ascii=False)
def extract_json(text, key="behaviour"):
    return json.loads(text[text.find("{"):])
def parse_eval(data):
    return extract_json(data) if isinstance(data, str) else data
def eval_behavior_all(data_dir, client, task):
    final_input = f"""
    {{ "input_clean": {input_clean_text}, "input_noisy": {input_noisy_text},
       "output_clean": {output_clean_text}, "output_noisy": {output_noisy_text} }}
    """
''')
    (tmp_path / "behavior.txt").write_text("Label it.")
    b = fdb_reference.load_behavior(src, tmp_path / "behavior.txt")
    t = {"clean_input": {"text": "a"}, "input": {"text": "b"}, "clean_output": {"text": "c"}, "output": {"text": "d"}}
    msg = b.user_message(t)
    assert '"input_clean": {"text":"a"}' in msg and '"output_noisy": {"text":"d"}' in msg
    assert b.instruction == "Label it." and b.parse_eval('x {"behaviour": ["C_RESUME"]}') == {"behaviour": [S]}


def test_nested_row_strings_are_scrubbed_and_redacted_pairs_are_not_rejudged(tmp_path):
    row = {"id": "a", "ok": True, "asr": {"output": {"text": "see 10.1.2.3", "chunks": [{"text": "10.1.2.3"}]}},
           "overlap": {"frames": 3}}
    RunResult("fdb", "x", {}, {}, [duplex_fdb_repeat([row])]).finalize().save(tmp_path / "r.json")
    back = RunResult.load(tmp_path / "r.json").repeats[0].requests[0]
    assert "10.1.2.3" not in json.dumps(back) and back["scrubbed"] == ["asr"]
    assert not duplex_fdb.assessable(back) and duplex_fdb.assessable(row)


def duplex_fdb_repeat(rows):
    from benchmark.eval.schema import RepeatResult

    return RepeatResult(0, 1.0, {"requests": len(rows)}, rows)


# --- end to end against fakes -----------------------------------------------------------------

OUT_RATE = 22050


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "healthy"})
        if self.path.startswith("/auth/") and self.headers.get("Authorization") != "Bearer k-123":
            return self._json({"error": "unauthorized"}, 401)
        if self.path.endswith("/v1/models"):
            return self._json({"data": [{"id": "fake-judge", "root": "fake/judge"}]})
        self._json({}, 404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/generate":
            return self._generate(body)
        req = json.loads(body)
        judge = self.path.split("/")[1]  # /a/v1/... or /b/v1/...
        if "response_format" in req:  # semantic batch: accept every axis, quoting the output transcript
            cases = [json.loads(ln) for ln in req["messages"][1]["content"].split("in input order:\n")[1].splitlines()]
            axis = {"status": "accept", "reason": "ok", "source": "output_asr_text", "quote": "agent"}
            content = {"assessments": [{"sample_id": c["sample_id"], "axes": dict.fromkeys(llm_judge.AXES, axis)}
                                       for c in cases]}
        else:  # behavior: judge a resumes everywhere, judge b responds on interruptions
            user = req["messages"][1]["content"]
            label = S if judge == "a" or "interrupt" not in user else R
            content = {"behaviour": [label]}
        self._json({"model": req["model"], "choices": [{"message": {"content": json.dumps(content)},
                                                        "finish_reason": "stop"}]})

    def _generate(self, body):
        from email.parser import BytesParser
        from email.policy import HTTP

        msg = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
        parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                 for p in msg.iter_parts()}
        import io
        import wave

        with wave.open(io.BytesIO(parts["files"]), "rb") as w:
            n = w.getnframes()
        if n == 40000:  # a 2.5 s clip: the session the tests make fail
            return self._json({"error": "overloaded"}, 503)
        frames = n // 1280
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        t = np.arange(int(0.08 * OUT_RATE)) / OUT_RATE
        tone = (0.3 * 32767 * np.sin(2 * np.pi * 220 * t)).astype("<i2").tobytes()
        for i in range(frames):
            speaking = 10 <= i < 30  # agent speaks 0.8 to 2.4 s
            tok = "agent" if i == 10 else ("<pad>" if not speaking else " x")
            pcm = tone if speaking else b"\x00\x00" * len(t)
            for modality, data, meta in (("text", tok.encode(), {}), ("audio", pcm, {"sample_rate": OUT_RATE})):
                line = {"modality": modality, "data": base64.b64encode(data).decode(), "metadata": meta}
                self.wfile.write((json.dumps(line) + "\n").encode())
        self.wfile.flush()


@pytest.fixture
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _fake_timing():
    """A stand-in for the official get_timing.py: energy VAD spans, the same interval rules."""
    m = types.SimpleNamespace(SR=16000, USER_MERGE_GAP=0.6, MODEL_MERGE_GAP=0.5, OUT_FILENAME="latency_intervals.json")
    m.load_wav = lambda p: read_wav(p)[0]
    m.seg_sec = lambda wav, gap: [(s, e) for s, e in speech_segments(wav, 16000)]

    def process_folder(folder):
        user = m.seg_sec(m.load_wav(folder / "input.wav"), 0.6)
        model = m.seg_sec(m.load_wav(folder / "output.wav"), 0.5)
        stop = [[max(us, ms), min(ue, me)] for us, ue in user for ms, me in model if min(ue, me) > max(us, ms)]
        resp = [[ue, next(ms for ms, _ in model if ms > ue)] for _, ue in user if any(ms > ue for ms, _ in model)]
        (folder / m.OUT_FILENAME).write_text(json.dumps({"latency_stop_list": stop, "latency_resp_list": resp}))

    m.process_folder = process_folder
    return fdb_reference.Timing(m, {"vad_branch": "fake", "audio_loader": "fake"})


class _FakeASR:
    id = "fake-asr"

    def transcribe(self, audio, stage):
        name = f"{Path(audio).parent.name} {Path(audio).stem}"  # "<item id> <file>", so judges can tell pairs apart
        return {"text": f"{name} agent words", "chunks": [{"text": name, "timestamp": [0.1, 0.4]}]}


def _dataset(tmp_path, failing_clean=None):
    rate = 16000
    t = np.arange(int(1.0 * rate)) / rate
    speech = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    silence = np.zeros(rate, dtype=np.float32)
    items = []
    for cat, n in (("user_interruption", 2), ("background_speech", 1)):
        for k in range(1, n + 1):
            d = tmp_path / "audio" / cat / str(k)
            d.mkdir(parents=True)
            # user: 0 to 1 s, an event at 2 to 3 s (overlapping the agent), then silence to 4 s
            write_wav(d / "input.wav", np.concatenate([speech, silence, speech, silence]), rate)
            clean = speech[: rate // 2] if (cat, k) == failing_clean else silence  # 2.5 s: the fake server fails it
            write_wav(d / "clean_input.wav", np.concatenate([speech, silence, clean]), rate)
            items.append(Item(id=f"{cat}-{k}", audio_path=str(d / "input.wav"), lang="en", extra={
                "category": cat, "sample_id": f"{cat}/{k}", "clean_audio": "clean_input.wav",
                "context_text": "Tell me about rain.", "event_text": "interrupt: and snow?" if "inter" in cat else "bg",
                "event_s": [2.0, 3.0], "dataset_revision": "sha256:test"}))
    return save_manifest(items, tmp_path / "manifest.jsonl")


def _run_fdb_with_two_judges(tmp_path, fake_server, monkeypatch):
    """`run fdb` on the fake dataset with judges qwen, other and a skipped gpt4o; returns the out dir."""
    from benchmark.eval.cli import main

    assets = llm_judge.SemanticAssets(
        prompt="Grade.", schema={"type": "object"},
        controls=[{"sample_id": "s01", "output_asr_text": "an agent reply"}],
        expected={"s01": dict.fromkeys(llm_judge.AXES, "accept")})
    monkeypatch.setattr(fdb_reference, "load_timing", _fake_timing)
    monkeypatch.setattr(fdb_reference, "ParakeetASR", lambda device: _FakeASR())
    monkeypatch.setattr(fdb_reference, "load_behavior", _behavior)
    monkeypatch.setattr(llm_judge.SemanticAssets, "load", classmethod(lambda cls: assets))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manifest = _dataset(tmp_path)
    out = tmp_path / "out"
    main(["run", "fdb", "--url", fake_server, "--data", str(manifest), "--repeats", "2", "--warmup", "0",
          "--concurrency", "2", "--label", "fake", "--out", str(out),
          "--judge", f"qwen={fake_server}/a/v1", "--judge", f"other={fake_server}/b/v1", "--judge", "gpt4o=openai"])
    return out


def test_fdb_report_and_agreement_charts(tmp_path, fake_server, monkeypatch):
    from benchmark.eval.cli import main

    pytest.importorskip("matplotlib")
    out = _run_fdb_with_two_judges(tmp_path, fake_server, monkeypatch)
    main(["report", str(out), "--md", str(tmp_path / "t.md"), "--chart", str(tmp_path / "c.png")])
    assert (tmp_path / "c.png").stat().st_size > 10_000
    main(["agreement", str(out), "--md", str(tmp_path / "a.md"), "--chart", str(tmp_path / "cm.png")])
    assert (tmp_path / "cm.png").stat().st_size > 1_000


def test_fdb_suite_end_to_end_with_two_judges_agreement_and_rejudge(tmp_path, fake_server, monkeypatch, capsys):
    from benchmark.eval.cli import main

    out = _run_fdb_with_two_judges(tmp_path, fake_server, monkeypatch)
    r = RunResult.load(out / "fdb-fake-c2.json")
    s = r.summary
    assert s["errors"]["mean"] == 0 and s["all.pairs"]["mean"] == 3 and s["all.timed_overlap_sessions"]["mean"] == 3
    # Agent speaks 0.8 to 2.4 s; the user's event starts at 2 s: 0.4 s of overlap per pair, plus the
    # query's own 0.2 s overlap (0.8 to 1 s). Responses: query end (1 s) to agent start? none later, so
    # only the event end (3 s) has no later model start either: no response intervals.
    assert s["user_interruption.stop_latency_s"]["mean"] == pytest.approx(0.3, abs=0.03)
    assert s["user_interruption.response_latency_s"]["n"] == 0
    assert s["all.qwen.C_RESUME"]["mean"] == 1.0 and s["user_interruption.other.C_RESPOND"]["mean"] == 1.0
    assert s["background_speech.other.C_RESUME"]["mean"] == 1.0
    assert s["agreement.qwen.other.agreement"]["mean"] == pytest.approx(1 / 3)
    assert s["all.semantic.joint_quality"]["mean"] == 1.0 and s["all.semantic.joint_coverage"]["mean"] == 1.0
    cfg = r.config["suite"]
    assert [j["name"] for j in cfg["judges"]] == ["qwen", "other", "gpt4o"]
    assert cfg["judges"][2]["skipped"] == "OPENAI_API_KEY is not set" and cfg["semantic"]["judge"] == "qwen"
    assert cfg["semantic"]["controls"]["passed"] and cfg["reference"]["fdb"]["license"] == "CC BY-NC 4.0"
    row = r.repeats[0].requests[0]
    assert row["overlap"]["boundary"] == {"playout_active_at_T": False, "audio_received_after_T": False}
    assert row["overlap"]["native_text"] == "agent" + " x" * 19
    assert set(row["asr"]) == {"input", "clean_input", "output", "clean_output"}

    main(["report", str(out), "--md", str(tmp_path / "t.md")])
    table = (tmp_path / "t.md").read_text()
    assert "| user_interruption | 2 | 2 |" in table and "Judge `gpt4o` skipped: OPENAI_API_KEY is not set." in table

    main(["agreement", str(out), "--md", str(tmp_path / "a.md")])
    text = (tmp_path / "a.md").read_text()
    assert "over 6 pairs both labeled validly" in text and "Label agreement: 33.3%" in text
    assert "| user_interruption | 4 | 0.0% |" in text
    for bad in ("qwen", "qwen,other,late", "qwen,"):
        with pytest.raises(SystemExit, match="two judge names"):
            main(["agreement", str(out), "--judges", bad])

    # A third judge later, from the stored transcripts: no server, no GPU.
    main(["rejudge", str(out / "fdb-fake-c2.json"), "--judge", f"late={fake_server}/b/v1",
          "--out", str(tmp_path / "rejudged")])
    back = RunResult.load(tmp_path / "rejudged" / "fdb-fake-c2.json")
    assert back.summary["agreement.other.late.agreement"]["mean"] == 1.0
    assert back.summary["agreement.qwen.other.agreement"]["mean"] == pytest.approx(1 / 3)
    (entry,) = back.config["rejudged"]
    assert (entry["judges"], entry["semantic"], entry["usage"]["late"]["requests"]) == (["late"], None, 6)
    assert back.summary["user_interruption.stop_latency_s"] == s["user_interruption.stop_latency_s"]
    assert back.config["suite"]["semantic"]["identity"]["model"] == "fake-judge"
    assert back.config["suite"]["semantic"]["decoding"]["max_tokens"] == 32768  # a count, not a secret

    # Replacing a judge by name: it keeps its place, every derived metric is recomputed, and the
    # semantic grades it gave are dropped unless regraded.
    main(["rejudge", str(tmp_path / "rejudged"), "--judge", f"qwen={fake_server}/b/v1",
          "--out", str(tmp_path / "replaced")])
    rep = RunResult.load(tmp_path / "replaced" / "fdb-fake-c2.json")
    assert [j["name"] for j in rep.config["suite"]["judges"]] == ["qwen", "other", "gpt4o", "late"]
    assert rep.summary["agreement.qwen.other.agreement"]["mean"] == 1.0
    assert not any(k.startswith(("agreement.other.qwen", "agreement.late.qwen")) for k in rep.summary)
    assert rep.config["suite"]["semantic"] is None and not any(".semantic." in k for k in rep.summary)
    assert all("semantic" not in r for rp in rep.repeats for r in rp.requests)
    # Semantic regrading alone keeps every behavior label and the judge list.
    main(["rejudge", str(tmp_path / "rejudged"), "--judge", f"sem={fake_server}/a/v1", "--semantic-judge", "sem",
          "--semantic-only", "--out", str(tmp_path / "semantic")])
    sem = RunResult.load(tmp_path / "semantic" / "fdb-fake-c2.json")
    assert [j["name"] for j in sem.config["suite"]["judges"]] == ["qwen", "other", "gpt4o", "late"]
    assert sem.config["suite"]["semantic"]["judge"] == "sem" and sem.config["rejudged"][-1]["semantic_only"]
    assert sem.config["rejudged"][-1]["usage"]["sem"]["requests"] == 1 + 2  # the controls, then a batch per repeat
    assert sem.summary["all.semantic.joint_quality"]["mean"] == 1.0
    assert sem.summary["agreement.qwen.other.agreement"] == back.summary["agreement.qwen.other.agreement"]
    assert all("sem" not in r["labels"] for rp in sem.repeats for r in rp.requests)

    # --retry-failed relabels only the failed pair, and only for the judge that labeled the file.
    data = json.loads((tmp_path / "rejudged" / "fdb-fake-c2.json").read_text())
    data["repeats"][1]["requests"][0]["labels"]["late"] = {"status": "failed", "label": None}
    (tmp_path / "failed").mkdir()
    (tmp_path / "failed" / "fdb-fake-c2.json").write_text(json.dumps(data))
    main(["rejudge", str(tmp_path / "failed"), "--judge", f"late={fake_server}/b/v1", "--retry-failed",
          "--out", str(tmp_path / "retried")])
    ret = RunResult.load(tmp_path / "retried" / "fdb-fake-c2.json")
    assert ret.config["rejudged"][-1]["usage"]["late"]["requests"] == 1
    assert ret.repeats[1].requests[0]["labels"]["late"]["status"] == "valid"
    assert ret.summary["agreement.other.late.agreement"]["mean"] == 1.0
    with pytest.raises(SystemExit, match="not the judge that labeled"):
        main(["rejudge", str(tmp_path / "failed"), "--judge", f"late={fake_server}/b/v1", "--judge-revision",
              "late=r2", "--retry-failed", "--out", str(tmp_path / "retried2")])

    # Pooling files where one name means two different judges is refused.
    monkeypatch.setattr(llm_judge, "resolve", lambda j: (setattr(j, "model", "other-model"), j)[1])
    main(["rejudge", str(out / "fdb-fake-c2.json"), "--judge", f"other={fake_server}/b/v1",
          "--out", str(tmp_path / "swapped")])
    with pytest.raises(SystemExit, match="different model"):
        main(["agreement", str(out), str(tmp_path / "swapped"), "--judges", "qwen,other"])


def test_local_judge_discovery_sends_the_judge_key(fake_server, monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    (j,) = llm_judge.parse_judges([f"q={fake_server}/auth/v1"])
    with pytest.raises(SystemExit, match="unreachable"):
        llm_judge.resolve(j)
    monkeypatch.setenv("JUDGE_API_KEY", "k-123")
    assert llm_judge.resolve(j).model == "fake-judge"


def test_a_failed_clean_session_keeps_the_overlap_session_timed(tmp_path, fake_server, monkeypatch):
    from benchmark.eval.cli import main

    monkeypatch.setattr(fdb_reference, "load_timing", _fake_timing)
    monkeypatch.setattr(fdb_reference, "ParakeetASR", lambda device: _FakeASR())
    monkeypatch.setattr(fdb_reference, "load_behavior", _behavior)
    main(["run", "fdb", "--url", fake_server, "--data", str(_dataset(tmp_path, ("background_speech", 1))),
          "--repeats", "1", "--warmup", "0", "--label", "fake", "--out", str(tmp_path / "out"),
          "--judge", f"qwen={fake_server}/a/v1", "--semantic-judge", "none"])
    r = RunResult.load(tmp_path / "out" / "fdb-fake-c1.json")
    s = r.summary
    assert s["errors"]["mean"] == 1 and s["all.pairs"]["mean"] == 3
    assert s["all.timed_overlap_sessions"]["mean"] == 3 and s["background_speech.stop_latency_s"]["mean"] is not None
    assert s["all.qwen.judged_pairs"]["mean"] == 2 and s["background_speech.qwen.judged_pairs"]["mean"] == 0
    bad = next(row for row in r.repeats[0].requests if row["id"] == "background_speech-1")
    assert (bad["ok"], bad["error"]) == (False, "clean: HTTP 503") and "asr" not in bad and "clean" not in bad


def test_fdb_failed_controls_leave_semantic_ungraded(tmp_path, fake_server, monkeypatch):
    from benchmark.eval.cli import main

    assets = llm_judge.SemanticAssets(prompt="Grade.", schema={},
                                      controls=[{"sample_id": "s01", "output_asr_text": "x"}],
                                      expected={"s01": dict.fromkeys(llm_judge.AXES, "fail")})
    monkeypatch.setattr(fdb_reference, "load_timing", _fake_timing)
    monkeypatch.setattr(fdb_reference, "ParakeetASR", lambda device: _FakeASR())
    monkeypatch.setattr(fdb_reference, "load_behavior", _behavior)
    monkeypatch.setattr(llm_judge.SemanticAssets, "load", classmethod(lambda cls: assets))
    main(["run", "fdb", "--url", fake_server, "--data", str(_dataset(tmp_path)), "--repeats", "1", "--warmup", "0",
          "--label", "fake", "--out", str(tmp_path / "out"), "--judge", f"qwen={fake_server}/a/v1"])
    r = RunResult.load(tmp_path / "out" / "fdb-fake-c1.json")
    assert not r.config["suite"]["semantic"]["controls"]["passed"]
    assert r.summary["all.semantic.joint_coverage"]["mean"] == 0.0 and r.summary["all.semantic.joint_quality"]["n"] == 0
    assert r.summary["all.qwen.judged_pairs"]["mean"] == 3
