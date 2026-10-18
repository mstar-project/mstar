"""CPU tests for the AMD benchmark tables (benchmark/amd) and the runner pieces they use."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.amd import catalog, health, sweep, table, tables
from benchmark.amd.launch import Launch, gpu_list


def test_server_args_tp_and_context():
    row = catalog.ROWS["qwen3_5_9b_tp2"]
    args, env = catalog.server_args(catalog.ARMS["vllm-aiter"], row)
    assert args[:2] == ["--attention-backend", "ROCM_AITER_FA"]
    assert ["--tensor-parallel-size", "2"] == args[args.index("--tensor-parallel-size"):][:2]
    assert ["--max-model-len", "32768"] == args[args.index("--max-model-len"):][:2]
    assert env == {"VLLM_ROCM_USE_AITER": "1", "NCCL_MIN_NCHANNELS": "112"}
    args, env = catalog.server_args(catalog.ARMS["sglang-recipe"], row)
    assert "--enable-aiter-allreduce-fusion" in args and ["--tp", "2"] == args[args.index("--tp"):][:2]
    assert env["SGLANG_MAMBA_SSM_DTYPE"] == "bfloat16"
    args, env = catalog.server_args(catalog.ARMS["vllm-aiter"], catalog.ROWS["qwen3_5_0.8b"])
    assert "--tensor-parallel-size" not in args and "NCCL_MIN_NCHANNELS" not in env


def test_arms_for_filters_by_family():
    names = ["mstar", "vllm", "ref-kokoro"]
    assert [a.name for a in catalog.arms_for(catalog.ROWS["kokoro"], names)] == ["mstar", "ref-kokoro"]
    assert [a.name for a in catalog.arms_for(catalog.ROWS["qwen3_5_2b"], names)] == ["mstar", "vllm"]
    with pytest.raises(KeyError):
        catalog.arms_for(catalog.ROWS["kokoro"], ["nope"])


def test_every_table_names_known_rows_arms_and_workloads():
    for t in tables.TABLES.values():
        assert all(r in catalog.ROWS for r in t.rows), t.name
        assert all(a in catalog.ARMS for a in t.arms), t.name
        assert all(w in tables.WORKLOADS for w in t.workloads), t.name
        for r in t.rows:
            assert catalog.arms_for(catalog.ROWS[r], list(t.arms)), (t.name, r)


def test_runner_command_text_and_sets(tmp_path):
    row, arm = catalog.ROWS["qwen3_5_4b"], catalog.ARMS["sglang-aiter"]
    cmd = sweep.runner_command(arm, row, tables.WORKLOADS["text-long-1k"], 16, 8400, tmp_path / "r1", 3)
    s = " ".join(cmd)
    assert "--inference-system sglang" in s and "--model qwen3.5" in s and "--model-id Qwen/Qwen3.5-4B" in s
    assert "--ignore-eos --output-len-min 64 --output-len-max 64" in s and "--unique-prompts" in s
    assert "--num-requests 80" in s and "--max-concurrency 16" in s and "--num-warmup 3" in s
    assert json.loads(cmd[cmd.index("--chat-template-kwargs") + 1]) == {"enable_thinking": False}
    tts = sweep.runner_command(catalog.ARMS["mstar"], catalog.ROWS["kokoro"], tables.WORKLOADS["tts-seed"], 1, 8400,
                               tmp_path / "r1", 1, sets_root=Path("/data"))
    assert tts[tts.index("--dataset") + 1] == "sample_set"
    assert tts[tts.index("--sample-set-dir") + 1] == "/data/sets/tts_seedtts_v1"
    assert "--chat-template-kwargs" not in tts and tts[tts.index("--num-requests") + 1] == "40"
    asr = sweep.runner_command(catalog.ARMS["mstar"], catalog.ROWS["whisper_large"], tables.WORKLOADS["asr-libri"], 8,
                               8400, tmp_path / "r1", 1)
    assert asr[asr.index("--inference-system") + 1] == "openai_transcribe"


def _results(per, wall=10.0, **agg):
    return {"completed": len(per), "failed": 0, "wall_time_s": wall, "request_throughput": len(per) / wall,
            "aggregate": agg, "per_request": per}


def test_repeat_stats_text_tpot_and_tokens():
    per = [{"request_id": i, "jct_ms": 1000.0 + 100 * i, "ttft_s": {"text": 0.1}, "output_text_tokens": 11}
           for i in range(3)]
    res = _results(per, ttft_s={"text": {"p50": 0.1, "p95": 0.12}}, itl_s={"text": {"p50": 0.09, "p95": 0.1}},
                   e2e_latency_s={"p50": 1.1, "p95": 1.2}, text_token_throughput=3.3, total_text_tokens=33,
                   total_input_tokens=30)
    st = sweep.repeat_stats(res, "text")
    assert st["tpot_p50_s"] == pytest.approx((1.1 - 0.1) / 10)
    assert st["tok_s"] == 3.3 and st["out_tokens_per_req"] == 11 and st["in_tokens_per_req"] == 10
    assert st["ttft_p95_s"] == 0.12 and st["itl_p50_s"] == 0.09
    # a fixed output length of 10 counts 10 of the 11 tokens each request sent
    assert sweep.repeat_stats(res, "text", output_len=10)["tok_s"] == pytest.approx(30 / 10.0)


def test_repeat_stats_tts_and_asr():
    res = _results([{"request_id": 0, "jct_ms": 500.0}], ttft_s={"audio": {"p50": 0.05, "p95": 0.07}},
                   rtf={"p50": 0.1}, audio_seconds_throughput=12.0, audio_duration_mean_s=5.0)
    st = sweep.repeat_stats(res, "tts")
    assert (st["ttfa_p50_s"], st["rtf_p50"], st["audio_s_per_s"]) == (0.05, 0.1, 12.0)
    per = [{"request_id": i, "jct_ms": 100.0} for i in range(4)]
    st = sweep.repeat_stats(_results(per, wall=2.0), "asr", input_seconds=[3.0, 5.0])
    assert st["rtfx"] == pytest.approx((3 + 5 + 3 + 5) / 2.0)


def test_summary_spread():
    reps = [{"tok_s": 100.0, "x": None}, {"tok_s": 104.0}, {"tok_s": 98.0}]
    s = sweep.summarize_repeats(reps)
    assert s["tok_s"] == {"median": 100.0, "min": 98.0, "max": 104.0, "n": 3}
    assert sweep.spread(s, "tok_s") == pytest.approx(0.06)
    assert sweep.spread(s, "missing") is None


def test_run_rows_shares_the_gpu_pool(monkeypatch):
    seen, lock = [], __import__("threading").Lock()

    def fake_row(plan, row, index, gpus, port):
        with lock:
            seen.append((row.name, tuple(gpus), port))

    monkeypatch.setattr(sweep, "run_row", fake_row)
    rows = [catalog.ROWS[n] for n in ("qwen3_5_0.8b", "qwen3_5_9b_tp2", "qwen3_5_2b", "qwen3_5_4b")]
    plan = type("P", (), {"log": staticmethod(lambda m: None)})()
    sweep.run_rows(plan, rows, [4, 5, 6], 2, 8400)
    assert sorted(n for n, _, _ in seen) == sorted(r.name for r in rows)
    assert [g for n, g, _ in seen if n == "qwen3_5_9b_tp2"] == [(4, 5)]  # wide row first
    assert len({p for _, _, p in seen}) == 4
    seen.clear()
    sweep.run_rows(plan, [catalog.ROWS["qwen3_5_9b_tp2"]], [0], 1, 8400)
    assert seen == []  # does not fit, skipped


def test_health_check_scopes_other_gpus():
    rows = [{"scope": "NODE", "status": "WARN", "check": "bare_metal", "detail": "VM"},
            {"scope": "GPU0", "status": "FAIL", "check": "occupancy", "detail": "procs=[python:1]"},
            {"scope": "GPU5", "status": "FAIL", "check": "power_cap", "detail": "500 W"},
            {"scope": "GPU1", "status": "PASS", "check": "perf_level", "detail": "auto"}]
    v = health.interpret_check(rows, own_gpus=[4, 5])
    assert v.status == "FAIL" and any("power_cap" in r for r in v.reasons)
    v = health.interpret_check(rows, own_gpus=[1])
    assert v.status == "WARN" and any("occupancy" in n for n in v.notes)
    assert health.interpret_check(rows, own_gpus=None).status == "FAIL"


def test_health_arm_verdict_and_rows():
    text = "\n".join([
        "node_health delta  a -> b",
        "GPU0   FAIL  thermal_residency        TVIOL(socket)=2.00 HBM=0.00 PROCHOT=0.00 VR=0.00 (% of window)",
        "GPU1   INFO  power_residency          PVIOL(PPT)=30.00% of window",
        "GPU1   PASS  thermal_residency        TVIOL(socket)=0.00",
    ])
    rows = health.parse_rows(text)
    assert len(rows) == 3
    summary = ("1      10      9 | 2000 1900 1800 | 700 750 10 | 80 70 | ok\n"
               "0  10 9 | 1 1 1 | 1 1 1 | 95 70 | HOT(hotspot>=90C)")
    v = health.arm_verdict(rows, summary, own_gpus=[1])
    assert v.status == "PASS" and any("PVIOL" in n for n in v.notes)
    v = health.arm_verdict(rows, summary, own_gpus=[0])
    assert v.status == "FAIL" and len(v.reasons) == 2


class _Tok:
    eos = {9: "<|im_end|>"}

    def decode(self, ids):
        return "".join(self.eos.get(i, f"t{i}") for i in ids)


def test_chunks_from_chat_vllm_ids_and_sglang_text():
    from benchmark.amd.correctness import chunks_from_chat

    vllm = {"choices": [{"logprobs": {"content": [{"token": "token_id:1"}, {"token": "token_id:2"},
                                                  {"token": "token_id:9"}]}}], "usage": {"prompt_tokens": 7}}
    assert chunks_from_chat(json.dumps(vllm).encode(), _Tok(), {9}) == (["t1", "t2"], 7)
    sgl = {"choices": [{"logprobs": {"content": [{"token": "Hi"}, {"token": " there"}, {"token": "<|im_end|>"}]}}]}
    assert chunks_from_chat(json.dumps(sgl).encode(), _Tok(), {9}) == (["Hi", " there"], None)


def test_launch_commands_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    wt = Path(__file__).resolve().parents[2]
    m = Launch(catalog.ARMS["mstar"], catalog.ROWS["qwen3_5_9b_tp2"], [4, 5], 8400, tmp_path / "a", worktree=wt)
    cmd = m.command()
    assert cmd[cmd.index("--gpus") + 1] == "4,5" and cmd[cmd.index("--config") + 1].endswith("qwen3_5_9b_tp2.yaml")
    prefix = cmd[cmd.index("--socket-path-prefix") + 1]
    assert prefix.startswith("/tmp/") and len(prefix) < 40
    env = m.environment()
    assert env["HIP_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert env["MSTAR_SHM_ARENA"] == "1" and env["PYTHONPATH"].split(":")[0] == str(wt)
    w = Launch(catalog.ARMS["vllm-whisper"], catalog.ROWS["whisper_large"], [3], 8401, tmp_path / "b", worktree=wt)
    env = w.environment()
    assert env["GPUS"] == "3" and env["VLLM_ROCM_USE_AITER"] == "1" and "AITER_JIT_DIR" not in env
    assert "CUDA_VISIBLE_DEVICES=3" in env["CTR_ENV"] and "PYTHONPATH=" in env["CTR_ENV"]
    assert "CTR_SCRATCH" not in env
    lane = Launch(catalog.ARMS["vllm"], catalog.ROWS["qwen3_5_2b"], [1], 8402, tmp_path / "d", worktree=wt,
                  lane="lane1")
    assert lane.environment()["CTR_SCRATCH"].endswith("ctr-123-vllm-openai-rocm_v0.31.0-lane1")
    assert w.command()[1:4] == ["vllm", "vllm", "serve"]
    with pytest.raises(ValueError):
        Launch(catalog.ARMS["mstar"], catalog.ROWS["qwen3_5_9b_tp2"], [4], 8400, tmp_path / "c", worktree=wt)
    assert gpu_list("0, 3") == [0, 3]
    with pytest.raises(ValueError):
        gpu_list("1,1")


def test_gpu_locks(tmp_path, monkeypatch):
    from benchmark.amd import launch

    monkeypatch.setattr(launch, "LOCK_DIR", tmp_path)
    wt = Path(__file__).resolve().parents[2]
    a = Launch(catalog.ARMS["mstar"], catalog.ROWS["qwen3_5_2b"], [2], 8400, tmp_path / "a", worktree=wt)
    b = Launch(catalog.ARMS["vllm"], catalog.ROWS["qwen3_5_2b"], [2], 8401, tmp_path / "b", worktree=wt)
    a._take_locks()
    with pytest.raises(RuntimeError, match="held by"):
        b._take_locks()
    (tmp_path / "gpu2").write_text(json.dumps({"pid": 2 ** 22 + 12345, "run_dir": "gone"}))  # dead owner
    b._take_locks()
    b._release_locks()
    assert not (tmp_path / "gpu2").exists()


def test_table_render(tmp_path):
    run = tmp_path / "run"
    meta = {"table": "qwen35-dense", "run_id": "x", "dry": False, "rows": ["qwen3_5_0.8b"],
            "arms": ["mstar", "vllm-aiter"], "workloads": ["text-chat-256"], "repeats": 3, "spread_limit": 0.05,
            "node": {"gpu_names": {"0": "AMD Instinct MI325X"}, "host": "h", "partition": "mi3258x"}, "gpus": [0]}
    run.mkdir()
    (run / "sweep.json").write_text(json.dumps(meta))
    for arm, tok, out_tok in (("mstar", 1000.0, 256), ("vllm-aiter", 900.0, 300)):
        d = run / "qwen3_5_0.8b" / arm
        (d / "cells/text-chat-256/c8").mkdir(parents=True)
        (d / "arm.json").write_text(json.dumps({"arm": arm, "row": "qwen3_5_0.8b", "status": "done",
                                                "startup": {"ready_s": 100.0, "first_request_s": 1.5}}))
        summ = {"tok_s": {"median": tok, "min": tok - 5, "max": tok + 5, "n": 3},
                "out_tokens_per_req": {"median": out_tok, "min": out_tok, "max": out_tok, "n": 3}}
        (d / "cells/text-chat-256/c8/cell.json").write_text(json.dumps(
            {"arm": arm, "row": "qwen3_5_0.8b", "workload": "text-chat-256", "concurrency": 8, "summary": summ,
             "flags": ["rerun"] if arm == "mstar" else [], "errors": [], "status": "done"}))
    (run / "correctness.json").write_text(json.dumps({"qwen3_5_0.8b/mstar": {"verdict": "PASS",
                                                                             "exact_vs_reference": "30/32"}}))
    md = table.write(run).read_text()
    assert "AMD Instinct MI325X" in md and "1000 [995-1005]" in md and "rerun" in md
    assert "PASS exact 30/32" in md and "output tokens/request 300 vs mstar 256" in md
    assert json.loads((run / "table.json").read_text())["meta"]["table"] == "qwen35-dense"


def test_sample_set_dataset(tmp_path):
    from benchmark.base import RequestType
    from benchmark.dataset import SampleSetDataset

    (tmp_path / "audio").mkdir()
    (tmp_path / "audio/0.wav").write_bytes(b"RIFF")
    (tmp_path / "manifest.json").write_text(json.dumps({"items": [
        {"id": "000", "prompt": "", "files": {"audio": "audio/0.wav"}},
        {"id": "001", "prompt": "hello", "files": {"audio": "audio/0.wav"}}]}))
    ds = SampleSetDataset(str(tmp_path), 5, RequestType.A2T)
    reqs = ds.get_requests()
    assert len(reqs) == 5 and reqs[0].audio_path.endswith("audio/0.wav") and reqs[0].get_bytes("audio") == b"RIFF"
    assert ds.item_ids == ["000", "001"]
    text = SampleSetDataset(str(tmp_path), 2, RequestType.T2S).get_requests()
    assert [r.prompt for r in text] == ["", "hello"] and text[1].audio_path is None


def test_runner_tags_prompts_and_chat_template_kwargs():
    from benchmark.base import ModelType, RequestType
    from benchmark.request import OpenAIText, OursOpenAI, RequestInput
    from benchmark.runner import Benchmark, BenchmarkConfig, DatasetType, InferenceSystemType

    def make(system, unique):
        cfg = BenchmarkConfig(url="http://x", model=ModelType.QWEN3_5.inst(), dataset=DatasetType.TEXT,
                              num_requests=1, request_type=RequestType.T2T, local_cache_dir="/tmp",
                              inference_system=system, chat_template_kwargs={"enable_thinking": False},
                              unique_prompts=unique)
        return Benchmark(cfg)

    b = make(InferenceSystemType.VLLM, True)
    assert isinstance(b.inference_system, OpenAIText)
    assert b.inference_system._extra_fields() == {"chat_template_kwargs": {"enable_thinking": False}}
    req = RequestInput(req_type=RequestType.T2T, prompt="hi")
    a, c = b._tag_prompt(req, "w0"), b._tag_prompt(req, "m0")
    assert a.prompt != c.prompt and a.prompt.endswith("] hi") and req.prompt == "hi"
    assert make(InferenceSystemType.VLLM, False)._tag_prompt(req, "m0") is req
    payload = b.inference_system._finish_payload({"max_tokens": 4, "max_output_tokens": 4}, "text")
    assert payload == {"max_tokens": 4}
    ours = make(InferenceSystemType.OURS_OPENAI, False).inference_system
    assert isinstance(ours, OursOpenAI) and ours._extra_fields() == {"enable_thinking": False}
    assert ours._finish_payload({}, "text") == {"modalities": ["text"]}
    with pytest.raises(ValueError):
        make(InferenceSystemType.OURS, False)


def test_gpu_query_parsers():
    from benchmark import gpu_query

    assert gpu_query.parse_nvidia_processes("11, 100\n\n12, 2.5\n") == [(11, 100.0), (12, 2.5)]
    doc = [{"gpu": 1, "process_list": [{"process_info": {"pid": 7, "memory_usage": {
        "vram_mem": {"value": 2 * 2**20, "unit": "B"}}}}]}, {"gpu": 0, "process_list": []}]
    assert gpu_query.parse_amd_processes(doc, 1) == [(7, 2.0)]
    assert gpu_query.parse_amd_processes(doc, 0) == []
    assert gpu_query._mib({"value": 3, "unit": "GB"}) == 3072 and gpu_query._mib(5) == 5.0


def test_onnx_providers():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "benchmark/kokoro_onnx_baselines.py"
    spec = importlib.util.spec_from_file_location("kokoro_onnx_baselines", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    amd = ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
    assert mod.onnx_providers("auto", amd) == ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
    assert mod.onnx_providers("rocm", amd) == ["ROCMExecutionProvider", "CPUExecutionProvider"]
    assert mod.onnx_providers("cpu", amd) == ["CPUExecutionProvider"]
    assert mod.onnx_providers("auto", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    with pytest.raises(RuntimeError):
        mod.onnx_providers("cuda", amd)


def test_reference_arms_only_serve_their_model_and_split_gpus(tmp_path):
    names = ["mstar", "ref-kokoro", "ref-orpheus-vllm"]
    assert [a.name for a in catalog.arms_for(catalog.ROWS["kokoro"], names)] == ["mstar", "ref-kokoro"]
    assert [a.name for a in catalog.arms_for(catalog.ROWS["orpheus"], names)] == ["mstar", "ref-orpheus-vllm"]
    wt = Path(__file__).resolve().parents[2]
    arm = catalog.ARMS["ref-orpheus-vllm"]
    two = Launch(arm, catalog.ROWS["orpheus"], [6, 7], 8400, tmp_path / "a", worktree=wt)
    assert two.server_gpus == [6] and two.client_gpu == 7 and two.environment()["GPUS"] == "6"
    one = Launch(arm, catalog.ROWS["orpheus_colocated"], [3], 8400, tmp_path / "b", worktree=wt)
    assert one.server_gpus == [3] and one.client_gpu == 3
    env = sweep.client_env(arm, catalog.ROWS["orpheus"], [6, 7])
    assert env["HIP_VISIBLE_DEVICES"] == "7" and catalog.REFENV in env["PYTHONPATH"].split(":")
    assert sweep.client_env(catalog.ARMS["vllm"], catalog.ROWS["qwen3_5_2b"], [1])["HIP_VISIBLE_DEVICES"] == ""


def test_orpheus_vllm_decode_window():
    import torch

    from benchmark.request import OrpheusVLLM

    seen = {}

    class FakeSnac:
        def decode(self, levels):
            seen["levels"] = [lv.tolist() for lv in levels]
            return torch.linspace(-1, 1, 8192).view(1, 1, -1)

    c = OrpheusVLLM(device="cpu")
    c._torch, c._snac = torch, FakeSnac()
    codes = list(range(1, 29))
    pcm = c._decode(codes)
    assert seen["levels"][0] == [[1, 8, 15, 22]]
    assert seen["levels"][1] == [[2, 5, 9, 12, 16, 19, 23, 26]]
    assert seen["levels"][2][0][:4] == [3, 4, 6, 7]
    assert len(pcm) == 2 * 2048  # the middle 2048 samples, int16
    assert c._decode([5000] + codes[1:]) is None


def test_vllm_omni_deploy_overlays():
    import yaml

    tts = catalog.ROWS["qwen3tts"]
    args, env = catalog.server_args(catalog.ARMS["vllm-omni-tuned"], tts)
    deploy = Path(args[args.index("--deploy-config") + 1])
    assert deploy.is_file() and env == {"VLLM_ROCM_USE_AITER": "1", "MIOPEN_FIND_MODE": "FAST"}
    doc = yaml.safe_load(deploy.read_text())
    stages = {s["stage_id"]: s for s in doc["platforms"]["rocm"]["stages"]}
    assert doc["base_config"].endswith("/vllm_omni/deploy/qwen3_tts.yaml")
    assert stages[0]["attention_backend"] == "ROCM_AITER_FA" and stages[1]["enforce_eager"] is False
    assert "--deploy-config" not in catalog.server_args(catalog.ARMS["vllm-omni"], tts)[0]
    one = catalog.ROWS["qwen3omni_colocated"]
    for arm, name in (("vllm-omni", "qwen3_omni_one_gpu.yaml"),
                      ("vllm-omni-tuned", "qwen3_omni_one_gpu_rocm_tuned.yaml")):
        args, env = catalog.server_args(catalog.ARMS[arm], one)
        assert args[args.index("--deploy-config") + 1].endswith(name)
        assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    placed = yaml.safe_load((catalog.VLLM_OMNI_DEPLOY / "qwen3_omni_one_gpu.yaml").read_text())
    assert {s["devices"] for s in placed["stages"]} == {"0"}
    assert sum(s["gpu_memory_utilization"] for s in placed["stages"]) < 0.95
    two = catalog.ROWS["qwen3omni_2gpu"]
    assert "--deploy-config" not in catalog.server_args(catalog.ARMS["vllm-omni"], two)[0]


def test_sample_set_mixed_modalities_and_generator_params(tmp_path):
    from benchmark.base import RequestType
    from benchmark.dataset import SampleSetDataset

    (tmp_path / "a.wav").write_bytes(b"RIFF")
    (tmp_path / "i.jpg").write_bytes(b"\xff\xd8")
    (tmp_path / "manifest.json").write_text(json.dumps({"items": [
        {"id": "img", "prompt": "what food?", "files": {"image": "i.jpg"}, "params": {"seed": 3}},
        {"id": "aud", "prompt": "what is said?", "files": {"audio": "a.wav"}, "params": {"seed": 4}}]}))
    audio = SampleSetDataset(str(tmp_path), 2, RequestType.A2S)
    assert audio.item_ids == ["aud"] and all(r.audio_path for r in audio.get_requests())
    assert audio.get_requests()[0].model_kwargs == {}  # params only go to generators
    with pytest.raises(ValueError):
        SampleSetDataset(str(tmp_path), 1, RequestType.V2S)
    (tmp_path / "manifest.json").write_text(json.dumps({"items": [
        {"id": "00", "prompt": "a fox", "params": {"seed": 0, "height": 768, "width": 1344, "other": 1}}]}))
    req = SampleSetDataset(str(tmp_path), 1, RequestType.T2I).get_requests()[0]
    assert req.model_kwargs == {"seed": 0, "height": 768, "width": 1344}


def test_media_fields_and_runner_flags(tmp_path):
    from benchmark.base import ModelType, RequestType
    from benchmark.request import RequestInput, _media_fields

    req = RequestInput(req_type=RequestType.T2I, prompt="p", model_kwargs={"seed": 2, "height": 768, "width": 1344})
    fields = _media_fields(ModelType.FLUX2_KLEIN.inst(), req, {"num_inference_steps": 4})
    assert fields == {"seed": 2, "size": "1344x768", "num_inference_steps": 4}
    img = sweep.runner_command(catalog.ARMS["sglang-diffusion"], catalog.ROWS["z_image_turbo"],
                               tables.WORKLOADS["image-gen"], 4, 8400, tmp_path / "r", 1,
                               request_kwargs={"num_inference_steps": 8, "guidance_scale": 0.0})
    assert img[img.index("--inference-system") + 1] == "sglang_diffusion"
    assert json.loads(img[img.index("--request-kwargs") + 1]) == {"num_inference_steps": 8, "guidance_scale": 0.0}
    omni = sweep.runner_command(catalog.ARMS["vllm-omni"], catalog.ROWS["qwen3omni_colocated"],
                                tables.WORKLOADS["omni-audio-speech"], 4, 8400, tmp_path / "r", 1)
    assert omni[omni.index("--chat-audio-format") + 1] == "pcm" and "--ignore-eos" not in omni
    ours = sweep.runner_command(catalog.ARMS["mstar"], catalog.ROWS["qwen3omni_colocated"],
                                tables.WORKLOADS["omni-audio-speech"], 4, 8400, tmp_path / "r", 1)
    assert "--chat-audio-format" not in ours and ours[ours.index("--inference-system") + 1] == "ours_openai"
    asr = sweep.runner_command(catalog.ARMS["mstar"], catalog.ROWS["whisper_large"], tables.WORKLOADS["asr-libri"],
                               1, 8400, tmp_path / "r", 1)
    assert asr[asr.index("--inference-system") + 1] == "openai_transcribe"


def test_repeat_stats_omni_image_video():
    st = sweep.repeat_stats(_results([{"request_id": 0, "jct_ms": 900.0}], ttft_s={"audio": {"p50": 0.3},
                                                                                     "text": {"p50": 0.1}},
                                     rtf={"p50": 0.2}, audio_seconds_throughput=8.0), "omni")
    assert (st["ttfa_p50_s"], st["text_ttft_p50_s"], st["audio_s_per_s"]) == (0.3, 0.1, 8.0)
    assert sweep.repeat_stats(_results([{"request_id": 0, "jct_ms": 1.0}], wall=4.0), "image")["images_per_s"] == 0.25
    assert sweep.repeat_stats(_results([{"request_id": 0, "jct_ms": 1.0}], wall=30.0), "video")["clips_per_min"] == 2.0


def test_media_request_kwargs_come_from_the_correctness_families():
    from benchmark.amd import correctness

    wt = Path(__file__).resolve().parents[2]
    plan = type("P", (), {"correctness_src": wt, "worktree": wt})()
    assert correctness.media_request_kwargs(plan, catalog.ROWS["flux2_klein"]) == {
        "num_inference_steps": 4, "guidance_scale": 1.0}
    assert correctness.media_request_kwargs(plan, catalog.ROWS["z_image_turbo"]) == {
        "num_inference_steps": 8, "guidance_scale": 0.0}
    wan = correctness.media_request_kwargs(plan, catalog.ROWS["wan22"])
    assert {k: wan[k] for k in ("height", "width", "num_frames", "num_inference_steps", "fps", "guidance_scale")} == {
        "height": 480, "width": 832, "num_frames": 33, "num_inference_steps": 20, "fps": 24, "guidance_scale": 5.0}
    assert wan["negative_prompt"] == ""


def test_omni_chat_body_and_stream():
    import base64

    from benchmark.amd import correctness

    item = type("I", (), {"params": {"modality": "image"}, "files": {"image": "x.jpg"}, "prompt": "what food?"})()
    sset = type("S", (), {"path": staticmethod(lambda rel: Path(__file__))})()
    fam = type("F", (), {"max_new_tokens": 128})()
    body = correctness.omni_chat_body(fam, sset, item, 7, "Qwen/Qwen3-Omni-30B-A3B-Instruct", "You are Qwen.")
    user = body["messages"][1]["content"]
    assert user[0]["type"] == "image_url" and user[0]["image_url"]["url"].startswith("data:")
    assert body["modalities"] == ["text", "audio"] and body["audio"] == {"format": "pcm"} and body["seed"] == 7
    pcm = b"\x01\x00\x02\x00"
    events = [{"choices": [{"delta": {"content": "Pas"}}]}, {"choices": [{"delta": {"content": "ta"}}]},
              {"modality": "audio", "choices": [{"delta": {"content": base64.b64encode(pcm).decode()}}]},
              {"choices": [{"delta": {"audio": {"data": base64.b64encode(pcm).decode()}}}]}]
    stream = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"
    assert correctness.parse_omni_stream(stream) == (["Pas", "ta"], pcm + pcm)


def test_long_prompt_cuts_phase_records_by_cell_window():
    import datetime

    from benchmark.amd import long_prompt

    def line(t, bs, exec_ms, sync_ms, it_ms):
        ts = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        return (f"{ts} INFO [worker_0] mstar.worker.worker: Worker worker_0 phase-timing iter={int(t)} bs={bs}: "
                f"iter_total: p50={it_ms}ms p95={it_ms}ms mean={it_ms}ms n=20 | "
                f"worker.gpu_thread.exec: p50={exec_ms}ms p95={exec_ms}ms mean={exec_ms}ms n=20 | "
                f"worker.postprocess.event_sync: p50={sync_ms}ms p95={sync_ms}ms mean={sync_ms}ms n=20")

    t = 1_800_000_000.0
    log = "\n".join([line(t + k, 1.0, 40.0, 1.0, 50.0) for k in range(4)]
                    + [line(t + 100 + k, 1.0, 5.0, 1.0, 10.0) for k in range(4)])
    cut = long_prompt.cut_by_window(log, {"prefill-long": (t - 1, t + 10), "prefill-short": (t + 99, t + 110),
                                          "burst-long": (t + 500, t + 510)})
    assert cut["prefill-long"]["phases"]["worker.gpu_thread.exec"]["mean_ms"] == 40.0
    assert cut["burst-long"]["phases"] == {}
    out = long_prompt.readout(cut, refusals=3)
    assert out["prefill-long"]["exec_share"] == 0.8 and out["prefill-long"]["event_sync_share"] == 0.02
    assert out["exec_ratio_long_over_short"] == 8.0 and out["prefill_buckets_running_eagerly"] == 3
    names = [wl.name for wl, _ in long_prompt.cells(Path("/tmp/p400.txt"))]
    assert names == ["prefill-long", "prefill-short", "burst-long", "burst-short"]
