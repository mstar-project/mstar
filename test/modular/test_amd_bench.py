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
    dense = list(tables.TABLES["qwen35-dense"].arms)
    assert "mstar-tpasync" in [a.name for a in catalog.arms_for(catalog.ROWS["qwen3_5_9b_tp2"], dense)]
    assert "mstar-tpasync" not in [a.name for a in catalog.arms_for(catalog.ROWS["qwen3_5_9b"], dense)]
    _, env = catalog.server_args(catalog.ARMS["mstar-tpasync"], catalog.ROWS["qwen3_5_9b_tp2"])
    assert env == {"MSTAR_SHM_ARENA": "1", "MSTAR_TP_ASYNC_SCHED": "1", "GLOO_SOCKET_IFNAME": "lo"}
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


def test_resume_refuses_another_chip(tmp_path):
    mi325 = {"node_type": "mi325x", "gpu_names": {"0": "AMD Instinct MI325X"}}
    mi300 = {"node_type": "mi300x", "gpu_names": {"0": "AMD Instinct MI300X"}}
    sweep.refuse_mixed_chips(mi325, {**mi325, "host": "other"}, tmp_path)
    for now in (mi300, {**mi325, "gpu_names": {"0": "AMD Instinct MI300X"}}):
        with pytest.raises(SystemExit):
            sweep.refuse_mixed_chips(mi325, now, tmp_path)


def test_resume_of_part_of_a_table_keeps_the_rest(tmp_path):
    table_ = tables.TABLES["qwen35-dense"]
    prev = {"started": "t0", "rows": list(table_.rows), "arms": ["mstar", "vllm", "vllm-aiter", "sglang-aiter"],
            "workloads": list(table_.workloads), "concurrency": {"text-chat-256": [1, 64]}, "gpus": [0, 1, 2, 4, 5, 6],
            "lanes": 5, "numa": True, "resumed": [{"at": "t1", "job": "1"}]}
    meta = {"started": "t2", "node": {"host": "k002-002"}, "job": "468005", "gpus": [0, 1, 4], "lanes": 2,
            "numa": True, "driver": "/d", "rows": ["qwen3_5_4b", "qwen3_5_9b_tp2"], "arms": ["mstar", "mstar-tpasync"],
            "workloads": ["text-chat-256"], "concurrency": {"text-chat-256": [1, 4]}}
    out = sweep.resumed_meta(meta, prev, table_)
    assert out["rows"] == list(table_.rows) and out["workloads"] == list(table_.workloads)
    assert out["arms"] == ["mstar", "mstar-tpasync", "vllm", "vllm-aiter", "sglang-aiter"]
    assert out["started"] == "t0" and out["gpus"] == [0, 1, 2, 4, 5, 6] and out["lanes"] == 5
    assert out["concurrency"] == {"text-chat-256": [1, 4]}
    assert [r["job"] for r in out["resumed"]] == ["1", "468005"]
    assert out["resumed"][-1]["gpus"] == [0, 1, 4] and out["resumed"][-1]["lanes"] == 2


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


def test_arms_of_a_row_get_their_own_ports(monkeypatch):
    ports = []
    monkeypatch.setattr(sweep, "run_arm", lambda plan, row, arm, gpus, port, lane: ports.append((arm.name, port)))
    plan = type("P", (), {"arms": ["mstar", "vllm-omni", "vllm-omni-tuned", "sglang-omni"], "trees": {}})()
    sweep.run_row(plan, catalog.ROWS["qwen3tts_1p7b"], 1, [3], 8480)
    assert sorted(p for _, p in ports) == [8480, 8481, 8482, 8483] and ports[0][0] == "vllm-omni"


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
    cfg = tmp_path / "cap1024.yaml"
    cfg.write_text("model: qwen3_5\nresources:\n  linear_attn:\n    max_captured_prefill_tokens: 1024\n")
    exp = Launch(catalog.ARMS["mstar"], catalog.ROWS["qwen3_5_2b"], [1], 8403, tmp_path / "e", worktree=wt, config=cfg)
    assert exp.command()[exp.command().index("--config") + 1] == str(cfg)
    with pytest.raises(ValueError):
        Launch(catalog.ARMS["mstar"], catalog.ROWS["qwen3_5_9b_tp2"], [4], 8400, tmp_path / "c", worktree=wt)
    assert gpu_list("0, 3") == [0, 3]
    with pytest.raises(ValueError):
        gpu_list("1,1")


def test_sglang_omni_launch(tmp_path):
    from benchmark.amd.launch import backends_in_log

    wt = Path(__file__).resolve().parents[2]
    arm, row = catalog.ARMS["sglang-omni"], catalog.ROWS["qwen3tts_1p7b"]
    s = Launch(arm, row, [2], 8403, tmp_path / "so", worktree=wt)
    cmd = s.command()
    assert cmd[cmd.index("--endpoints.base_path") + 1] == str(s.ipc_dir)
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--config") + 1] == "examples/configs/qwen3_tts_1_7b_customvoice.yaml"
    env = s.environment()
    # ctr.sh reads CTR_PWD on the host; the SGLANG_ limits reach the container by prefix.
    assert env["CTR_PWD"] == "/workspace/sglang-omni" and "CTR_PWD" not in env["CTR_ENV"]
    assert env["SGLANG_OMNI_STARTUP_TIMEOUT"] == "2700" and env["SGLANG_OMNI_WARMUP_TIMEOUT"] == "1800"
    assert s.startup_wait >= 2700 + 1800
    omni = Launch(catalog.ARMS["mstar"], catalog.ROWS["qwen3omni_colocated"], [3], 8404, tmp_path / "om", worktree=wt)
    assert omni.startup_wait == 5400
    assert Launch(catalog.ARMS["mstar"], catalog.ROWS["kokoro"], [3], 8405, tmp_path / "ko",
                  worktree=wt).startup_wait == 3600
    t = tables.TABLES["tts"]
    assert [a.name for a in catalog.arms_for(row, list(t.arms))] == ["mstar", "mstar-colocated", "mstar-base",
                                                                     "vllm-omni", "vllm-omni-tuned", "sglang-omni"]
    log = tmp_path / "server.log"
    log.write_text("SGLang runtime configuration: gpu_id=0, sm=94, attention_backend=aiter, "
                   "decode_attention_backend=aiter, prefill_attention_backend=aiter, sampling_backend=pytorch\n")
    assert backends_in_log(log) == ["aiter"]


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


def test_mstar_arms_serve_their_own_configs(tmp_path):
    wt = tmp_path / "wt"
    (wt / "configs").mkdir(parents=True)
    for name in ("qwen3tts_1p7b", "qwen3tts_1p7b_split_bf16", "kokoro"):
        (wt / "configs" / f"{name}.yaml").write_text(f"model: {name}\n")
    row = catalog.ROWS["qwen3tts_1p7b"]
    tts = list(tables.TABLES["tts"].arms)
    assert [a.name for a in catalog.arms_for(row, tts)][:2] == ["mstar", "mstar-colocated"]
    assert "mstar-colocated" not in [a.name for a in catalog.arms_for(catalog.ROWS["kokoro"], tts)]

    def served(arm, r=row, gpu=2):
        cmd = Launch(catalog.ARMS[arm], r, [gpu], 8400, tmp_path / arm / r.name, worktree=wt).command()
        return Path(cmd[cmd.index("--config") + 1]).name, cmd[cmd.index("serve") + 1]

    assert served("mstar") == ("qwen3tts_1p7b_split_bf16.yaml", "qwen3tts_1p7b_split_bf16")
    assert served("mstar-colocated") == ("qwen3tts_1p7b.yaml", "qwen3tts_1p7b")
    assert served("mstar", catalog.ROWS["kokoro"]) == ("kokoro.yaml", "kokoro")
    assert catalog.mstar_config_name(row, "vllm-omni") == "qwen3tts_1p7b"
    # A config the tree under test lacks fails the arm at start instead of serving another one.
    (wt / "configs" / "qwen3tts_1p7b_split_bf16.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        served("mstar")


def test_base_arms_serve_the_base_tree(tmp_path, monkeypatch):
    from benchmark.amd.launch import arm_worktree, tree_list

    head, base = tmp_path / "head", tmp_path / "base"
    for t in (head, base):
        (t / "configs").mkdir(parents=True)
        for name in ("qwen3tts_1p7b", "qwen3tts_1p7b_split_bf16"):
            (t / "configs" / f"{name}.yaml").write_text(f"model: {name}\n")
    trees = tree_list([f"base={base}"])
    assert trees == {"base": base.resolve()}
    for bad in ("base", f"={base}", f"base={tmp_path / 'nope'}"):
        with pytest.raises(ValueError):
            tree_list([bad])
    mstar, mstar_base = catalog.ARMS["mstar"], catalog.ARMS["mstar-base"]
    assert arm_worktree(mstar, head, trees) == head and arm_worktree(mstar_base, head, {}) is None
    # The base arm serves its own tree's default config, the single-worker one on Qwen3-TTS 1.7B.
    row = catalog.ROWS["qwen3tts_1p7b"]
    b = Launch(mstar_base, row, [2], 8400, tmp_path / "b", worktree=arm_worktree(mstar_base, head, trees))
    assert b.mstar_config() == base.resolve() / "configs" / "qwen3tts_1p7b.yaml"
    seen = []
    monkeypatch.setattr(sweep, "run_arm", lambda plan, row, arm, gpus, port, lane: seen.append((arm.name, port)))
    dense, tp2 = list(tables.TABLES["qwen35-dense"].arms), catalog.ROWS["qwen3_5_9b_tp2"]
    # Without --tree base the row runs as before: same arms, order and ports.
    sweep.run_row(type("P", (), {"arms": dense, "trees": {}})(), tp2, 0, [0, 1], 8400)
    assert seen == [("mstar", 8400), ("mstar-tpasync", 8401), ("vllm", 8402), ("vllm-aiter", 8403),
                    ("sglang-aiter", 8404)]
    seen.clear()
    sweep.run_row(type("P", (), {"arms": dense, "trees": trees})(), tp2, 0, [0, 1], 8400)
    assert [n for n, _ in seen][:4] == ["mstar", "mstar-tpasync", "mstar-base", "mstar-base-tpasync"]


def test_correctness_reads_each_arm_s_tree(tmp_path):
    from benchmark.amd import correctness

    d = tmp_path / "arm"
    (d / "server").mkdir(parents=True)
    assert correctness._arm_tree(d, tmp_path) == tmp_path
    (d / "server" / "versions.json").write_text(json.dumps({"git": {"worktree": "/trees/base"}}))
    assert correctness._arm_tree(d, tmp_path) == Path("/trees/base")


def test_mstar_rust_must_come_from_a_tree_that_builds_its_own(tmp_path):
    from benchmark.amd.launch import rust_from_worktree

    venv = {"mstar_rust_file": "/venv/site-packages/mstar_rust/__init__.py"}
    assert rust_from_worktree(venv, tmp_path) is False  # no build in the tree: recorded, not refused
    (tmp_path / "mstar_rust").mkdir()
    with pytest.raises(RuntimeError):
        rust_from_worktree(venv, tmp_path)
    with pytest.raises(RuntimeError):
        rust_from_worktree({"mstar_rust_error": "ImportError()"}, tmp_path)
    assert rust_from_worktree({"mstar_rust_file": str(tmp_path / "mstar_rust" / "__init__.py")}, tmp_path) is True


def test_table_render(tmp_path):
    run = tmp_path / "run"
    meta = {"table": "qwen35-dense", "run_id": "x", "dry": False, "rows": ["qwen3_5_0.8b"],
            "arms": ["mstar", "vllm-aiter"], "workloads": ["text-chat-256"], "repeats": 3, "spread_limit": 0.05,
            "node": {"gpu_names": {"0": "AMD Instinct MI325X"}, "host": "h", "partition": "mi3258x"}, "gpus": [0],
            "resumed": [{"at": "2026-10-19T09:00:00-0500", "host": "h", "job": "1"},
                        {"at": "2026-10-20T03:00:00-0500", "host": "h", "job": "2", "gpus": [0, 1, 4], "lanes": 2,
                         "measured": ["qwen3_5_4b/mstar", "qwen3_5_9b_tp2/mstar-tpasync"]}]}
    run.mkdir()
    (run / "sweep.json").write_text(json.dumps(meta))
    (run / "health").mkdir()
    for name in ("check_before", "check_resume1"):
        (run / "health" / f"{name}.json").write_text(json.dumps({"verdict": {"status": "PASS", "reasons": []}}))
    (run / "health" / "check_after.json").write_text(json.dumps({"verdict": {"status": "WARN", "reasons": [
        "NODE bare_metal: VM/SR-IOV slice (virt=none, ifwi=AMD MI325X_SRIOV_CVS_1VF): use for correctness/dev, not "
        "for throughput tables"]}}))
    (run / "health" / "check_resume1.raw.json").write_text("{}")
    for arm, tok, out_tok in (("mstar", 1000.0, 256), ("vllm-aiter", 900.0, 300)):
        d = run / "qwen3_5_0.8b" / arm
        (d / "cells/text-chat-256/c8").mkdir(parents=True)
        first = {"image-gen": {"1024x1024": 35.2, "1344x768": "failed (HTTP 500, see x)"}} if arm == "mstar" else None
        (d / "arm.json").write_text(json.dumps({"arm": arm, "row": "qwen3_5_0.8b", "status": "done",
                                                "startup": {"ready_s": 100.0, "first_request_s": 1.5},
                                                "first_size": first}))
        if arm == "mstar":
            (d / "server").mkdir()
            (d / "server" / "versions.json").write_text(json.dumps(
                {"git": {"branch": "HEAD", "head": "6e6474b62abc"}, "config": "/wt/configs/qwen3_5_0.8b.yaml",
                 "rust_from_worktree": False, "stack": {"torch": "2.12.1"}}))
        summ = {"tok_s": {"median": tok, "min": tok - 5, "max": tok + 5, "n": 3},
                "out_tokens_per_req": {"median": out_tok, "min": out_tok, "max": out_tok, "n": 3}}
        # mstar's client-bound comes from the timed-window measure and stays; vllm-aiter's predates it and goes.
        loop = {"client_loop_frac": 0.95} if arm == "mstar" else {"client_cpu_frac": 1.3}
        (d / "cells/text-chat-256/c8/cell.json").write_text(json.dumps(
            {"arm": arm, "row": "qwen3_5_0.8b", "workload": "text-chat-256", "concurrency": 8, "summary": summ,
             "flags": ["client-bound", "rerun"] if arm == "mstar" else ["client-bound"], "errors": [],
             "status": "done", "attempts": [{"repeats": [loop]}]}))
        # Served arms record co-tenants per repeat, reference arms once per cell.
        rec = d / "cells/text-chat-256/c8" / ("a1r1" if arm == "mstar" else "")
        rec.mkdir(exist_ok=True)
        (rec / "cotenants.json").write_text(json.dumps(
            {"markers": ["ws-j", "ws-j-moe-tune"] if arm == "mstar" else ["ws-j"], "loadavg": [12.4, 9.0, 7.0]}))
    (run / "correctness.json").write_text(json.dumps({"qwen3_5_0.8b/mstar": {"verdict": "PASS",
                                                                             "exact_vs_reference": "30/32"}}))
    (run / "notes.txt").write_text("HIP 4 runs hot on this node.\n\n")
    void = run / "qwen3_5_0.8b" / "mstar.health-fail-1"
    void.mkdir()
    (void / "arm.json").write_text(json.dumps({"arm": "mstar", "row": "qwen3_5_0.8b", "status": "done"}))
    md = table.write(run).read_text()
    assert "AMD Instinct MI325X" in md and "1000 [995-1005]" in md and "rerun" in md
    assert "| mstar HEAD 6e6474b62, configs/qwen3_5_0.8b.yaml | torch 2.12.1 mstar_rust from the venv |" in md
    assert "\n- note: HIP 4 runs hot on this node.\n" in md
    assert "| 100.0, 1.5; first request per size: image-gen 1024x1024 35.2 s, 1344x768 failed |" in md
    assert md.count("first request per size") == 1
    assert "| client-bound rerun |" in md and md.count("client-bound") == 1
    assert "resumed 2026-10-19T09:00:00-0500 on h (job 1); finished" in md and "node health check_resume1: PASS" in md
    assert ("resumed 2026-10-20T03:00:00-0500 on h (job 2), GPUs [0, 1, 4], lanes 2; measured qwen3_5_4b/mstar, "
            "qwen3_5_9b_tp2/mstar-tpasync; finished arms and cells kept") in md
    assert "  - NODE bare_metal: VM/SR-IOV slice (virt=none" in md
    assert "| ws-j-moe-tune; peak load 12 |" in md and "| none; peak load 12 |" in md
    assert "PASS exact 30/32" in md and "output tokens/request 300 vs mstar 256" in md
    assert json.loads((run / "table.json").read_text())["meta"]["table"] == "qwen35-dense"


def test_correctness_scores_skip_set_aside_arms(tmp_path, monkeypatch):
    from benchmark.amd import correctness

    run = tmp_path / "run"
    run.mkdir()
    (run / "sweep.json").write_text(json.dumps({"worktree": str(tmp_path), "correctness_src": str(tmp_path)}))
    # Path order puts "mstar" before "mstar.hip4-1", so a set-aside run scored last would win the arm's key.
    for d in ("mstar", "mstar.hip4-1"):
        (run / "qwen3_5_4b" / d / "correctness").mkdir(parents=True)
        (run / "qwen3_5_4b" / d / "arm.json").write_text(json.dumps({"arm": "mstar", "row": "qwen3_5_4b"}))
        (run / "qwen3_5_4b" / d / "correctness" / "pass.json").write_text("{}")
    monkeypatch.setattr(correctness, "load", lambda src: None)
    monkeypatch.setattr(correctness, "score_dir", lambda C, wt, corr, device: {"verdict": corr.parent.name})
    assert correctness.score(run) == {"qwen3_5_4b/mstar": {"verdict": "mstar"}}


def test_one_arms_scoring_error_keeps_the_others(tmp_path, monkeypatch):
    from benchmark.amd import correctness

    run = tmp_path / "run"
    (run / "wan22").mkdir(parents=True)
    (run / "sweep.json").write_text(json.dumps({"worktree": str(tmp_path), "correctness_src": str(tmp_path)}))
    for arm in ("mstar", "vllm-omni"):
        d = run / "wan22" / arm
        (d / "cells/video-t2v/c1/a1r1").mkdir(parents=True)
        (d / "cells/video-t2v/c1/a1r1/results.json").write_text("{}")
        (d / "cells/video-i2v/c1/a1r1").mkdir(parents=True)
        (d / "cells/video-i2v/c1/a1r1/results.json").write_text("{}")
        (d / "arm.json").write_text(json.dumps({"arm": arm, "row": "wan22"}))

    def fake_pass(C, tree, row, arm, cell, corr, ext, opts):
        if arm == "vllm-omni" and cell.parent.name == "video-t2v":
            raise FileNotFoundError("no outputs")
        corr.mkdir(parents=True, exist_ok=True)
        (corr / "pass.json").write_text("{}")

    monkeypatch.setattr(correctness, "load", lambda src: None)
    monkeypatch.setattr(correctness, "pass_from_cell", fake_pass)
    monkeypatch.setattr(correctness, "score_dir", lambda C, wt, corr, device: {"verdict": "PASS"})
    got = correctness.score(run)
    assert got["wan22/vllm-omni"] == {"verdict": "ERROR", "error": "FileNotFoundError: no outputs"}
    assert got["wan22/mstar"] == got["wan22/mstar/video-i2v"] == got["wan22/vllm-omni/video-i2v"] == {
        "verdict": "PASS"}


def test_moe_configs_reach_the_table(tmp_path):
    from benchmark.amd.launch import moe_configs_in_log

    thinker = "E=128,N=768,K=2048,topk=8,device_name=AMD_Instinct_MI325X.json"
    log = tmp_path / "server.log"
    log.write_text(f"INFO [worker_0] mstar.utils.fused_moe.config: fused MoE: tuned Triton configs from {thinker}\n"
                   "INFO Using configuration from /v/configs/E=128,N=768,device_name=AMD_Instinct_MI325X.json for "
                   "MoE layer.\nWARNING Using default MoE config. Performance might be sub-optimal! Config file not "
                   "found at /u/E=128,N=384,device_name=AMD_Instinct_MI325X.json, "
                   "/v/configs/E=128,N=384,device_name=AMD_Instinct_MI325X.json\n")
    got = moe_configs_in_log(log)
    assert got == {"tuned": [thinker, "E=128,N=768,device_name=AMD_Instinct_MI325X.json"],
                   "default": ["E=128,N=384,device_name=AMD_Instinct_MI325X.json"]}
    # mstar logs only the tables it finds: a missing Talker table is flagged, not passed over.
    run = tmp_path / "run"
    meta = {"table": "omni", "run_id": "x", "rows": ["qwen3omni_colocated"], "arms": ["mstar"],
            "workloads": ["omni-text-speech"], "node": {"gpu_names": {"0": "AMD Instinct MI325X"}}}
    d = run / "qwen3omni_colocated" / "mstar"
    (d / "cells/omni-text-speech/c1").mkdir(parents=True)
    (run / "sweep.json").write_text(json.dumps(meta))
    (d / "arm.json").write_text(json.dumps({"arm": "mstar", "row": "qwen3omni_colocated", "status": "done",
                                            "startup": {"moe_configs": {"tuned": [thinker], "default": []}}}))
    (d / "cells/omni-text-speech/c1/cell.json").write_text(json.dumps(
        {"arm": "mstar", "row": "qwen3omni_colocated", "workload": "omni-text-speech", "concurrency": 1,
         "summary": {}, "flags": [], "errors": [], "status": "done"}))
    md = table.write(run).read_text()
    assert "MoE E=128,N=768,K=2048,topk=8: AMD_Instinct_MI325X table" in md
    assert "MoE E=128,N=384,K=1024,topk=6: NO tuned table, default launch config" in md and "moe-untuned" in md


def test_aitermoe_arm_serves_a_config_copy_with_its_model_kwargs_right_after_mstar(tmp_path, monkeypatch):
    import yaml

    wt = tmp_path / "wt"
    (wt / "configs").mkdir(parents=True)
    (wt / "configs" / "qwen3omni_colocated.yaml").write_text("model: qwen3_omni\nmax_seq_len: 32768\n")
    row = catalog.ROWS["qwen3omni_colocated"]
    launch = Launch(catalog.ARMS["mstar-aitermoe"], row, [2], 8400, tmp_path / "run" / "server", worktree=wt)
    cfg = launch.mstar_config()
    data = yaml.safe_load(cfg.read_text())
    assert cfg.parent == (tmp_path / "run" / "server").resolve()
    assert data == {"model": "qwen3_omni", "max_seq_len": 32768, "model_kwargs": {"moe_backend": "aiter"}}
    assert launch.mstar_base_config() == (wt / "configs" / "qwen3omni_colocated.yaml").resolve()
    cmd = launch.command()
    assert cmd[cmd.index("--config") + 1] == str(cfg) and cmd[cmd.index("serve") + 1] == "qwen3_omni"
    plain = Launch(catalog.ARMS["mstar"], row, [2], 8401, tmp_path / "plain", worktree=wt)
    assert plain.mstar_config() == (wt / "configs" / "qwen3omni_colocated.yaml").resolve()
    # Back to back with mstar on each row, whatever the rotation puts first.
    seen = []
    monkeypatch.setattr(sweep, "run_arm", lambda plan, row, arm, gpus, port, lane: seen.append(arm.name))
    plan = type("P", (), {"arms": list(tables.TABLES["omni"].arms), "trees": {"base": tmp_path}})()
    sweep.run_row(plan, row, 0, [2], 8400)
    assert seen == ["mstar", "mstar-aitermoe", "mstar-base", "vllm-omni", "vllm-omni-tuned"]
    seen.clear()
    sweep.run_row(plan, catalog.ROWS["qwen3omni_2gpu"], 1, [0, 1], 8410)
    assert seen == ["mstar-base", "vllm-omni", "vllm-omni-tuned", "mstar", "mstar-aitermoe"]
    seen.clear()
    sweep.run_row(type("P", (), {"arms": ["mstar-aitermoe", "vllm-omni"], "trees": {}})(), row, 1, [2], 8400)
    assert seen == ["vllm-omni", "mstar-aitermoe"]
    # The table names the config and the override, and the MoE kernels without a tuned-table flag.
    run = tmp_path / "t"
    d = run / "qwen3omni_colocated" / "mstar-aitermoe"
    (d / "server").mkdir(parents=True)
    (run / "sweep.json").write_text(json.dumps({"table": "omni", "run_id": "x", "rows": ["qwen3omni_colocated"],
                                                "arms": ["mstar-aitermoe"], "workloads": ["omni-text-speech"]}))
    (d / "arm.json").write_text(json.dumps({"arm": "mstar-aitermoe", "row": "qwen3omni_colocated",
                                            "status": "done", "startup": {"moe_configs": {"tuned": [],
                                                                                          "default": []}}}))
    (d / "server" / "versions.json").write_text(json.dumps(
        {"git": {"branch": "HEAD", "head": "66b263e08abc"}, "config": str(cfg),
         "config_base": "/wt/configs/qwen3omni_colocated.yaml", "model_kwargs": {"moe_backend": "aiter"}}))
    md = table.write(run).read_text()
    assert "mstar HEAD 66b263e08, configs/qwen3omni_colocated.yaml with model_kwargs moe_backend: aiter" in md
    assert "MoE: AITER fused MoE (mstar's tuned Triton tables don't apply)" in md and "moe-untuned" not in md


def test_hip_devices_map_to_amd_smi_numbers(monkeypatch):
    from benchmark.amd import launch

    # k002-001: HIP counts GPUs in KFD node order, amd-smi by PCI address.
    nodes = {0: 3, 1: 5, 2: 4, 3: 2, 4: 7, 5: 9, 6: 8, 7: 6}  # amd-smi gpu -> KFD node
    small = {"memory_usage": {"vram_mem": {"value": 68000}}}

    def fake(*args):
        if args[0] == "list":
            return [{"gpu": g, "node_id": n} for g, n in nodes.items()]
        if args[0] == "metric":
            return {"gpu_data": [{"gpu": g, "mem_usage": {"used_vram": {"value": 9635 if g == 5 else 284}}}
                                 for g in nodes]}
        procs = lambda g: [{"process_info": {"name": "py", "pid": 1, **small}}] + (  # noqa: E731
            [{"process_info": {"name": "py", "pid": 2, "memory_usage": {"vram_mem": {"value": 9 << 30}}}}]
            if g == 5 else [])
        return {"gpu_data": [{"gpu": g, "process_list": procs(g)} for g in nodes]}

    monkeypatch.setattr(launch, "amd_smi_json", fake)
    launch.smi_index_map.cache_clear()
    try:
        assert launch.smi_index_map() == {0: 3, 1: 0, 2: 2, 3: 1, 4: 7, 5: 4, 6: 6, 7: 5}
        occ = launch.gpu_occupancy([0, 7])
        # HIP 7 is amd-smi 5, the GPU with memory in use; process lists are recorded, memory decides.
        assert occ[0] == {"vram_used_mb": 284.0, "processes": ["py:1"]}
        assert occ[7] == {"vram_used_mb": 9635.0, "processes": ["py:1", "py:2"]}
        assert health.smi_gpus([4, 5]) == [7, 4]
        numa = {"gpu_data": [{"gpu": 7, "numa": {"node": 1, "affinity": 1}}]}  # amd-smi 26
        older = [{"gpu": 3, "numa": {"node": 0}}]
        monkeypatch.setattr(sweep, "amd_smi_json", lambda *a: numa if a[-1] == "7" else older)
        assert sweep._numa_of(4) == 1 and sweep._numa_of(0) == 0
    finally:
        launch.smi_index_map.cache_clear()
    rows = health.parse_rows("GPU3   FAIL  occupancy   procs=[python3.12:1] vram_used=284/262128 MB gfx_activity=2%\n"
                             "GPU5   FAIL  occupancy   procs=[python3.12:2] vram_used=9635/262128 MB gfx_activity=0%\n")
    assert health.interpret_check(rows, [3, 0]).status == "PASS"
    assert health.interpret_check(rows, [3, 5]).status == "FAIL"


def test_allreduce_fallback_marks_the_row(tmp_path):
    from benchmark.amd.launch import allreduce_in_log

    log = tmp_path / "server.log"
    line = ("AITER fused all-reduce + RMSNorm failed its startup check (tp=2 hidden=4096 torch.bfloat16, residual "
            "rounds toward zero (900 of 4096 values), needs an AITER build with OPUS_FP32_to_BF16_DEFAULT=0), "
            "using RCCL + separate norm")
    log.write_text(f"2026-10-19 01:30:00,000 ERROR [worker_0] mstar.distributed.ar_fusion: {line}\n")
    assert allreduce_in_log(log) == {"fused": False, "line": line}
    log.write_text("INFO all-reduce fusion workspace: AITER CustomAllreduce, up to 80 tokens x 4096, max_size 8 MiB\n")
    assert allreduce_in_log(log) == {"fused": True, "backend": "AITER CustomAllreduce, up to 80 tokens x 4096"}
    run = tmp_path / "run"
    d = run / "qwen3_5_9b_tp2" / "mstar"
    (d / "cells/text-chat-256/c1").mkdir(parents=True)
    (run / "sweep.json").write_text(json.dumps({"table": "qwen35-dense", "rows": ["qwen3_5_9b_tp2"], "arms": ["mstar"],
                                                "workloads": ["text-chat-256"]}))
    (d / "arm.json").write_text(json.dumps({"arm": "mstar", "row": "qwen3_5_9b_tp2", "status": "done",
                                            "startup": {"allreduce": {"fused": False, "line": line}}}))
    (d / "cells/text-chat-256/c1/cell.json").write_text(json.dumps(
        {"arm": "mstar", "row": "qwen3_5_9b_tp2", "workload": "text-chat-256", "concurrency": 1, "summary": {},
         "flags": [], "errors": [], "status": "done"}))
    md = table.write(run).read_text()
    assert f"fused all-reduce: RCCL fallback ({line})" in md
    assert md.count("fused all-reduce: RCCL fallback") == 2  # the arm's line and its cell


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


def test_phase_server_runs_mstar_from_the_tree_under_test(tmp_path):
    import shlex

    from benchmark.amd import long_prompt

    wt = tmp_path / "base tree"
    (wt / "configs").mkdir(parents=True)
    (wt / "configs" / "whisper_large.yaml").write_text("model: whisper\n")
    launch = Launch(catalog.ARMS["mstar"], catalog.ROWS["whisper_large"], [0], 8710, tmp_path / "run", worktree=wt)
    cmd = long_prompt.served_command(launch)
    cd, rest = cmd.split(" && ", 1)
    assert shlex.split(cd) == ["cd", str(wt.resolve())]
    assert shlex.split(rest)[1:4] == ["-m", "mstar.cli.main", "serve"]


def test_phase_cells_cuts_one_window_per_concurrency(tmp_path, monkeypatch):
    import contextlib
    import datetime
    import time

    from benchmark.amd import long_prompt, phase_cells

    wt, out = tmp_path / "wt", tmp_path / "out"
    (wt / "configs").mkdir(parents=True)
    (wt / "configs" / "whisper_large.yaml").write_text("model: whisper\n")

    @contextlib.contextmanager
    def fake_server(launch, out_dir, period):
        (out_dir / "server.log").write_text("")
        yield 12.5

    def fake_client(cmd, env, log_path):
        # Phase records inside this run's window, with event_sync growing with the concurrency.
        c = int(cmd[cmd.index("--max-concurrency") + 1])
        t = time.time()
        time.sleep(0.02)
        ts = datetime.datetime.fromtimestamp(t + 0.01).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        with open(out / "server.log", "a") as f:
            for k in range(3):
                f.write(f"{ts} INFO [worker_0] mstar.worker.worker: Worker worker_0 phase-timing iter={k} bs={c}: "
                        f"iter_total: p50=10.0ms p95=10.0ms mean=10.0ms n=20 | "
                        f"worker.postprocess.event_sync: p50={c / 8}ms p95={c / 8}ms mean={c / 8}ms n=20\n")
        rdir = Path(cmd[cmd.index("--output-dir") + 1])
        (rdir / "results.json").write_text(json.dumps({"completed": 5, "failed": 0, "wall_time_s": 2.0}))
        return 0, 0.1

    monkeypatch.setattr(long_prompt, "phase_server", fake_server)
    monkeypatch.setattr(phase_cells.sweep, "run_client", fake_client)
    monkeypatch.setattr(phase_cells, "GAP_S", 0)
    monkeypatch.setattr(phase_cells.sweep, "input_seconds_of", lambda row, wl: [1.0])
    assert phase_cells.main(["--row", "whisper_large", "--workload", "asr-libri", "--concurrency", "8,32",
                             "--gpus", "0", "--out", str(out), "--worktree", str(wt)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert list(summary["windows"]) == ["c8", "c32"] and summary["startup_s"] == 12.5
    assert summary["cells"]["c8"]["phases"]["worker.postprocess.event_sync"]["mean_ms"] == 1.0
    assert summary["cells"]["c32"]["phases"]["worker.postprocess.event_sync"]["mean_ms"] == 4.0
    assert summary["cells"]["c32"]["client"]["completed"] == 5
    assert "### c8" in (out / "phases.txt").read_text()
    with pytest.raises(SystemExit):
        phase_cells.main(["--row", "whisper_large", "--workload", "text-chat-256", "--concurrency", "1",
                          "--gpus", "0", "--out", str(out), "--worktree", str(wt)])


def test_image_to_video_requests_carry_the_first_frame(tmp_path):
    import asyncio

    from benchmark.base import ModelType, RequestType
    from benchmark.dataset import SampleSetDataset
    from benchmark.request import MediaRoutes, RequestMetrics

    assert RequestType.I2V.get_output_modalities() == "video_file"
    assert RequestType.I2V.get_input_modalities() == "image"
    (tmp_path / "f.png").write_bytes(b"\x89PNG frame")
    (tmp_path / "manifest.json").write_text(json.dumps({"items": [
        {"id": "00", "prompt": "a cat", "files": {"image": "f.png"}, "params": {"seed": 0}}]}))
    req = SampleSetDataset(str(tmp_path), 1, RequestType.I2V).get_requests()[0]
    assert req.image_path.endswith("f.png") and req.model_kwargs == {"seed": 0}

    sent = {}

    class Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return sent.pop("reply")

        async def read(self):
            return b"mp4"

    class Session:
        def post(self, url, json=None, data=None, timeout=None):
            sent["url"], sent["json"], sent["data"] = url, json, data
            sent["reply"] = {"data": [{"b64_json": "AAAA"}]} if json is not None else {"id": "v1"}
            return Resp()

        def get(self, url, timeout=None):
            sent["reply"] = {"status": "completed"}
            return Resp()

    model = ModelType.WAN22.inst()
    routes = MediaRoutes()
    metrics = RequestMetrics(request_id=0, type=RequestType.I2V, expected_output_modalities=["video_file"])
    asyncio.run(routes._send_request_video_sync(Session(), req, "http://x", model, metrics, {"num_frames": 33}))
    assert sent["url"].endswith("/v1/videos/generations")
    assert sent["json"]["image"].startswith("data:image/png;base64,") and sent["json"]["num_frames"] == 33
    metrics = RequestMetrics(request_id=0, type=RequestType.I2V, expected_output_modalities=["video_file"])
    asyncio.run(routes._send_request_video_job(Session(), req, "http://x", model, metrics, {}))
    parts = {f[0]["name"]: f for f in sent["data"]._fields}
    assert "input_reference" in parts and parts["input_reference"][0].get("filename") == "f.png"


def test_media_rows_workloads_and_request_settings():
    from benchmark.amd import correctness

    wt = Path(__file__).resolve().parents[2]
    plan = type("P", (), {"correctness_src": wt, "worktree": wt,
                          "workloads": [tables.WORKLOADS[w] for w in tables.TABLES["video"].workloads]})()
    assert [w.name for w in sweep.row_workloads(plan, catalog.ROWS["cosmos3_edge"])] == ["video-t2v"]
    assert [w.name for w in sweep.row_workloads(plan, catalog.ROWS["wan22"])] == ["video-t2v", "video-i2v"]
    assert catalog.ROWS["wan22"].concurrency["video-t2v"] == (1, 2, 4)
    i2v, t2v = tables.WORKLOADS["video-i2v"], tables.WORKLOADS["video-t2v"]
    assert correctness.family_opts(i2v) == {"mode": "i2v"} and correctness.family_opts(t2v) == {}
    # generator_device only to engines whose protocol takes it for the family
    flux = catalog.ROWS["flux2_klein"]
    assert correctness.media_request_kwargs(plan, flux, catalog.ARMS["vllm-omni"])["generator_device"] == "cpu"
    assert "generator_device" not in correctness.media_request_kwargs(plan, flux, catalog.ARMS["mstar"])
    cosmos = catalog.ROWS["cosmos3_nano"]
    sg = correctness.media_request_kwargs(plan, cosmos, catalog.ARMS["sglang-diffusion"], t2v)
    assert sg["generator_device"] == "cuda"
    assert "generator_device" not in correctness.media_request_kwargs(plan, cosmos, catalog.ARMS["vllm-omni"], t2v)
    edge = correctness.media_request_kwargs(plan, catalog.ROWS["cosmos3_edge"], catalog.ARMS["vllm-omni"], t2v)
    assert edge["flow_shift"] == 12.0 and edge["num_frames"] == 33 and edge["num_inference_steps"] == 20
    names = [a.name for a in catalog.arms_for(cosmos, list(tables.TABLES["video"].arms))]
    assert "vllm-omni-rc" in names and "sglang-diffusion-compile" in names and "sglang-diffusion-bcg" not in names
    assert "vllm-omni-rc" not in [a.name for a in catalog.arms_for(catalog.ROWS["wan22"],
                                                                   list(tables.TABLES["video"].arms))]


def test_first_size_requests_one_per_size(tmp_path, monkeypatch):
    calls = []

    def fake_client(cmd, env, log_path, timeout_s=None):
        assert timeout_s == sweep.FIRST_SIZE_TIMEOUT_S
        sdir = Path(cmd[cmd.index("--sample-set-dir") + 1])
        item = json.loads((sdir / "manifest.json").read_text())["items"][0]
        calls.append((item["params"], json.loads(cmd[cmd.index("--request-kwargs") + 1]),
                      cmd[cmd.index("--num-requests") + 1], cmd[cmd.index("--num-warmup") + 1]))
        out = Path(cmd[cmd.index("--output-dir") + 1])
        (out / "results.json").write_text(json.dumps({"per_request": [{"request_id": 0, "jct_ms": 1500.0}]}))
        return 0, 0.1

    monkeypatch.setattr(sweep, "run_client", fake_client)
    plan = type("P", (), {})()
    got = sweep.first_size_requests(plan, catalog.ARMS["vllm-omni"], catalog.ROWS["flux2_klein"],
                                    tables.WORKLOADS["image-gen"], 8400, tmp_path / "a", [3], None,
                                    {"num_inference_steps": 4})
    assert got == {"1024x1024": 1.5, "1344x768": 1.5}
    assert all(n == "1" and w == "0" for _, _, n, w in calls)
    calls.clear()
    got = sweep.first_size_requests(plan, catalog.ARMS["mstar"], catalog.ROWS["wan22"],
                                    tables.WORKLOADS["video-i2v"], 8400, tmp_path / "b", [3], None,
                                    {"height": 480, "width": 832})
    assert got == {"832x480": 1.5} and len(calls) == 1

    def failing_client(cmd, env, log_path, timeout_s=None):
        out = Path(cmd[cmd.index("--output-dir") + 1])
        (out / "results.json").write_text(json.dumps({"per_request": [],
                                                      "errors": [{"request_id": 0, "error": "HTTP 400: no image"}]}))
        return 0, 0.1

    monkeypatch.setattr(sweep, "run_client", failing_client)
    got = sweep.first_size_requests(plan, catalog.ARMS["mstar"], catalog.ROWS["wan22"],
                                    tables.WORKLOADS["video-i2v"], 8400, tmp_path / "c", [3], None,
                                    {"height": 480, "width": 832})
    assert got["832x480"].startswith("failed (HTTP 400: no image, see ")

    def hung_client(cmd, env, log_path, timeout_s=None):
        raise TimeoutError("still waiting")

    monkeypatch.setattr(sweep, "run_client", hung_client)
    with pytest.raises(RuntimeError, match="the first video-i2v request at 832x480 did not finish in 1800 s"):
        sweep.first_size_requests(plan, catalog.ARMS["mstar"], catalog.ROWS["wan22"], tables.WORKLOADS["video-i2v"],
                                  8400, tmp_path / "d", [3], None, {"height": 480, "width": 832})


def test_run_client_kills_a_runner_past_its_timeout(tmp_path):
    import os
    import sys
    import time

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        sweep.run_client([sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ),
                         tmp_path / "log", timeout_s=0.5)
    assert time.monotonic() - t0 < 30
    assert sweep.run_client([sys.executable, "-c", "pass"], dict(os.environ), tmp_path / "log2", timeout_s=30)[0] == 0


def test_a_failed_first_request_leaves_out_that_workloads_cells(tmp_path, monkeypatch):
    from benchmark.amd import correctness

    stopped, cells = [], []

    class FakeLaunch:
        def __init__(self, **kw):
            pass

        def start(self):
            return {"ready_s": 1.0, "first_request_s": 1.0}

        def stop(self):
            stopped.append(1)
            return {}

    class FakeHealth:
        monitor = None

        def __init__(self, *a):
            pass

        def begin(self):
            pass

        def end(self):
            return {}

    first = {"video-t2v": {"832x480": 2.0}, "video-i2v": {"832x480": "failed (HTTP 400: no image, see x)"}}
    monkeypatch.setattr(sweep, "Launch", FakeLaunch)
    monkeypatch.setattr(sweep.health, "ArmHealth", FakeHealth)
    monkeypatch.setattr(sweep.health, "smi_gpus", lambda gpus: gpus)
    monkeypatch.setattr(correctness, "media_request_kwargs", lambda *a: {"height": 480, "width": 832})
    monkeypatch.setattr(sweep, "first_size_requests", lambda plan, arm, row, wl, *a: first[wl.name])
    monkeypatch.setattr(sweep, "run_cell", lambda plan, arm, row, wl, c, *a: cells.append((wl.name, c)) or {
        "summary": {}, "spread": None, "flags": [], "errors": [], "workload": wl.name, "concurrency": c,
        "status": "done"})
    plan = type("P", (), {"run_dir": tmp_path, "resume": False, "measured": [], "numa": False, "log": print,
                          "correctness": False, "worktree": tmp_path, "trees": {}, "skips": [],
                          "fixed_concurrency": True,
                          "workloads": [tables.WORKLOADS[w] for w in ("video-t2v", "video-i2v")]})()
    info = sweep.run_arm(plan, catalog.ROWS["wan22"], catalog.ARMS["mstar"], [5], 8400, "")
    assert info["status"] == "done" and {w for w, _ in cells} == {"video-t2v"} and stopped
    assert info["error"].startswith("first request failed, its cells left out: video-i2v failed (HTTP 400")

    def hung(plan, arm, row, wl, *a):
        raise RuntimeError(f"the first {wl.name} request at 832x480 did not finish in 1800 s")

    monkeypatch.setattr(sweep, "first_size_requests", hung)
    cells.clear(), stopped.clear()
    info = sweep.run_arm(plan, catalog.ROWS["wan22"], catalog.ARMS["mstar"], [5], 8400, "")
    assert info["status"] == "error" and "did not finish in 1800 s" in info["error"] and not cells and stopped


def test_diffusers_reference_runs_each_workload_with_its_family_mode(tmp_path, monkeypatch):
    from benchmark.amd import reference

    seen = []
    monkeypatch.setattr(reference, "_cell", lambda plan, row, arm, wl, arm_dir, cmd, gpus: seen.append(
        (wl.name, json.loads(cmd[cmd.index("--family-opts") + 1]), cmd[cmd.index("--config") + 1])) or {})
    plan = type("P", (), {"workloads": [tables.WORKLOADS[w] for w in ("video-t2v", "video-i2v")], "repeats": 3,
                          "worktree": tmp_path, "correctness_src": tmp_path})()
    reference.run_arm(plan, catalog.ROWS["cosmos3_nano"], catalog.ARMS["ref-diffusers"], [5], tmp_path / "arm")
    assert seen == [("video-t2v", {}, "cosmos3_nano"), ("video-i2v", {"mode": "i2v"}, "cosmos3_nano")]
    assert reference.run_arm(plan, catalog.ROWS["cosmos3_nano"], catalog.ARMS["ref-xdit"], [5], tmp_path / "x")[
        "skipped"]


def test_dry_media_sweep_plan(tmp_path, monkeypatch):
    seen = {}

    def fake_rows(plan, rows, gpus, lanes, port_base):
        seen["plan"], seen["rows"] = plan, [r.name for r in rows]

    monkeypatch.setattr(sweep, "check_node", lambda plan, node_type: {"host": "h", "gpu_names": {}})
    monkeypatch.setattr(sweep, "run_rows", fake_rows)
    monkeypatch.setattr(sweep.health, "snapshot", lambda *a, **k: None)
    import benchmark.amd.table as table_writer
    monkeypatch.setattr(table_writer, "write", lambda run_dir: run_dir / "table.md")
    base = tmp_path / "base"
    base.mkdir()
    rc = sweep.main(["--table", "video", "--node-type", "mi300x", "--gpus", "7", "--run-root", str(tmp_path),
                     "--run-id", "dry", "--dry", "--skip-health", "--rows", "wan22", "--arms",
                     "mstar,mstar-base,sglang-diffusion", "--concurrency", "1", "--repeats", "1", "--requests", "2",
                     "--no-correctness", "--workloads", "video-i2v", "--tree", f"base={base}"])
    plan = seen["plan"]
    assert rc == 0 and seen["rows"] == ["wan22"] and plan.fixed_concurrency and plan.trees == {"base": base}
    assert [(w.name, w.num_requests(1), w.warmup_waves_first) for w in plan.workloads] == [("video-i2v", 2, 0)]
    with pytest.raises(SystemExit):
        sweep.main(["--table", "video", "--node-type", "mi300x", "--gpus", "7", "--run-root", str(tmp_path),
                    "--run-id", "timed", "--requests", "2"])


def test_i2v_rows_send_their_own_first_frames(tmp_path):
    i2v, t2v = tables.WORKLOADS["video-i2v"], tables.WORKLOADS["video-t2v"]
    wan = sweep.workload_set_dir(catalog.ROWS["wan22"], i2v, tmp_path / "wan")
    items = json.loads((wan / "manifest.json").read_text())["items"]
    assert wan == tmp_path / "wan" / "set-video-i2v" and len(items) == 5
    assert all(it["files"]["image"] == f"{catalog.ROWS['wan22'].i2v_frames}/{it['id']}.png" for it in items)
    own = sweep.workload_set_dir(catalog.ROWS["cosmos3_nano"], i2v, tmp_path / "cosmos")
    assert own == sweep.SETS_ROOT / "sets" / "video_v1" and not (tmp_path / "cosmos").exists()
    assert sweep.workload_set_dir(catalog.ROWS["wan22"], t2v, tmp_path / "t") == sweep.SETS_ROOT / "sets" / "video_v1"
    cmd = sweep.runner_command(catalog.ARMS["mstar"], catalog.ROWS["wan22"], i2v, 1, 8400, tmp_path / "r", 0,
                               set_dir=wan)
    assert cmd[cmd.index("--sample-set-dir") + 1] == str(wan)
    assert cmd[cmd.index("--request-type") + 1] == "image_to_video"


def test_skip_patterns_leave_out_arms_and_workloads(monkeypatch):
    skips = ["cosmos3_edge/vllm-omni-rc/*", "*/vllm-omni/video-i2v", "*/sglang-diffusion-compile"]
    plan = type("P", (), {"skips": skips, "arms": list(tables.TABLES["video"].arms), "trees": {},
                          "workloads": [tables.WORKLOADS[w] for w in tables.TABLES["video"].workloads]})()
    edge, wan, nano = catalog.ROWS["cosmos3_edge"], catalog.ROWS["wan22"], catalog.ROWS["cosmos3_nano"]
    rc, vo = catalog.ARMS["vllm-omni-rc"], catalog.ARMS["vllm-omni"]
    i2v, t2v = tables.WORKLOADS["video-i2v"], tables.WORKLOADS["video-t2v"]
    assert sweep.skipped(plan, edge, rc) and not sweep.skipped(plan, nano, rc)
    assert sweep.skipped(plan, wan, vo, i2v) and not sweep.skipped(plan, wan, vo, t2v)
    assert not sweep.skipped(plan, wan, vo)
    seen = []
    monkeypatch.setattr(sweep, "run_arm", lambda plan, row, arm, gpus, port, lane: seen.append((row.name, arm.name)))
    for row in (edge, nano):
        sweep.run_row(plan, row, 0, [5], 8640)
    assert ("cosmos3_edge", "vllm-omni-rc") not in seen and ("cosmos3_nano", "vllm-omni-rc") in seen
    assert not any(a == "sglang-diffusion-compile" for _, a in seen)
    assert ("cosmos3_edge", "vllm-omni") in seen  # its i2v skip doesn't touch Edge, which runs t2v only
    plan.skips = ["*/vllm-omni/video-t2v"]
    seen.clear()
    sweep.run_row(plan, edge, 0, [5], 8640)
    assert ("cosmos3_edge", "vllm-omni") not in seen  # its only workload there is skipped, so no server starts


def test_cosmos3_t2i_row_settings():
    from benchmark.amd import correctness

    wt = Path(__file__).resolve().parents[2]
    plan = type("P", (), {"correctness_src": wt, "worktree": wt})()
    row, wl = catalog.ROWS["cosmos3_nano_t2i"], tables.WORKLOADS["image-gen"]
    assert catalog.mstar_config_name(row, "mstar") == "cosmos3_nano"
    assert correctness.family_opts(wl, row) == {"mode": "t2i", "size": "480x832"}
    ours = correctness.media_request_kwargs(plan, row, catalog.ARMS["mstar"], wl)
    theirs = correctness.media_request_kwargs(plan, row, catalog.ARMS["vllm-omni"], wl)
    assert (ours["height"], ours["width"]) == (theirs["height"], theirs["width"]) == (480, 832)
    assert ours["flow_shift"] == 1.0 and "flow_shift" not in theirs and theirs["generator_device"] == "cuda"
    names = [a.name for a in catalog.arms_for(row, list(tables.TABLES["image"].arms))]
    assert "vllm-omni-rc" in names and "sglang-diffusion-bcg" not in names and "ref-diffusers" in names


def test_video_job_poll_gives_up(monkeypatch):
    import asyncio

    from benchmark.base import ModelType, RequestType
    from benchmark.request import MediaRoutes, RequestInput, RequestMetrics

    class Resp:
        status = 200

        def __init__(self, body):
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return self.body

    class Session:
        def post(self, url, data=None, json=None, timeout=None):
            return Resp({"id": "v1"})

        def get(self, url, timeout=None):
            return Resp({"status": "in_progress"})

    routes = MediaRoutes()
    routes.video_poll_s, routes.video_job_timeout_s = 0.001, 0.01
    req = RequestInput(req_type=RequestType.T2V, prompt="p")
    metrics = RequestMetrics(request_id=0, type=RequestType.T2V, expected_output_modalities=["video_file"])
    asyncio.run(routes._send_request_video_job(Session(), req, "http://x", ModelType.WAN22.inst(), metrics, {}))
    assert metrics.error and "still in_progress" in metrics.error
