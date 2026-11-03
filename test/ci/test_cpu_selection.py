"""Check the CPU test selection without requiring PyTorch or Docker."""

import fnmatch
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / "scripts/ci/select_cpu_tests.py"
SELECTOR_NS = runpy.run_path(str(SELECTOR))
SELECT_TESTS = SELECTOR_NS["select_tests"]
CPU_TEST_PATHS = SELECTOR_NS["CPU_TEST_PATHS"]
NOT_CPU_CORE = SELECTOR_NS["NOT_CPU_CORE"]
EXPANDED_PATHS = SELECTOR_NS["expand_paths"](REPO_ROOT)
IS_COLLECTED = SELECTOR_NS["is_collected"]
# pytest's default python_files
TEST_MODULE_PATTERNS = ("test_*.py", "*_test.py")
# runs in its own CPU Core shard, beside the rest
TP_SIM = "test/modular/test_tp_async_lockstep_sim.py"

# test/cpu-core.txt as it was when CPU Core stopped reading it. Every module
# it ran must stay selected; a module deleted on purpose moves to RETIRED.
FROZEN_CPU_CORE = (
    "mstar/model/cosmos3/tests/test_dit_step_guard.py",
    "mstar/model/cosmos3/tests/test_windowed_decode.py",
    "test/distributed/test_fanout_shapes.py",
    "test/modular/test_accelerator_graph_backend.py",
    "test/modular/test_accelerator_graph_capture.py",
    "test/modular/test_accelerator_graph_runtime.py",
    "test/modular/test_admit_failure_handling.py",
    "test/modular/test_admit_refusal_recovery.py",
    "test/modular/test_api_completion_guard.py",
    "test/modular/test_api_key.py",
    "test/modular/test_api_result_delivery.py",
    "test/modular/test_api_server_helpers.py",
    "test/modular/test_ar_fusion_fallback.py",
    "test/modular/test_attn_backend_auto.py",
    "test/modular/test_bagel_cfg_parallel.py",
    "test/modular/test_bagel_graph_switch.py",
    "test/modular/test_batched_model_output.py",
    "test/modular/test_batched_rows.py",
    "test/modular/test_bench_summary.py",
    "test/modular/test_binary_framing.py",
    "test/modular/test_body_limit.py",
    "test/modular/test_cfg_tp_publish.py",
    "test/modular/test_chatterbox_watermark_import.py",
    "test/modular/test_client_sdk.py",
    "test/modular/test_collect_outputs_fast_path.py",
    "test/modular/test_combined_split.py",
    "test/modular/test_combined_walk_tp_groups.py",
    "test/modular/test_cosmos3_graph_switch.py",
    "test/modular/test_cpu_core_list.py",
    "test/modular/test_data_worker_late_release.py",
    "test/modular/test_data_worker_wakeup.py",
    "test/modular/test_default_socket_prefix.py",
    "test/modular/test_deployment_lock.py",
    "test/modular/test_device_capability.py",
    "test/modular/test_diffusion_denoise_loop.py",
    "test/modular/test_dropped_row_stream_stop.py",
    "test/modular/test_dynamo_compose.py",
    "test/modular/test_e2e_validators.py",
    "test/modular/test_empty_tensor_transport.py",
    "test/modular/test_engine_rows_only_tail.py",
    "test/modular/test_engine_uniform_inputs.py",
    "test/modular/test_eval_framework.py",
    "test/modular/test_fast_tokenize.py",
    "test/modular/test_flashinfer_force_prefill.py",
    "test/modular/test_flex_attention_resource.py",
    "test/modular/test_flux2_klein_compile_exact.py",
    "test/modular/test_flux2_klein_model.py",
    "test/modular/test_gc_freeze.py",
    "test/modular/test_gdn_gate_pad_alignment.py",
    "test/modular/test_gdn_no_sink_slot.py",
    "test/modular/test_gdn_prefill_padding_rows.py",
    "test/modular/test_gdn_tensor_parallel.py",
    "test/modular/test_graph_io.py",
    "test/modular/test_h2d_to_device.py",
    "test/modular/test_image_load_rgb.py",
    "test/modular/test_images_n_cap.py",
    "test/modular/test_images_n_release.py",
    "test/modular/test_ingest_rollback.py",
    "test/modular/test_ingest_unpersist.py",
    "test/modular/test_inline_tokens.py",
    "test/modular/test_kokoro_chunk_loop.py",
    "test/modular/test_kv_decode_plan_cache.py",
    "test/modular/test_kv_multi_label_plans.py",
    "test/modular/test_kv_pinned_index_ring.py",
    "test/modular/test_kv_plan_host_lens.py",
    "test/modular/test_kv_prefix_batch.py",
    "test/modular/test_kv_prefix_stage.py",
    "test/modular/test_kv_publish_retrieve.py",
    "test/modular/test_launch_script_socket_prefix.py",
    "test/modular/test_lazy_per_rid.py",
    "test/modular/test_lazy_publish.py",
    "test/modular/test_ltx2_5_model.py",
    "test/modular/test_mamba2_geometry.py",
    "test/modular/test_mamba2_resource.py",
    "test/modular/test_media_decode_limits.py",
    "test/modular/test_media_paths.py",
    "test/modular/test_micro_scheduler.py",
    "test/modular/test_model_kwargs_tensor_transport.py",
    "test/modular/test_model_registration.py",
    "test/modular/test_mooncake_cuda_hint.py",
    "test/modular/test_mooncake_flat_layout.py",
    "test/modular/test_mooncake_requirement.py",
    "test/modular/test_nccl_loopback.py",
    "test/modular/test_openai_realtime.py",
    "test/modular/test_orpheus_snac_stream.py",
    "test/modular/test_orpheus_weight_tie.py",
    "test/modular/test_packed_token_layout.py",
    "test/modular/test_parallel_groups.py",
    "test/modular/test_per_request_state.py",
    "test/modular/test_pi05_model.py",
    "test/modular/test_platform_dispatch.py",
    "test/modular/test_platforms.py",
    "test/modular/test_preplan_error_paths.py",
    "test/modular/test_profiler_nvtx.py",
    "test/modular/test_psnr_tool.py",
    "test/modular/test_publish_snapshot_batch.py",
    "test/modular/test_qwen3_5_bucket_knobs.py",
    "test/modular/test_qwen3_5_check_stop_batched.py",
    "test/modular/test_qwen3_5_config.py",
    "test/modular/test_qwen3_5_device_loopback.py",
    "test/modular/test_qwen3_5_mixed.py",
    "test/modular/test_qwen3_omni_code2wav_stream.py",
    "test/modular/test_qwen_tts_install.py",
    "test/modular/test_ragged_attention_cpu.py",
    "test/modular/test_recurrent_padding_rows.py",
    "test/modular/test_recurrent_preplan.py",
    "test/modular/test_rendezvous_store.py",
    "test/modular/test_request_failure_propagation.py",
    "test/modular/test_request_id_reuse.py",
    "test/modular/test_request_media_totals.py",
    "test/modular/test_resample_rate_bound.py",
    "test/modular/test_resident_delta.py",
    "test/modular/test_resource_graph_capability.py",
    "test/modular/test_resource_runner.py",
    "test/modular/test_rid_handles.py",
    "test/modular/test_ring_kv_resource.py",
    "test/modular/test_row_output_store.py",
    "test/modular/test_sampler_for_cache.py",
    "test/modular/test_sampler_kept_rids.py",
    "test/modular/test_sampler_last_token.py",
    "test/modular/test_sampler_static_gather_skip.py",
    "test/modular/test_sampler_trash_row.py",
    "test/modular/test_send_prep_entries.py",
    "test/modular/test_shm_tensor_comm.py",
    "test/modular/test_spec_chain_key.py",
    "test/modular/test_speculation_external_inputs.py",
    "test/modular/test_speech_chunking.py",
    "test/modular/test_startup_graph_validation.py",
    "test/modular/test_step_context_rows.py",
    "test/modular/test_step_prep_host_syncs.py",
    "test/modular/test_stream_consumer_final_flag.py",
    "test/modular/test_streams.py",
    "test/modular/test_tensor_uuid.py",
    "test/modular/test_transcription_utf8.py",
    "test/modular/test_transport_microbench.py",
    "test/modular/test_trust_remote_code.py",
    "test/modular/test_tts_speech_wer.py",
    "test/modular/test_video_frame_protocol.py",
    "test/modular/test_wan22_fast_decode.py",
    "test/modular/test_wan_vae_compile_options.py",
    "test/modular/test_wan_vae_decode.py",
    "test/modular/test_waypoint_benchmark_videos.py",
    "test/modular/test_waypoint_checkpoint.py",
    "test/modular/test_waypoint_components.py",
    "test/modular/test_waypoint_dit.py",
    "test/modular/test_waypoint_live_server_benchmark.py",
    "test/modular/test_waypoint_packaging.py",
    "test/modular/test_waypoint_reference_compat.py",
    "test/modular/test_waypoint_shell.py",
    "test/modular/test_waypoint_weight_loader.py",
    "test/modular/test_whisper_kv_sizing.py",
    "test/modular/test_windowing.py",
    "test/modular/test_wire_codec.py",
    "test/modular/test_wire_security.py",
    "test/modular/test_worker_device_loopback.py",
    "test/modular/test_worker_drain.py",
    "test/modular/test_worker_in_flight.py",
    "test/modular/test_worker_message_handling.py",
    "test/modular/test_worker_spec_knobs.py",
    "test/modular/test_worker_speculation_loop_counter.py",
    "test/modular/test_xpu_attention.py",
    "test/modular/test_z_image_model.py",
    "test/sampling_test/test_penalty_gate.py",
)
RETIRED = {
    "test/modular/test_cpu_core_list.py": "checked test/cpu-core.txt and was deleted with it",
}


def manifest_lines(name):
    return [line.strip() for line in (REPO_ROOT / name).read_text().splitlines() if line.strip()]


def repo_test_modules():
    """Tracked and untracked (not ignored) test modules, by pytest's default names."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    return sorted(
        path for path in listed
        if path and (REPO_ROOT / path).is_file()
        and any(fnmatch.fnmatch(Path(path).name, pattern) for pattern in TEST_MODULE_PATTERNS)
    )


class CpuSelectionTests(unittest.TestCase):
    def test_no_hand_kept_cpu_test_list(self):
        self.assertFalse(
            (REPO_ROOT / "test/cpu-core.txt").exists(),
            "test/cpu-core.txt is gone: CPU Core runs every module under CPU_TEST_PATHS "
            "(scripts/ci/select_cpu_tests.py). "
            "Don't list tests; make a module that needs a GPU, weights or an optional "
            "package skip itself (see CONTRIBUTING.md).",
        )

    def test_every_module_the_old_list_ran_is_still_selected(self):
        self.assertEqual(len(FROZEN_CPU_CORE), len(set(FROZEN_CPU_CORE)))
        for path, reason in RETIRED.items():
            self.assertIn(path, FROZEN_CPU_CORE)
            self.assertTrue(reason)
            self.assertFalse((REPO_ROOT / path).exists(), f"{path} exists again; drop it from RETIRED")
        kept = [path for path in FROZEN_CPU_CORE if path not in RETIRED]
        missing = [path for path in kept if not (REPO_ROOT / path).is_file()]
        self.assertEqual(missing, [], "deleted or renamed: add a deleted module to RETIRED with the reason")
        dropped = [path for path in kept if not IS_COLLECTED(path, EXPANDED_PATHS)]
        self.assertEqual(dropped, [], "CPU Core no longer runs these modules; add their directory to CPU_TEST_PATHS")

    def test_every_test_module_is_selected_or_excluded_with_a_reason(self):
        for path, reason in NOT_CPU_CORE.items():
            self.assertTrue((REPO_ROOT / path).exists(), f"NOT_CPU_CORE entry does not exist: {path}")
            self.assertTrue(reason.strip(), path)
            self.assertFalse(IS_COLLECTED(path, EXPANDED_PATHS), f"{path} is both selected and excluded")
        modules = repo_test_modules()
        self.assertGreater(len(modules), 300)
        unaccounted = [
            path for path in modules
            if not IS_COLLECTED(path, EXPANDED_PATHS) and not IS_COLLECTED(path, NOT_CPU_CORE)
        ]
        self.assertEqual(
            unaccounted, [],
            "CPU Core does not run these test modules: put them under a CPU_TEST_PATHS entry, or "
            "list them in NOT_CPU_CORE (scripts/ci/select_cpu_tests.py) with where they run instead",
        )
        unused = [path for path in NOT_CPU_CORE if not any(IS_COLLECTED(m, [path]) for m in modules)]
        self.assertEqual(unused, [], "NOT_CPU_CORE entries that cover no test module")

    def test_selected_modules_outside_packages_have_unique_basenames(self):
        # pytest's default import mode imports these by basename, so two of them stop
        # the whole run at collection with "import file mismatch"
        by_name = {}
        for path in repo_test_modules():
            if IS_COLLECTED(path, EXPANDED_PATHS) and not (REPO_ROOT / path).with_name("__init__.py").exists():
                by_name.setdefault(Path(path).name, []).append(path)
        self.assertEqual({name: paths for name, paths in by_name.items() if len(paths) > 1}, {})

    def test_model_local_test_directories_are_selected(self):
        model_tests = sorted(p.parent.relative_to(REPO_ROOT).as_posix()
                             for p in REPO_ROOT.glob("mstar/model/*/tests/test_*.py"))
        self.assertIn("mstar/model/cosmos3/tests", model_tests)
        for path in model_tests:
            self.assertIn(path, EXPANDED_PATHS)

    def test_a_glob_that_matches_nothing_fails(self):
        with self.assertRaisesRegex(ValueError, "matches nothing: mstar/model/\\*/no_such_dir"):
            SELECTOR_NS["expand_paths"](REPO_ROOT, ("test/modular", "mstar/model/*/no_such_dir"))

    def test_github_job_runs_the_same_paths(self):
        workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text().splitlines()
        commands = [line.strip() for line in workflow if not line.strip().startswith("#")]
        self.assertIn(" ".join(CPU_TEST_PATHS), commands)

    def test_the_tp_simulation_runs_in_its_own_shard(self):
        workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
        self.assertTrue((REPO_ROOT / TP_SIM).is_file())
        self.assertTrue(IS_COLLECTED(TP_SIM, EXPANDED_PATHS))
        self.assertIn(f"--ignore={TP_SIM}", workflow)
        self.assertIn(f"args: -n 3 {TP_SIM}", workflow)

    def test_real_manifest_excludes_only_documented_incompatible_modules(self):
        excluded = manifest_lines("test/cpu-mac-exclude.txt")
        self.assertEqual(set(excluded), {
            "test/modular/test_waypoint_components.py",
            "test/modular/test_waypoint_dit.py",
            "test/modular/test_flex_attention_resource.py",
            "test/modular/test_waypoint_benchmark_videos.py",
        })
        self.assertEqual(
            SELECT_TESTS(EXPANDED_PATHS, excluded),
            EXPANDED_PATHS + [f"--ignore={path}" for path in excluded],
        )
        for path in EXPANDED_PATHS:
            self.assertTrue((REPO_ROOT / path).exists(), path)
        for path in excluded:
            self.assertTrue((REPO_ROOT / path).is_file(), path)

    def test_exclusions_become_ignores_and_listed_files_are_dropped(self):
        self.assertEqual(
            SELECT_TESTS(["tests", "extra.py"], ["tests/arm.py", "extra.py"]),
            ["tests", "--ignore=tests/arm.py"],
        )

    def test_manifest_reader_ignores_blank_lines_and_surrounding_whitespace(self):
        read_manifest = SELECTOR_NS["read_manifest"]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "tests.txt"
            manifest.write_text("\n first.py \n\nlast.py\n \n")
            self.assertEqual(read_manifest(manifest), ["first.py", "last.py"])

    def test_unknown_exclusion_fails(self):
        for excluded in ("unknown.py", "testsuite/a.py"):
            with self.subTest(excluded=excluded):
                with self.assertRaisesRegex(ValueError, f"absent from CPU test paths: {excluded}"):
                    SELECT_TESTS(["tests"], [excluded])

    def test_duplicate_exclusions_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate paths"):
            SELECT_TESTS(["tests"], ["tests/a.py", "tests/a.py"])

    def test_empty_selection_fails(self):
        with self.assertRaisesRegex(ValueError, "selection is empty"):
            SELECT_TESTS(["only.py"], ["only.py"])

    def test_cli_from_other_directory_separates_paths_and_explanation(self):
        with tempfile.TemporaryDirectory() as other_directory:
            result = subprocess.run(
                [sys.executable, str(SELECTOR)], cwd=other_directory,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        excluded = manifest_lines("test/cpu-mac-exclude.txt")
        self.assertEqual(result.stdout.splitlines(), SELECT_TESTS(EXPANDED_PATHS, excluded))
        self.assertIn("requires AVX2", result.stderr)
        self.assertIn(f"Excluded {len(excluded)} modules", result.stderr)
        for path in excluded:
            self.assertIn(path, result.stderr)


if __name__ == "__main__":
    unittest.main()
