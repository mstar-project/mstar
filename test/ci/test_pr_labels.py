"""Check the PR labeler's rules (.github/labels/label_prs.py) against the repo."""

import contextlib
import io
import json
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELER = runpy.run_path(str(REPO_ROOT / ".github/labels/label_prs.py"))
areas_of = LABELER["areas_of"]
labels_for = LABELER["labels_for"]
ALIASES = LABELER["model_aliases"](str(REPO_ROOT), [])


def changed(path, lines):
    return {"filename": path, "additions": lines, "deletions": 0}


def pull(number, body="", sha=None, state="open", labels=()):
    return {"number": number, "body": body, "state": state, "head": {"sha": sha or f"sha{number}"},
            "labels": [{"name": lb} for lb in labels]}


class FakeGh:
    """Answers `gh api <path>` from a dict of path to JSON; a missing path fails like gh does."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, *args, data=None):
        path = next(a for a in args if a.startswith("repos/"))
        self.calls.append(path)
        if path not in self.responses:
            raise RuntimeError(f"gh api {path}: HTTP 404")
        out = self.responses[path]
        # gh_pages asks for --slurp: a list of pages.
        return json.dumps([out] if "--slurp" in args else out)


def with_gh(fake):
    return mock.patch.dict(LABELER["main"].__globals__, {"gh": fake})


class TestPrLabels(unittest.TestCase):
    def test_configs_find_their_model(self):
        for path, label in [
            ("configs/qwen3_5_9b_tp2.yaml", "model/qwen3_5"),
            ("configs/qwen3_8_27b_tp4.yaml", "model/qwen3_5"),  # by registry key
            ("configs/qwen3omni_2gpu.yaml", "model/qwen3_omni"),
            ("configs/qwen3tts_1p7b.yaml", "model/qwen3_tts"),
            ("configs/cosmos3_nano_tp2.yaml", "model/cosmos3"),
        ]:
            self.assertEqual(areas_of(path, ALIASES), (label,), path)

    def test_rust_mirrors_get_both_areas(self):
        self.assertEqual(set(areas_of("rust/src/graph/runtime.rs", ALIASES)), {"area/rust", "area/graph"})
        self.assertEqual(set(areas_of("rust/src/shm.rs", ALIASES)), {"area/rust", "area/transport"})
        self.assertEqual(set(areas_of("rust/server/src/main.rs", ALIASES)), {"area/rust", "area/api"})

    def test_unknown_path_is_other(self):
        self.assertEqual(areas_of("mstar/brand_new_package/x.py", ALIASES), ("area/other",))

    def test_small_layer_touch_is_dropped_but_models_are_not(self):
        labels = labels_for([changed("mstar/engine/resources/kv/manager.py", 5),
                             changed("mstar/model/orpheus/submodules.py", 2),
                             changed("mstar/api_server/app.py", 40)], ALIASES)
        self.assertEqual(labels, {"area/api", "model/orpheus", "size/S"})

    def test_any_change_to_a_core_file_labels_its_area(self):
        labels = labels_for([changed("mstar/worker/worker.py", 6),
                             changed("mstar/engine/fp32_precision.py", 125)], ALIASES)
        self.assertEqual(labels, {"area/worker", "area/engine", "size/M"})

    def test_many_models_collapse(self):
        labels = labels_for([changed(f"mstar/model/{m}/x.py", 1)
                             for m in ("bagel", "orpheus", "qwen3_5", "whisper")], ALIASES)
        self.assertEqual(labels, {"model/many", "size/XS"})

    def test_docs_and_tests_label_only_alone(self):
        self.assertEqual(labels_for([changed("docs/models.rst", 30)], ALIASES),
                         {"area/docs", "size/S"})
        self.assertEqual(labels_for([changed("docs/models.rst", 30), changed("test/modular/test_x.py", 500),
                                     changed("mstar/api_server/app.py", 12)], ALIASES),
                         {"area/api", "size/XS"})

    def test_tests_inside_a_model_folder_are_tests(self):
        path = "mstar/model/cosmos3/tests/test_dit_step_guard.py"
        self.assertEqual(areas_of(path, ALIASES), ("area/tests",))
        # Neither a model label nor size: an API change with a 2-line fixture edit.
        labels = labels_for([changed("mstar/api_server/app.py", 15), changed(path, 10)], ALIASES)
        self.assertEqual(labels, {"area/api", "size/XS"})
        self.assertEqual(areas_of("mstar/model/cosmos3/config.py", ALIASES), ("model/cosmos3",))

    def test_utils_name_the_layer_that_uses_them(self):
        for path, labels in [
            ("mstar/utils/ipc_format.py", {"area/worker"}),
            ("mstar/utils/socket_prefix.py", {"area/api"}),
            ("mstar/utils/runtime_dir.py", {"area/api"}),
            ("mstar/utils/utf8_stream.py", {"area/api"}),
            ("mstar/utils/client_errors.py", {"area/api"}),
            ("mstar/utils/profiler.py", {"area/observability"}),
            ("mstar/utils/attention.py", {"area/model-shared"}),
            ("mstar/utils/fused_moe/runner.py", {"area/model-shared"}),
            # Shared by several layers.
            ("mstar/utils/knobs.py", {"area/utils"}),
            ("mstar/utils/orphan.py", {"area/utils"}),
        ]:
            self.assertEqual(set(areas_of(path, ALIASES)), labels, path)

    def test_validation_and_worker_metrics(self):
        self.assertEqual(areas_of("mstar/validation/placement.py", ALIASES), ("area/conductor",))
        self.assertEqual(set(areas_of("mstar/worker/worker_metrics.py", ALIASES)),
                         {"area/observability", "area/worker"})

    def test_rust_server_lockfile_is_infra_and_not_sized(self):
        self.assertEqual(areas_of("rust/server/Cargo.lock", ALIASES), ("area/infra",))
        labels = labels_for([changed("rust/server/src/main.rs", 15),
                             changed("rust/server/Cargo.lock", 19)], ALIASES)
        self.assertEqual(labels, {"area/rust", "area/api", "area/infra", "size/XS"})


class TestStackedPrs(unittest.TestCase):
    def test_stacked_on_reads_the_opening_lines(self):
        stacked_on = LABELER["stacked_on"]
        for body, parent in [
            ("Stacked on #496; review the top commit(s) only.\n\n## What does this PR do?", 496),
            ("Fixes #597\nStacked on #496; review the top commit(s) only.\n", 496),
            ("**Stacked on #465.** The fork can't target #465's branch.", 465),
            ("Full duplex (#186), part 4 of 6; stacked on #593. This PR ...", 593),
            ("Fixes #418\n\n## What does this PR do?\n\nStacked on #12 in an earlier version.", None),
            ("Fixes #1\n\n<details><summary>Depends on</summary>\nStacked on #419\n</details>", None),
            ("Depends on #419.", None),
            ("#629 is stacked on this PR.", None),
            ("Stacked on #700.", None),  # its own number
            ("", None),
            (None, None),
        ]:
            self.assertEqual(stacked_on(body, 700), parent, body)

    def compare_path(self, parent_sha="sha496", child_sha="sha659"):
        return f"repos/o/r/compare/{parent_sha}...{child_sha}?per_page=1"

    def stacked(self, files, responses, open_prs=None, body="Stacked on #496; review the top commit(s) only."):
        fake = FakeGh(responses)
        with with_gh(fake), contextlib.redirect_stderr(io.StringIO()):
            got = LABELER["stacked_files"]("o/r", pull(659, body), files, open_prs)
        return got, fake.calls

    def test_takes_the_files_from_the_parents_head(self):
        files = [changed("pyproject.toml", 30), changed("mstar/api_server/app.py", 400),
                 changed("mstar/model/voxcpm2/model.py", 200)]
        own = [changed("mstar/model/voxcpm2/model.py", 120)]
        got, calls = self.stacked(files, {"repos/o/r/pulls/496": pull(496),
                                          self.compare_path(): {"files": own}})
        self.assertEqual(got, own)
        self.assertEqual(calls, ["repos/o/r/pulls/496", self.compare_path()])

    def test_all_mode_uses_the_open_pr_list(self):
        own = [changed("a.py", 1)]
        got, calls = self.stacked([changed("a.py", 9)], {self.compare_path("p"): {"files": own}},
                                  open_prs={496: pull(496, sha="p")})
        self.assertEqual((got, calls), (own, [self.compare_path("p")]))
        # A parent missing from the open list is closed or merged: no call at all.
        got, calls = self.stacked([changed("a.py", 9)], {}, open_prs={})
        self.assertEqual((got, calls), (None, []))

    def test_falls_back_to_the_pr_file_list(self):
        files = [changed(f"f{i}.py", 1) for i in range(300)]
        for name, responses in [
            ("parent merged", {"repos/o/r/pulls/496": pull(496, state="closed")}),
            ("parent lookup fails", {}),
            ("compare fails", {"repos/o/r/pulls/496": pull(496)}),
            ("compare lists no files", {"repos/o/r/pulls/496": pull(496), self.compare_path(): {"files": []}}),
            ("compare may be cut off", {"repos/o/r/pulls/496": pull(496), self.compare_path(): {"files": files}}),
            ("compare has a file the PR doesn't", {"repos/o/r/pulls/496": pull(496),
                                                   self.compare_path(): {"files": [changed("other.py", 1)]}}),
        ]:
            got, _ = self.stacked(files, responses)
            self.assertIsNone(got, name)
        got, calls = self.stacked(files, {}, body="Fixes #597")
        self.assertEqual((got, calls), (None, []))

    def test_main_sizes_a_stacked_pr_against_its_parent(self):
        files = [changed("pyproject.toml", 30), changed("mstar/api_server/app.py", 2000),
                 changed("mstar/model/voxcpm2/model.py", 200)]
        fake = FakeGh({
            "repos/o/r/pulls/659": pull(659, "Stacked on #496; review the top commit(s) only.",
                                        labels=["size/XL", "type/perf"]),
            "repos/o/r/pulls/659/files?per_page=100": files,
            "repos/o/r/pulls/496": pull(496),
            self.compare_path(): {"files": [changed("mstar/model/voxcpm2/model.py", 200)]},
        })
        out = io.StringIO()
        argv = ["label_prs.py", "--repo", "o/r", "659", "--dry-run", "--root", str(REPO_ROOT)]
        with with_gh(fake), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as exit_:
                LABELER["main"]()
        self.assertEqual(exit_.exception.code, 0)
        self.assertEqual(out.getvalue().split("  ")[0], "#659: model/voxcpm2 size/M")


if __name__ == "__main__":
    unittest.main()
