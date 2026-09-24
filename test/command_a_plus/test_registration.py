"""Offline deployment and metadata-only resolution checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mstar.cli.main import _resolve_config
from mstar.model.command_a_plus.assets import CHECKPOINT_REVISION, METADATA_FILES, MODEL_ID, resolve_metadata
from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.registry import HF_MODELS, get_model_class
from test.command_a_plus.test_integration import FIXTURE


class RegistrationTests(unittest.TestCase):
    def test_registry_cli_and_tp8_deployment_agree(self):
        self.assertIs(get_model_class("command_a_plus"), CommandAPlusModel)
        self.assertEqual(HF_MODELS["command_a_plus"]["model_path_hf"], MODEL_ID)
        config = yaml.safe_load(Path(_resolve_config("command_a_plus", None)).read_text())
        self.assertEqual(config["model"], "command_a_plus")
        directory = self.enterContext(tempfile.TemporaryDirectory())
        (Path(directory) / "config.json").write_text(FIXTURE.read_text())
        model = CommandAPlusModel(MODEL_ID, checkpoint_dir=directory)
        graphs = model.get_worker_graphs(_resolve_config("command_a_plus", None))
        self.assertEqual(len(graphs), 2)
        self.assertTrue(all(g.tp_size == 8 and g.sp_size == 1 for g in graphs))
        # Fail before the parameter constructor allocates any storage.
        with patch("mstar.model.command_a_plus.components.language_model.CommandAPlusForCausalLM") as constructor:
            with self.assertRaisesRegex(FileNotFoundError, "obtain the checkpoint explicitly"):
                model.get_submodule("LLM")
            constructor.assert_not_called()

    def test_metadata_download_is_pinned_and_excludes_weights(self):
        with patch("huggingface_hub.snapshot_download", return_value="/tmp/metadata") as download:
            self.assertEqual(resolve_metadata(MODEL_ID, "/tmp/cache"), Path("/tmp/metadata"))
            download.assert_called_once_with(repo_id=MODEL_ID, revision=CHECKPOINT_REVISION,
                                             cache_dir="/tmp/cache", allow_patterns=list(METADATA_FILES))
        self.assertFalse(any(name.endswith(".safetensors") or "*" in name for name in METADATA_FILES))


if __name__ == "__main__":
    unittest.main()
