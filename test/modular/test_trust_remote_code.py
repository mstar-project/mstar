"""Model loaders never run Python from a checkpoint repo (#509).

Qwen3-Omni, OmniVoice and the Nemotron duplex tokenizer passed
``trust_remote_code=True``. None of their checkpoints needs repo code:
Qwen3-Omni and Nemotron-H are native in transformers, and ``omnivoice`` is
registered with AutoConfig by the installed package. The mocked tests pin the
flag at each call site; the Nemotron ones run the real AutoTokenizer against a
directory whose repo code would leave a marker file if it ran.
"""

import importlib.util
import json
import sys
import types

import pytest

_MARKER = "remote_code_ran"


class _StopLoadError(Exception):
    """Ends a loader right after the call under test."""


def _patch_from_pretrained(monkeypatch, cls, calls, result=None):
    def from_pretrained(path, *args, **kwargs):
        calls.append(kwargs)
        if result is _StopLoadError:
            raise _StopLoadError
        return result if result is not None else types.SimpleNamespace()
    monkeypatch.setattr(cls, "from_pretrained", from_pretrained)


@pytest.fixture
def qwen3_omni(monkeypatch):
    # The talker imports flashinfer at module level; none of it runs here.
    before = set(sys.modules)
    if importlib.util.find_spec("flashinfer") is None:
        monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    from mstar.model.qwen3_omni import qwen3_omni_model
    yield qwen3_omni_model
    for mod in set(sys.modules) - before:
        if mod.startswith("mstar.model.qwen3_omni"):
            del sys.modules[mod]


def test_qwen3_omni_tokenizer_and_processor_without_remote_code(qwen3_omni, monkeypatch, tmp_path):
    from transformers import AutoProcessor, AutoTokenizer

    from mstar.model.qwen3_omni import config as qwen_config

    tok_calls, proc_calls = [], []
    monkeypatch.setattr(qwen3_omni, "_resolve_local_hf_snapshot", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(qwen_config.Qwen3OmniModelConfig, "from_pretrained",
                        classmethod(lambda cls, path: cls()))
    _patch_from_pretrained(monkeypatch, AutoTokenizer, tok_calls)
    _patch_from_pretrained(monkeypatch, AutoProcessor, proc_calls)

    model = qwen3_omni.Qwen3OmniModel("Qwen/Qwen3-Omni-30B-A3B-Instruct")

    assert model._processor is not None
    assert [c["trust_remote_code"] for c in tok_calls + proc_calls] == [False, False]


def test_qwen3_omni_vision_encoder_config_without_remote_code(qwen3_omni, monkeypatch, tmp_path):
    from transformers import AutoConfig

    calls = []
    _patch_from_pretrained(monkeypatch, AutoConfig, calls, result=_StopLoadError)
    model = qwen3_omni.Qwen3OmniModel.__new__(qwen3_omni.Qwen3OmniModel)
    model.local_dir = str(tmp_path)

    with pytest.raises(_StopLoadError):
        model._create_vision_encoder_submodule("cpu")

    assert [c["trust_remote_code"] for c in calls] == [False]


def test_omnivoice_checkpoint_config_without_remote_code(monkeypatch):
    from transformers import AutoConfig

    from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel

    calls, seen = [], []
    config = types.SimpleNamespace()
    _patch_from_pretrained(monkeypatch, AutoConfig, calls, result=config)
    model = OmniVoiceModel("k2-fsa/OmniVoice")
    monkeypatch.setattr(model, "_refresh_checkpoint_defaults", seen.append)

    model._refresh_from_checkpoint_config()

    assert [c["trust_remote_code"] for c in calls] == [False]
    assert seen == [config]


def _nemotron_tokenizer_dir(path, tokenizer_config, extra_files):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (path / "special_tokens_map.json").write_text("{}")
    marker = path / _MARKER
    for name, text in extra_files.items():
        (path / name).write_text(text.replace("MARKER", repr(str(marker))))
    return marker


def _nemotron_model(monkeypatch, tokenizer_dir):
    import huggingface_hub

    from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel

    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, f, **k: str(tokenizer_dir / f.split("/")[-1]))
    return NemotronDuplexModel("nvidia/NVIDIA-NemotronLabs-VoiceChat-11B")


def test_nemotron_tokenizer_ignores_config_repo_code(monkeypatch, tmp_path):
    """The Spark repo's ``nano/config.json`` maps AutoConfig to repo code."""
    marker = _nemotron_tokenizer_dir(
        tmp_path,
        {"tokenizer_class": "PreTrainedTokenizerFast"},
        {
            "config.json": json.dumps({
                "model_type": "nemotron_h",
                "auto_map": {"AutoConfig": "configuration_nemotron_h.NemotronHConfig"},
            }),
            "configuration_nemotron_h.py": (
                "import pathlib\n"
                "pathlib.Path(MARKER).write_text('ran')\n"
                "from transformers import PretrainedConfig\n"
                "class NemotronHConfig(PretrainedConfig):\n"
                "    model_type = 'nemotron_h'\n"
            ),
        },
    )
    model = _nemotron_model(monkeypatch, tmp_path)

    ids = model.tokenizer("hello world")["input_ids"]

    assert ids == [1, 2]
    assert not marker.exists()


def test_nemotron_tokenizer_needing_repo_code_fails_without_running_it(monkeypatch, tmp_path):
    marker = _nemotron_tokenizer_dir(
        tmp_path,
        {
            "tokenizer_class": "RepoTokenizer",
            "auto_map": {"AutoTokenizer": [None, "tokenization_repo.RepoTokenizer"]},
        },
        {
            "tokenization_repo.py": (
                "import pathlib\n"
                "pathlib.Path(MARKER).write_text('ran')\n"
                "from transformers import PreTrainedTokenizerFast\n"
                "class RepoTokenizer(PreTrainedTokenizerFast):\n"
                "    pass\n"
            ),
        },
    )
    model = _nemotron_model(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="custom code"):
        model.tokenizer  # noqa: B018

    assert not marker.exists()
