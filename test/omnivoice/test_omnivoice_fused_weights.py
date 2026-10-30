"""The boot check for what ``OmniVoiceBackbone`` reads off the patched model:
its embedding merge and audio head, and the fused weights ``apply_flashinfer``
leaves on each layer; and for the packed runner the loader drops."""

from types import SimpleNamespace

import pytest

from mstar.model.omnivoice.components.backbone import assert_flashinfer_patched


def _layer(**drop):
    attn = {"_fi_w_qkv": 0, "_fi_qkv_split": (1, 1, 1), "_fi_rope_theta": 1e6}
    mlp = {"_fi_w_gate_up": 0}
    for name in drop:
        attn.pop(name, None)
        mlp.pop(name, None)
    return SimpleNamespace(self_attn=SimpleNamespace(**attn), mlp=SimpleNamespace(**mlp))


def _model(*layers):
    return SimpleNamespace(
        llm=SimpleNamespace(layers=list(layers)), _prepare_embed_inputs=None, audio_heads=None, _fi_runner=None,
    )


def test_a_patched_model_passes():
    assert_flashinfer_patched(_model(_layer(), _layer()))


@pytest.mark.parametrize("name", ["_fi_w_qkv", "_fi_qkv_split", "_fi_rope_theta", "_fi_w_gate_up"])
def test_a_missing_fused_weight_fails_at_boot(name):
    with pytest.raises(SystemExit, match=name):
        assert_flashinfer_patched(_model(_layer(), _layer(**{name: True})))


def test_a_model_without_the_packed_runner_fails_at_boot():
    model = _model(_layer())
    del model._fi_runner
    with pytest.raises(SystemExit, match="_fi_runner"):
        assert_flashinfer_patched(model)
