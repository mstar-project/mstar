"""The boot check for the fused weights ``apply_flashinfer`` leaves on each
layer, which ``OmniVoiceBackbone.encode`` reads."""

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
    return SimpleNamespace(llm=SimpleNamespace(layers=list(layers)))


def test_a_patched_model_passes():
    assert_flashinfer_patched(_model(_layer(), _layer()))


@pytest.mark.parametrize("name", ["_fi_w_qkv", "_fi_qkv_split", "_fi_rope_theta", "_fi_w_gate_up"])
def test_a_missing_fused_weight_fails_at_boot(name):
    with pytest.raises(RuntimeError, match=name):
        assert_flashinfer_patched(_model(_layer(), _layer(**{name: True})))
