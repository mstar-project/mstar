"""Loading fails loudly instead of serving garbage: an fp8 checkpoint whose quant
config went undetected, and a Hub fetch that failed."""
import pytest
import torch

from mstar.model.glm52 import glm52_model
from mstar.model.glm52.weight_loader import _refuse_fp8


def test_fp8_tensors_without_a_quant_config_are_refused():
    ok = [("a.weight", torch.zeros(2, dtype=torch.bfloat16))]
    assert list(_refuse_fp8(iter(ok))) == ok
    with pytest.raises(ValueError, match="quantization_config"):
        list(_refuse_fp8(iter([("b.weight", torch.zeros(2, dtype=torch.float8_e4m3fn))])))
    with pytest.raises(ValueError, match="quantization_config"):
        list(_refuse_fp8(iter([("b.weight_scale_inv", torch.ones(1))])))


def test_a_failed_hub_fetch_raises(monkeypatch, tmp_path):
    import huggingface_hub

    def offline(**kwargs):
        raise OSError("offline, cold cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    # a local checkpoint never reaches the Hub
    assert glm52_model._resolve_local_hf_snapshot(str(tmp_path)) == str(tmp_path)
    # a repo id that cannot be fetched used to come back as if it were a local path
    with pytest.raises(RuntimeError, match="could not fetch"):
        glm52_model._resolve_local_hf_snapshot("zai-org/some-repo")
