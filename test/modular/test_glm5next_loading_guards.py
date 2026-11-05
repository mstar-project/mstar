"""Loading fails loudly instead of serving garbage: an fp8 checkpoint whose quant
config went undetected, and a Hub fetch that failed."""
import pytest
import torch

from mstar.model.glm5_next import glm5_next_model
from mstar.model.glm5_next.weight_loader import _refuse_fp8


def test_fp8_tensors_without_a_quant_config_are_refused():
    ok = [("a.weight", torch.zeros(2, dtype=torch.bfloat16))]
    assert list(_refuse_fp8(iter(ok))) == ok
    with pytest.raises(ValueError, match="quantization_config"):
        list(_refuse_fp8(iter([("b.weight", torch.zeros(2, dtype=torch.float8_e4m3fn))])))
    with pytest.raises(ValueError, match="quantization_config"):
        list(_refuse_fp8(iter([("b.weight_scale_inv", torch.ones(1))])))


def test_a_failed_hub_fetch_raises(monkeypatch, tmp_path):
    import huggingface_hub

    def offline(*args, **kwargs):
        raise OSError("offline, cold cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    m = object.__new__(glm5_next_model.Glm5NextModel)
    m.cache_dir = None
    # a local checkpoint never reaches the Hub
    m.model_path_hf = str(tmp_path)
    assert m._resolve_checkpoint() == str(tmp_path)
    # a repo id that cannot be fetched raises rather than reading as a local path
    m.model_path_hf = "zai-org/some-repo"
    with pytest.raises(OSError, match="offline"):
        m._resolve_checkpoint()
