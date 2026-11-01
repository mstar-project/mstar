"""``decode_tensor_cores`` on an attention spec reaches every FlashInfer
decode wrapper its manager builds, and defaults to main's tensor-core kernel.

CPU only: the wrapper classes are replaced by recorders, so nothing imports
FlashInfer.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.attn import cross as cross_mod
from mstar.engine.resources.attn import flashinfer as fi_mod
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    CrossAttentionConfig,
    CrossAttentionSpec,
)
from mstar.engine.resources.attn.cross import CrossAttentionManager
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.step import BucketKey, SlotLease


class _Recorder:
    """Stands in for a FlashInfer wrapper class: keeps its kwargs."""

    built: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).built.append(self)


class _Decode(_Recorder):
    built = []


class _Prefill(_Recorder):
    built = []


class _Mixed(_Recorder):
    built = []


@pytest.fixture(autouse=True)
def _recorders(monkeypatch):
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    for cls in (_Decode, _Prefill, _Mixed):
        cls.built = []
    for mod in (fi_mod, cross_mod):
        monkeypatch.setattr(mod, "FlashInferDecodeWrapper", _Decode)
        monkeypatch.setattr(mod, "FlashInferPrefillWrapper", _Prefill)
    monkeypatch.setattr(fi_mod, "FlashInferMixedWrapper", _Mixed)


def _kv():
    return PagedKVConfig(num_layers=1, num_kv_heads=2, num_qo_heads=4, head_dim=64,
                         max_seq_len=256, max_num_pages=8, page_size=16)


def _info(kv):
    return SimpleNamespace(
        dependency=lambda name: SimpleNamespace(config=kv),
        device=torch.device("cpu"), kv_dtype=torch.bfloat16,
        joint_comm_group=None, head_shards=1, max_rows_in_combined={},
    )


@pytest.mark.parametrize("flag", [None, False, True])
def test_self_attention_flag_reaches_its_decode_wrappers(flag):
    spec = AttentionSpec(resource_key="attn", nodes={"n"}, config=AttentionConfig(
        kv_cache="kv", backend=AttnBackend.FLASHINFER,
    ))
    if flag is not None:
        spec.apply_yaml_overrides(decode_tensor_cores=flag)
    expected = True if flag is None else flag  # main's default
    manager = AttentionManager.build(spec, _info(_kv()))
    manager._eager_wrapper("text", is_decode=True, slot=0)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=4, num_tokens=4))
    manager._cg_wrapper(lease, "text", num_rows=4, force_prefill=False)
    assert [w.kwargs["use_tensor_cores"] for w in _Decode.built] == [expected, expected]


def test_a_split_walks_mixed_wrapper_takes_the_flag():
    from mstar.engine.resources.spec import CombinedSplit

    split = CombinedSplit(combined_walk="mixed", prefill_walks=("prefill",))
    manager = fi_mod.FlashInferManager(
        kv_cache="kv", device=torch.device("cpu"), dtype=torch.bfloat16, kv_config=_kv(),
        splits=(split,), decode_tensor_cores=False,
    )
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="mixed", bs=4, num_tokens=16))
    manager._cg_wrapper(lease, "text", num_rows=4, force_prefill=False)
    assert _Mixed.built[0].kwargs["use_tensor_cores"] is False


@pytest.mark.parametrize("flag", [None, False])
def test_cross_attention_flag_reaches_its_decode_wrappers(flag):
    spec = CrossAttentionSpec(resource_key="xattn", nodes={"n"}, config=CrossAttentionConfig(
        kv_cache="ctx", query_kv_cache="kv", backend=AttnBackend.FLASHINFER,
    ))
    if flag is not None:
        spec.apply_yaml_overrides(decode_tensor_cores=flag)
    expected = True if flag is None else flag
    manager = CrossAttentionManager.build(spec, _info(_kv()))
    manager._wrapper_for("text", None, num_rows=2, slot=0, decode=True)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=2, num_tokens=2))
    manager._wrapper_for("text", lease, num_rows=2, slot=0, decode=True)
    assert [w.kwargs["use_tensor_cores"] for w in _Decode.built] == [expected, expected]


@pytest.mark.parametrize("spec_cls,config", [
    (AttentionSpec, AttentionConfig(kv_cache="kv")),
    (CrossAttentionSpec, CrossAttentionConfig(kv_cache="ctx")),
])
def test_a_flag_that_is_not_a_yaml_boolean_is_refused(spec_cls, config):
    spec = spec_cls(resource_key="a", nodes={"n"}, config=config)
    with pytest.raises(ValueError, match="decode_tensor_cores must be true or false"):
        spec.apply_yaml_overrides(decode_tensor_cores="false")



def test_the_default_is_mains_tensor_core_kernel():
    assert AttentionConfig(kv_cache="kv").decode_tensor_cores is True
    assert CrossAttentionConfig(kv_cache="ctx").decode_tensor_cores is True
