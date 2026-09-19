"""KVLayout.MLA: one latent row per token, no K/V or head axis."""

import pytest
import torch

from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVConfig, KVLayout


def _config(**overrides) -> KVConfig:
    kwargs = dict(
        num_layers=2, num_kv_heads=1, head_dim=576, max_seq_len=64,
        max_num_pages=4, page_size=8, num_qo_heads=64, layout=KVLayout.MLA,
    )
    kwargs.update(overrides)
    return KVConfig(**kwargs)


def test_mla_config_rejects_several_kv_heads():
    with pytest.raises(ValueError, match="one latent head"):
        _config(num_kv_heads=4)


def test_mla_shard_keeps_the_latent_and_splits_qo_heads():
    cfg = _config()
    cfg.shard(8)
    assert (cfg.num_kv_heads, cfg.num_qo_heads) == (1, 8)
    cfg.shard(8)  # idempotent, like NHD
    assert (cfg.num_kv_heads, cfg.num_qo_heads) == (1, 8)


def test_mla_cache_shape_and_layer_view():
    cache = KVCache(_config(), torch.device("cpu"))
    assert cache.tensor.shape == (2, 4, 8, 576)
    assert cache.layer_view(1).shape == (4, 8, 576)
    assert cache.layer_view(1).data_ptr() == cache.tensor[1].data_ptr()


def test_mla_write_read_tokens_roundtrip_and_v_rejected():
    cache = KVCache(_config(), torch.device("cpu"))
    latent = torch.arange(3 * 576, dtype=torch.float32).reshape(3, 576).bfloat16()
    page_idx = torch.tensor([1, 1, 2])
    cache_idx = torch.tensor([0, 1, 0])
    cache.write_tokens(1, latent, None, page_idx, cache_idx)
    assert torch.equal(cache.read_tokens(1, page_idx, cache_idx), latent)
    assert torch.equal(cache.tensor[1, 1, 0], latent[0])
    assert torch.equal(cache.tensor[1, 2, 0], latent[2])
    # the other layer is untouched
    assert not cache.tensor[0].any()
    with pytest.raises(ValueError, match="v must be None"):
        cache.write_tokens(1, latent, latent, page_idx, cache_idx)


def test_mla_chunk_ptrs_one_contiguous_chunk():
    cache = KVCache(_config(), torch.device("cpu"))
    ptrs, nbytes = cache.chunk_ptrs(layer_idx=1, page_idx=2, token_start=3, token_end=6)
    assert len(ptrs) == 1
    assert nbytes == 3 * 576 * cache.tensor.element_size()
    assert ptrs[0] == cache.tensor[1, 2, 3].data_ptr()
    # a remote base pointer is offset the same way
    remote, _ = cache.chunk_ptrs(1, 2, 3, 6, base_ptr=1000)
    assert remote[0] - 1000 == ptrs[0] - cache.data_ptr()
    assert cache.chunk_view(1, 2, 3, 6).shape == (3, 576)


def test_mla_copy_pages_moves_whole_pages_on_every_layer():
    cache = KVCache(_config(), torch.device("cpu"))
    cache.tensor[:, 1] = 1.0
    cache.tensor[:, 2] = 2.0
    cache.copy_pages([1, 2], [3, 0])
    assert torch.equal(cache.tensor[:, 3], cache.tensor[:, 1])
    assert torch.equal(cache.tensor[:, 0], cache.tensor[:, 2])


def test_nhd_unchanged():
    cache = KVCache(
        _config(layout=KVLayout.NHD, num_kv_heads=2, head_dim=16, num_qo_heads=2),
        torch.device("cpu"),
    )
    assert cache.tensor.shape == (2, 4, 2, 8, 2, 16)
    ptrs, _ = cache.chunk_ptrs(0, 1, 0, 2)
    assert len(ptrs) == 2
