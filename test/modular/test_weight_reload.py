"""Reloading weights in place: the engine and Orpheus's ``reload_weights``.

A tiny Orpheus language model, built on the CPU from an explicit config and
loaded from one checkpoint, is reloaded from another through the real engine,
model and loader code. A reload must land in the existing parameter storage
(compiled code and captured graphs keep those pointers), route q/k/v and
gate/up into the fused parameters and TP shards as the startup load does, and
refuse a checkpoint that does not match before copying anything.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.engine import Engine
from mstar.engine.resources.base import Resource
from mstar.engine.resources.kv import cpu_page_pool
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.manager import KVManager
from mstar.model.base import Model
from mstar.model.loader import HF_DEFAULT_SKIP_FRAGMENTS, load_weights
from mstar.model.loader.reload import WeightReloadRefusedError, check_reload_compatible, snapshot_checkpoint
from mstar.model.orpheus.components.language_model import OrpheusForCausalLM
from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel, _snapshot_for_reload
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule

ROPE_SCALING = {
    "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
    "original_max_position_embeddings": 64, "rope_type": "llama3",
}
LAYERS, HEADS, KV_HEADS, HIDDEN, HEAD_DIM, INTER, VOCAB = 2, 4, 2, 32, 8, 48, 96


def _config() -> OrpheusModelConfig:
    return OrpheusModelConfig(
        num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
        hidden_size=HIDDEN, head_dim=HEAD_DIM, intermediate_size=INTER,
        max_position_embeddings=128, vocab_size=VOCAB, rope_scaling=dict(ROPE_SCALING),
        pad_token_id=VOCAB - 1, tie_word_embeddings=False,
    )


def _state(seed: int, dtype=torch.float32, tied: bool = False) -> dict[str, torch.Tensor]:
    """An HF Llama state dict for ``_config``, every tensor drawn from ``seed``."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(dtype)

    state = {"model.embed_tokens.weight": rand(VOCAB, HIDDEN), "model.norm.weight": rand(HIDDEN)}
    if not tied:
        state["lm_head.weight"] = rand(VOCAB, HIDDEN)
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        state[p + "self_attn.q_proj.weight"] = rand(HEADS * HEAD_DIM, HIDDEN)
        state[p + "self_attn.k_proj.weight"] = rand(KV_HEADS * HEAD_DIM, HIDDEN)
        state[p + "self_attn.v_proj.weight"] = rand(KV_HEADS * HEAD_DIM, HIDDEN)
        state[p + "self_attn.o_proj.weight"] = rand(HIDDEN, HEADS * HEAD_DIM)
        state[p + "mlp.gate_proj.weight"] = rand(INTER, HIDDEN)
        state[p + "mlp.up_proj.weight"] = rand(INTER, HIDDEN)
        state[p + "mlp.down_proj.weight"] = rand(HIDDEN, INTER)
        state[p + "input_layernorm.weight"] = rand(HIDDEN)
        state[p + "post_attention_layernorm.weight"] = rand(HIDDEN)
    return state


def _checkpoint(root, state, config: dict | None = None, extra_config: dict | None = None) -> str:
    root.mkdir()
    save_file(state, str(root / "model.safetensors"))
    cfg = config if config is not None else {
        "architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS, "torch_dtype": "float32",
    }
    (root / "config.json").write_text(json.dumps({**cfg, **(extra_config or {})}))
    return str(root)


def _lm(path: str, comm_group: CommGroup | None = None, dtype=torch.float32) -> OrpheusForCausalLM:
    """Built as ``_create_llm_submodule`` builds it: on meta, cast, then
    ``to_empty`` and the startup load."""
    with torch.device("meta"):
        lm = OrpheusForCausalLM(_config(), comm_group=comm_group or CommGroup.trivial())
    lm = lm.to(dtype)
    lm.to_empty(device="cpu")
    load_weights(lm, path)
    return lm


def _orpheus(weights_dir: str) -> OrpheusModel:
    """Orpheus as ``_create_llm_submodule`` leaves it, loaded from ``weights_dir``."""
    model = OrpheusModel.__new__(OrpheusModel)
    model.config = _config()
    model._llm_checkpoint = snapshot_checkpoint(weights_dir, HF_DEFAULT_SKIP_FRAGMENTS)
    return model


def _engine(model: Model, lm: nn.Module, kv: KVManager | None = None) -> Engine:
    engine = Engine.__new__(Engine)
    engine._model = model
    engine._device = torch.device("cpu")
    engine._submodules = {
        "LLM": SimpleNamespace(submodule=OrpheusLLMSubmodule(language_model=lm, config=_config())),
    }
    engine._resources = {} if kv is None else {"kv": kv}
    engine._resource_nodes = {} if kv is None else {"kv": frozenset({"LLM"})}
    return engine


def _pointers(module: nn.Module) -> dict[str, int]:
    return {name: p.data_ptr() for name, p in module.named_parameters()}


def _qkv(state, layer: int = 0) -> torch.Tensor:
    p = f"model.layers.{layer}.self_attn."
    return torch.cat([state[p + "q_proj.weight"], state[p + "k_proj.weight"], state[p + "v_proj.weight"]])


def _gate_up(state, layer: int = 0) -> torch.Tensor:
    p = f"model.layers.{layer}.mlp."
    return torch.cat([state[p + "gate_proj.weight"], state[p + "up_proj.weight"]])


@pytest.fixture
def ckpts(tmp_path):
    a, b = _state(seed=0), _state(seed=1)
    return SimpleNamespace(
        a=a, b=b,
        a_dir=_checkpoint(tmp_path / "a", a), b_dir=_checkpoint(tmp_path / "b", b),
        tmp=tmp_path,
    )


# ── the copy lands in place and takes effect ────────────────────────────────


def test_reload_copies_into_the_existing_storage(ckpts):
    lm = _lm(ckpts.a_dir)
    before = _pointers(lm)
    engine = _engine(_orpheus(ckpts.a_dir), lm)

    engine.check_weight_reload(ckpts.b_dir)
    engine.reload_weights(ckpts.b_dir, "v1")

    assert _pointers(lm) == before, "a parameter was given new storage"
    assert torch.equal(lm.model.embed_tokens.weight, ckpts.b["model.embed_tokens.weight"])
    assert torch.equal(lm.lm_head.weight, ckpts.b["lm_head.weight"])
    assert torch.equal(lm.model.norm.weight, ckpts.b["model.norm.weight"])


def test_fused_layers_take_the_new_pieces_and_their_outputs_follow(ckpts):
    lm = _lm(ckpts.a_dir)
    layer = lm.model.layers[1]
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(7))
    out_a = layer.self_attn.qkv_proj(x)
    assert torch.allclose(out_a, x @ _qkv(ckpts.a, 1).T, atol=1e-5)

    _engine(_orpheus(ckpts.a_dir), lm).reload_weights(ckpts.b_dir, "v1")

    assert torch.equal(layer.self_attn.qkv_proj.weight, _qkv(ckpts.b, 1))
    assert torch.equal(layer.mlp.gate_up_proj.weight, _gate_up(ckpts.b, 1))
    out_b = layer.self_attn.qkv_proj(x)
    assert torch.allclose(out_b, x @ _qkv(ckpts.b, 1).T, atol=1e-5)
    assert not torch.allclose(out_a, out_b)
    p = "model.layers.1.mlp."
    gate, up, down = (ckpts.b[p + n + "_proj.weight"] for n in ("gate", "up", "down"))
    expected = (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T
    assert torch.allclose(layer.mlp(x), expected, atol=1e-4)


def test_a_tp_rank_reloads_its_own_shard(ckpts):
    rank1 = CommGroup(my_global_rank=1, my_group_rank=1, group_members=[0, 1])
    lm = _lm(ckpts.a_dir, comm_group=rank1)
    before = _pointers(lm)

    _engine(_orpheus(ckpts.a_dir), lm).reload_weights(ckpts.b_dir, "v1")

    assert _pointers(lm) == before
    b, p = ckpts.b, "model.layers.0."
    q_rows, kv_rows = HEADS * HEAD_DIM // 2, HEAD_DIM  # one of two KV heads per rank
    expected_qkv = torch.cat([
        b[p + "self_attn.q_proj.weight"][q_rows:],
        b[p + "self_attn.k_proj.weight"][kv_rows:],
        b[p + "self_attn.v_proj.weight"][kv_rows:],
    ])
    assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, expected_qkv)
    assert torch.equal(
        lm.model.layers[0].self_attn.o_proj.weight, b[p + "self_attn.o_proj.weight"][:, q_rows:],
    )
    half = INTER // 2
    assert torch.equal(
        lm.model.layers[0].mlp.gate_up_proj.weight,
        torch.cat([b[p + "mlp.gate_proj.weight"][half:], b[p + "mlp.up_proj.weight"][half:]]),
    )
    assert torch.equal(lm.model.embed_tokens.weight, b["model.embed_tokens.weight"][VOCAB // 2:])


@pytest.mark.parametrize("param_dtype, ckpt_dtype", [
    (torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16),
])
def test_the_copy_converts_into_the_storage_dtype(ckpts, param_dtype, ckpt_dtype):
    lm = _lm(ckpts.a_dir, dtype=param_dtype)
    before = _pointers(lm)
    new = _state(seed=2, dtype=ckpt_dtype)
    new_dir = _checkpoint(ckpts.tmp / "converted", new)

    _engine(_orpheus(ckpts.a_dir), lm).reload_weights(new_dir, "v1")

    assert _pointers(lm) == before
    weight = lm.model.layers[0].self_attn.qkv_proj.weight
    assert weight.dtype == param_dtype
    assert torch.equal(weight, _qkv(new).to(param_dtype))


def test_a_tied_checkpoint_refills_the_head_from_the_embedding(tmp_path):
    a, b = _state(seed=0, tied=True), _state(seed=1, tied=True)
    a_dir, b_dir = _checkpoint(tmp_path / "a", a), _checkpoint(tmp_path / "b", b)
    with torch.device("meta"):
        lm = OrpheusForCausalLM(
            OrpheusModelConfig(**{**_config().__dict__, "tie_word_embeddings": True}),
            comm_group=CommGroup.trivial(),
        )
    lm.to_empty(device="cpu")
    load_weights(lm, a_dir)
    head = lm.lm_head.weight.data_ptr()

    _engine(_orpheus(a_dir), lm).reload_weights(b_dir, "v1")

    assert lm.lm_head.weight.data_ptr() == head
    assert torch.equal(lm.lm_head.weight, b["model.embed_tokens.weight"])


# ── refusals copy nothing ───────────────────────────────────────────────────


def _assert_still(lm, state):
    assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, _qkv(state))
    assert torch.equal(lm.model.embed_tokens.weight, state["model.embed_tokens.weight"])


def _refused(ckpts, new_dir, match):
    lm = _lm(ckpts.a_dir)
    with pytest.raises(WeightReloadRefusedError, match=match):
        _engine(_orpheus(ckpts.a_dir), lm).check_weight_reload(new_dir)
    _assert_still(lm, ckpts.a)


def test_a_reshaped_tensor_is_refused(ckpts):
    state = dict(ckpts.b)
    state["model.layers.0.mlp.gate_proj.weight"] = torch.randn(INTER + 2, HIDDEN)
    _refused(ckpts, _checkpoint(ckpts.tmp / "reshaped", state), "shape")


def test_a_missing_or_extra_tensor_is_refused(ckpts):
    missing = {k: v for k, v in ckpts.b.items() if "layers.1.mlp.up_proj" not in k}
    _refused(ckpts, _checkpoint(ckpts.tmp / "missing", missing), "missing")
    extra = {**ckpts.b, "model.layers.0.self_attn.q_norm.weight": torch.randn(HEAD_DIM)}
    _refused(ckpts, _checkpoint(ckpts.tmp / "extra", extra), "does not have")


def test_an_integer_tensor_in_place_of_a_float_one_is_refused(ckpts):
    state = dict(ckpts.b)
    state["model.norm.weight"] = torch.ones(HIDDEN, dtype=torch.int32)
    _refused(ckpts, _checkpoint(ckpts.tmp / "int", state), "dtype")


def test_a_different_architecture_config_is_refused_but_not_the_storage_dtype(ckpts):
    _refused(ckpts, _checkpoint(ckpts.tmp / "deeper", ckpts.b, extra_config={"num_hidden_layers": 3}),
             "num_hidden_layers")
    check_reload_compatible(
        _checkpoint(ckpts.tmp / "bf16cfg", ckpts.b, extra_config={"torch_dtype": "bfloat16"}),
        ckpts.a_dir,
    )


def test_a_checkpoint_without_a_config_is_checked_on_names_and_shapes(ckpts):
    bare = ckpts.tmp / "bare"
    bare.mkdir()
    save_file(ckpts.b, str(bare / "model.safetensors"))
    check_reload_compatible(str(bare), ckpts.a_dir)


def test_runtime_only_config_keys_do_not_block_a_reload(ckpts):
    runtime = {"use_cache": False, "_attn_implementation": "sdpa", "output_hidden_states": True}
    check_reload_compatible(_checkpoint(ckpts.tmp / "runtime", ckpts.b, extra_config=runtime), ckpts.a_dir)
    # and nested, as a multimodal config carries its text model's
    nested = {"text_config": {"hidden_size": HIDDEN, "use_cache": True}}
    ref = _checkpoint(ckpts.tmp / "nested_a", ckpts.a, extra_config=nested)
    nested["text_config"]["use_cache"] = False
    check_reload_compatible(_checkpoint(ckpts.tmp / "nested_b", ckpts.b, extra_config=nested), ref)
    nested["text_config"]["hidden_size"] = HIDDEN * 2
    with pytest.raises(WeightReloadRefusedError, match=r"config.json differs in \['text_config'\]"):
        check_reload_compatible(_checkpoint(ckpts.tmp / "nested_c", ckpts.b, extra_config=nested), ref)


def _with_tokenizer(root: str, files: dict[str, str]) -> str:
    for name, text in files.items():
        (Path(root) / name).write_text(text)
    return root


def test_tokenizer_files_compare_on_content_not_bytes(ckpts):
    vocab = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}, "added_tokens": [], "truncation": None}
    tok_cfg = {"bos_token": "<s>", "model_max_length": 2048, "padding_side": "right"}
    ref = _with_tokenizer(_checkpoint(ckpts.tmp / "tok_a", ckpts.a), {
        "tokenizer.json": json.dumps(vocab, separators=(",", ":")),
        "tokenizer_config.json": json.dumps(tok_cfg),
    })
    # pretty-printed, keys reordered, and only runtime settings changed
    same = _with_tokenizer(_checkpoint(ckpts.tmp / "tok_b", ckpts.b), {
        "tokenizer.json": json.dumps(
            {**dict(reversed(vocab.items())), "truncation": {"max_length": 512}}, indent=2,
        ),
        "tokenizer_config.json": json.dumps({**tok_cfg, "model_max_length": 131072, "padding_side": "left"}),
    })
    check_reload_compatible(same, ref)

    remapped = {**vocab, "model": {"type": "BPE", "vocab": {"a": 1, "b": 0}}}
    other_vocab = _with_tokenizer(_checkpoint(ckpts.tmp / "tok_c", ckpts.b), {
        "tokenizer.json": json.dumps(remapped, indent=2),
    })
    with pytest.raises(WeightReloadRefusedError, match="tokenizer.json differs"):
        check_reload_compatible(other_vocab, ref)
    other_bos = _with_tokenizer(_checkpoint(ckpts.tmp / "tok_d", ckpts.b), {
        "tokenizer_config.json": json.dumps({**tok_cfg, "bos_token": "<bos>"}),
    })
    with pytest.raises(WeightReloadRefusedError, match="tokenizer_config.json differs"):
        check_reload_compatible(other_bos, ref)


def test_truncated_or_garbage_files_are_refused(ckpts):
    data = (ckpts.tmp / "b" / "model.safetensors").read_bytes()
    truncated = ckpts.tmp / "truncated"
    truncated.mkdir()
    (truncated / "model.safetensors").write_bytes(data[:-64])
    _refused(ckpts, str(truncated), "unreadable")
    garbage = ckpts.tmp / "garbage"
    garbage.mkdir()
    (garbage / "model.safetensors").write_bytes(b"\xff" * 4096)
    _refused(ckpts, str(garbage), "unreadable")
    _refused(ckpts, str(ckpts.tmp / "nowhere"), "no checkpoint")


def test_an_index_naming_a_file_outside_the_checkpoint_is_refused(ckpts):
    outside = ckpts.tmp / "outside"
    outside.mkdir()
    (outside / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"model.norm.weight": "../b/model.safetensors"}},
    ))
    _refused(ckpts, str(outside), "outside")


def _hub_snapshot(hub, name: str, state) -> str:
    """``state`` laid out as the Hugging Face cache stores it: two shards and an
    index in ``snapshots/<rev>/``, each a symlink into ``blobs/``."""
    blobs, snap = hub / "blobs", hub / "snapshots" / name
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True)
    keys = sorted(state)
    shards = {"model-00001-of-00002.safetensors": keys[::2], "model-00002-of-00002.safetensors": keys[1::2]}
    files = {}
    for shard, names in shards.items():
        save_file({k: state[k] for k in names}, str(blobs / f"{name}-{shard}"))
        files[shard] = f"{name}-{shard}"
    index = {"weight_map": {k: shard for shard, names in shards.items() for k in names}}
    (blobs / f"{name}-index").write_text(json.dumps(index))
    files["model.safetensors.index.json"] = f"{name}-index"
    (blobs / f"{name}-config").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS, "torch_dtype": "float32",
    }))
    files["config.json"] = f"{name}-config"
    for link, blob in files.items():
        (snap / link).symlink_to(Path("..") / ".." / "blobs" / blob)
    return str(snap)


def test_a_checkpoint_in_the_hub_cache_layout_reloads(ckpts):
    """Every file of a cached snapshot is a symlink out of its directory, into
    ``blobs/``; those are the checkpoint's own files, not ones outside it."""
    hub = ckpts.tmp / "hub"
    a, b = _hub_snapshot(hub, "rev-a", ckpts.a), _hub_snapshot(hub, "rev-b", ckpts.b)
    lm = _lm(a)
    engine = _engine(_orpheus(a), lm)
    engine.reload_weights(b, "v1")
    assert torch.equal(lm.model.norm.weight, ckpts.b["model.norm.weight"])
    assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, _qkv(ckpts.b))


def test_an_index_naming_an_absolute_path_is_refused(ckpts):
    absolute = ckpts.tmp / "absolute"
    absolute.mkdir()
    (absolute / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"model.norm.weight": str(Path(ckpts.b_dir) / "model.safetensors")}},
    ))
    _refused(ckpts, str(absolute), "outside")


# ── the loaded checkpoint is read once, when it is loaded ─────────────────────


def test_a_deleted_startup_directory_does_not_block_a_reload(ckpts):
    start = _checkpoint(ckpts.tmp / "start", ckpts.a)
    lm = _lm(start)
    engine = _engine(_orpheus(start), lm)
    shutil.rmtree(start)  # a trainer cleaned up its old checkpoints

    engine.check_weight_reload(ckpts.b_dir)
    engine.reload_weights(ckpts.b_dir, "v1")
    assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, _qkv(ckpts.b))


def test_a_startup_directory_saved_over_in_place_is_checked_against_what_was_loaded(ckpts):
    """Re-reading the startup directory would compare a checkpoint saved over it
    with itself and pass, then fail partway through the copy."""
    start = _checkpoint(ckpts.tmp / "start", ckpts.a)
    lm = _lm(start)
    engine = _engine(_orpheus(start), lm)

    reshaped = dict(ckpts.b)
    reshaped["model.layers.0.mlp.gate_proj.weight"] = torch.randn(INTER + 2, HIDDEN)
    save_file(reshaped, str(Path(start) / "model.safetensors"))
    with pytest.raises(WeightReloadRefusedError, match="gate_proj.weight: shape"):
        engine.check_weight_reload(start)
    _assert_still(lm, ckpts.a)

    # the next save, with the right shapes, reloads from the same place
    save_file(ckpts.b, str(Path(start) / "model.safetensors"))
    engine.check_weight_reload(start)
    engine.reload_weights(start, "v1")
    assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, _qkv(ckpts.b))


def test_a_startup_checkpoint_that_cannot_be_read_disables_only_reload(ckpts, caplog):
    with caplog.at_level(logging.WARNING):
        reason = _snapshot_for_reload(str(ckpts.tmp / "nowhere"))
    assert isinstance(reason, str) and "weight reload disabled" in caplog.text
    model = _orpheus(ckpts.a_dir)
    model._llm_checkpoint = reason
    lm = _lm(ckpts.a_dir)
    with pytest.raises(WeightReloadRefusedError, match="could not be read at startup"):
        _engine(model, lm).check_weight_reload(ckpts.b_dir)


def test_a_rank_hosting_only_the_snac_decoder_has_nothing_to_reload(ckpts):
    _orpheus(ckpts.a_dir).reload_weights(ckpts.b_dir, {"snac_decoder": nn.Linear(2, 2)})


def test_importing_model_base_does_not_import_safetensors():
    """rust-transport does not install safetensors. Importing the model base
    used to load the loader package, and the rust tests could not collect."""
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'safetensors' or name.startswith('safetensors.'):\n"
        "            raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "from mstar.model.base import Model\n"
        "assert 'safetensors' not in sys.modules\n"
        "print('ok', Model.__name__)\n"
    )
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert got.returncode == 0, got.stderr


def test_a_model_refuses_by_default(ckpts):
    class _Plain:
        reload_weights = Model.reload_weights

    lm = _lm(ckpts.a_dir)
    engine = _engine(_Plain(), lm)
    with pytest.raises(WeightReloadRefusedError, match="_Plain does not support"):
        engine.check_weight_reload(ckpts.b_dir)
    with pytest.raises(WeightReloadRefusedError):
        engine.reload_weights(ckpts.b_dir, "v1")
    _assert_still(lm, ckpts.a)


def test_a_reload_that_moves_storage_is_caught(ckpts):
    """A model that rebinds ``.data`` would leave captured graphs reading the
    old weights; the engine says so instead of serving them."""
    lm = _lm(ckpts.a_dir)

    class _Rebinding(OrpheusModel):
        def reload_weights(self, path, submodules, *, check_only=False):
            weight = submodules["LLM"].language_model.model.norm.weight
            weight.data = weight.data.clone()

    model = _Rebinding.__new__(_Rebinding)
    with pytest.raises(RuntimeError, match="model.norm.weight"):
        _engine(model, lm).reload_weights(ckpts.b_dir, "v1")


# ── weight-derived caches ───────────────────────────────────────────────────


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass


def test_reload_drops_the_prefix_cache_and_reroots_it(ckpts, monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=16, page_size=16,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert kv.enable_prefix_cache(b"root")
    free = kv._arena.num_free
    first, second = kv._arena.acquire(2)
    kv._index.insert(fingerprint(b"root", b"k0"), first)
    kv._index.insert(fingerprint(b"root", b"k1"), second, first)
    kv._arena.release([first, second])  # the request that wrote them finished
    assert kv._arena.num_free == free - 2

    _engine(_orpheus(ckpts.a_dir), _lm(ckpts.a_dir), kv=kv).reload_weights(ckpts.b_dir, "v1")

    assert kv._index.num_pages() == 0 and kv._arena.num_free == free, "cached pages kept"
    assert kv._prefix_root != b"root", "a key from the old weights would still match"
    root_v1 = kv._prefix_root
    kv.drop_prefix_cache("v1")
    assert kv._prefix_root != root_v1, "a second reload under the same name reused the root"


def _kv_with_cached_pages(monkeypatch) -> KVManager:
    """A KV pool with its prefix cache open under ``b"root"`` and two pages
    cached by a request that has finished."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=16, page_size=16,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert kv.enable_prefix_cache(b"root")
    pages = kv._arena.acquire(2)
    kv._index.insert_run([fingerprint(b"root", k) for k in (b"k0", b"k1")], pages, None)
    kv._arena.release(pages)
    assert kv._index.num_pages() == 2
    return kv


class _CopyFails(OrpheusModel):
    """Checks, then fails before copying anything; records what the KV pool
    held when the copy was about to start."""

    def reload_weights(self, path, submodules, *, check_only=False):
        if check_only:
            return super().reload_weights(path, submodules, check_only=True)
        self.seen = (self.kv._index.num_pages(), self.kv._prefix_root)
        raise OSError("read error before the first tensor")


def test_the_prefix_cache_is_dropped_before_the_copy_even_when_the_copy_fails(ckpts, monkeypatch):
    kv = _kv_with_cached_pages(monkeypatch)
    model = _CopyFails.__new__(_CopyFails)
    model.config, model._llm_checkpoint, model.kv = _config(), _orpheus(ckpts.a_dir)._llm_checkpoint, kv
    lm = _lm(ckpts.a_dir)

    with pytest.raises(OSError, match="before the first tensor"):
        _engine(model, lm, kv=kv).reload_weights(ckpts.b_dir, "v1")

    assert model.seen == (0, fingerprint(b"root", "weights", "v1")), "dropped after the copy, or not at all"
    _assert_still(lm, ckpts.a)


class _Recording(Resource):
    """A resource that keeps something derived from the weights."""

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def __init__(self):
        self.dropped: list[str] = []

    def drop_weight_derived_state(self, salt):
        self.dropped.append(salt)
        return 1


class _Stateless(Resource):
    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError


def test_every_resource_hears_of_a_reload_and_the_default_keeps_nothing(ckpts):
    assert _Stateless().drop_weight_derived_state("v1") == 0
    recording = _Recording()
    lm = _lm(ckpts.a_dir)
    engine = _engine(_orpheus(ckpts.a_dir), lm)
    engine._resources = {"stateless": _Stateless(), "recording": recording}

    engine.reload_weights(ckpts.b_dir, "v1")
    engine.reload_weights(ckpts.a_dir, "v2")

    assert recording.dropped == ["v1", "v2"]
    assert torch.equal(lm.model.layers[0].self_attn.qkv_proj.weight, _qkv(ckpts.a))


class _UnpinnedTorch:
    """``torch`` for ``cpu_page_pool`` on a host with no pinned-memory allocator."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def empty(*args, pin_memory=False, **kwargs):
        del pin_memory
        return torch.empty(*args, **kwargs)


def test_reload_frees_the_host_tier_too(ckpts, monkeypatch, caplog):
    """With offload on, the host pool files pages under the same rooted keys.
    Re-rooting without evicting them would leave them resident where no key
    reaches them; the device evictions also go through the host pool's count."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    monkeypatch.setattr(cpu_page_pool, "torch", _UnpinnedTorch())
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=16, page_size=16, cpu_offload_pages=8,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert kv.enable_prefix_cache(b"root")
    keys = [fingerprint(b"root", k) for k in (b"k0", b"k1", b"k2")]
    device_free = kv._arena.num_free
    first, second = kv._arena.acquire(2)
    kv._index.insert_run(keys[:2], [first, second], None)
    kv._arena.release([first, second])
    # the host holds a copy of k0 and the only copy of k1 and k2
    host = kv._cpu_pool
    host_free = host.arena.num_free
    host_pages = host.arena.acquire(3)
    host.index.insert_run(keys, host_pages, None)
    host.arena.release(host_pages)
    assert host.arena.num_free == host_free - 3

    dropped_keys: list[bytes] = []
    count_dropped = host.count_dropped
    monkeypatch.setattr(host, "count_dropped", lambda ks: (dropped_keys.extend(ks), count_dropped(ks)))
    returned: list = []
    drop = kv.drop_prefix_cache
    monkeypatch.setattr(kv, "drop_prefix_cache", lambda salt: returned.append(drop(salt)))

    # caplog's handler fails the test on a record that does not format
    with caplog.at_level(logging.INFO, logger=manager_mod.logger.name):
        _engine(_orpheus(ckpts.a_dir), _lm(ckpts.a_dir), kv=kv).reload_weights(ckpts.b_dir, "v1")

    assert returned == [5] and type(returned[0]) is int
    assert sorted(dropped_keys) == sorted(keys[:2]), "device evictions skipped the host pool's count"
    assert kv._arena.num_free == device_free and kv._index.num_pages() == 0
    assert host.arena.num_free == host_free and host.index.num_pages() == 0, "host pages kept under the old root"
    assert "(2 device and 3 host pages freed)" in caplog.text
