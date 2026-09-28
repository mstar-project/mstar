"""Tests for Kimi-K2.7-Code's vision tower, preprocessing, and the
image-aware prefill schedule / process_prompt wiring."""

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import Sequential, TensorPointerInfo
from mstar.model.kimi_k2_7.components.vision import (
    KimiMMProjector,
    KimiVisionTower,
    _resized_and_padded_dims,
    num_image_tokens,
    preprocess_image,
)
from mstar.model.kimi_k2_7.config import KimiK2Config, KimiVisionConfig
from mstar.model.kimi_k2_7.kimi_model import VISION_NODE, KimiK2Model
from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule, KimiVisionEncoderSubmodule
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine


def _make_model() -> KimiK2Model:
    model = object.__new__(KimiK2Model)
    model.config = KimiK2Config.reduced()
    model.config.vision = KimiVisionConfig()
    model._submodule_cache = {}
    return model


def _tensor_info(name: str) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[1], dtype="torch.float32", nbytes=4, address=0, stride=[1],
        uuid=f"uuid-{name}", source_session_id="test:0", source_entity="test",
    )


def _reduced_vision_config() -> KimiVisionConfig:
    return KimiVisionConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, patch_size=2, pos_emb_height=4, pos_emb_width=4,
        merge_kernel_size=2, text_hidden_size=6,
    )


# --- preprocessing formula ---------------------------------------------------


def test_resized_and_padded_dims_never_upscales():
    cfg = KimiVisionConfig()
    new_w, new_h, _, _ = _resized_and_padded_dims(4, 4, cfg)
    assert (new_w, new_h) == (4, 4)


def test_resized_and_padded_dims_pads_to_merge_patch_multiple():
    cfg = KimiVisionConfig()
    new_w, new_h, pad_w, pad_h = _resized_and_padded_dims(4, 4, cfg)
    factor = cfg.merge_kernel_size * cfg.patch_size
    assert (new_w + pad_w) % factor == 0
    assert (new_h + pad_h) % factor == 0


def test_resized_and_padded_dims_caps_at_patch_limit_on_one_side():
    cfg = KimiVisionConfig(patch_limit_on_one_side=4, in_patch_limit=10**9)
    new_w, _, _, _ = _resized_and_padded_dims(cfg.patch_size * 100, cfg.patch_size * 4, cfg)
    assert new_w == cfg.patch_limit_on_one_side * cfg.patch_size


def test_num_image_tokens_hand_computed_example():
    # patch=14, merge=2: 40x20 resizes to 40x20 (under every cap), pads to
    # 56x28, i.e. a 2x1 grid of merge windows.
    assert num_image_tokens(40, 20, KimiVisionConfig()) == 2


def test_preprocess_image_padded_region_normalizes_to_minus_one():
    cfg = KimiVisionConfig()
    patches, gh, gw = preprocess_image(torch.rand(3, 4, 4), cfg)
    assert patches.shape == (gh * gw, 3, cfg.patch_size, cfg.patch_size)
    # a 4x4 image pads out to a much larger canvas; the last patch's last
    # pixel falls in the zero-padded region, which mean=std=0.5 sends to -1.
    assert torch.allclose(patches[-1, :, -1, -1], torch.tensor(-1.0))


# --- tower + projector shapes -------------------------------------------------


def test_tower_and_projector_shapes_match_num_image_tokens():
    cfg = _reduced_vision_config()
    gh, gw = 4, 4
    patches = torch.randn(gh * gw, 3, cfg.patch_size, cfg.patch_size)

    merged = KimiVisionTower(cfg)(patches, gh, gw)
    out = KimiMMProjector(cfg)(merged)

    expected_tokens = (gh // cfg.merge_kernel_size) * (gw // cfg.merge_kernel_size)
    assert merged.shape == (expected_tokens, cfg.merge_kernel_size**2, cfg.hidden_size)
    assert out.shape == (expected_tokens, cfg.text_hidden_size)


def test_vision_encoder_submodule_prepare_and_forward():
    cfg = _reduced_vision_config()
    submodule = KimiVisionEncoderSubmodule(
        vision_tower=KimiVisionTower(cfg), mm_projector=KimiMMProjector(cfg),
    )
    gh, gw = 4, 4
    patches = torch.randn(gh * gw, 3, cfg.patch_size, cfg.patch_size)

    node_inputs = submodule.prepare_inputs(
        graph_walk="prefill_vision", fwd_info=None,
        inputs={"image_inputs": [patches], "image_grids": [torch.tensor([gh, gw])]},
    )
    assert node_inputs.kwargs == {"gh": gh, "gw": gw}
    assert torch.equal(node_inputs.tensor_inputs["patches"], patches)

    out = submodule.forward(
        graph_walk="prefill_vision", engine_inputs=None, patches=patches, gh=gh, gw=gw,
    )
    expected_tokens = (gh // cfg.merge_kernel_size) * (gw // cfg.merge_kernel_size)
    assert out["image_embeds"][0].shape == (expected_tokens, cfg.text_hidden_size)


# --- prefill schedule ---------------------------------------------------------


def test_prefill_schedule_text_only():
    model = _make_model()
    t0 = _tensor_info("t0")
    assert model._prefill_schedule_from_signals({"text_inputs": [t0]}) == [
        ("prefill", {"text_inputs": [t0]}),
    ]


def test_prefill_schedule_text_only_tolerates_missing_text_inputs():
    # Dummy/fallback signals can omit ``text_inputs`` entirely; the no-image
    # branch must not index into an empty list.
    model = _make_model()
    assert model._prefill_schedule_from_signals({}) == [
        ("prefill", {"text_inputs": []}),
    ]


def test_prefill_schedule_interleaves_text_and_images():
    model = _make_model()
    t0, t1, t2 = _tensor_info("t0"), _tensor_info("t1"), _tensor_info("t2")
    i0, i1 = _tensor_info("i0"), _tensor_info("i1")
    g0, g1 = _tensor_info("g0"), _tensor_info("g1")

    schedule = model._prefill_schedule_from_signals({
        "text_inputs": [t0, t1, t2], "image_inputs": [i0, i1], "image_grids": [g0, g1],
    })

    assert schedule == [
        ("prefill", {"text_inputs": t0}),
        ("prefill_vision", {"image_inputs": i0, "image_grids": g0}),
        ("prefill", {"text_inputs": t1}),
        ("prefill_vision", {"image_inputs": i1, "image_grids": g1}),
        ("prefill", {"text_inputs": t2}),
    ]


def test_prefill_step_inputs_targets_vision_node_for_image_steps():
    model = _make_model()
    i0, g0 = _tensor_info("i0"), _tensor_info("g0")
    edges = model._prefill_step_inputs(
        ("prefill_vision", {"image_inputs": i0, "image_grids": g0})
    )
    assert {e.next_node for e in edges} == {VISION_NODE}
    assert {e.name for e in edges} == {"image_inputs", "image_grids"}


def test_initial_and_partition_args_drive_full_image_schedule():
    model = _make_model()
    t0, t1 = _tensor_info("t0"), _tensor_info("t1")
    i0, g0 = _tensor_info("i0"), _tensor_info("g0")
    input_signals = {"text_inputs": [t0, t1], "image_inputs": [i0], "image_grids": [g0]}

    initial = model.get_initial_forward_pass_args(
        "default", ["text", "image"], ["text"], input_signals,
    )
    assert initial.full_metadata.graph_walk == "prefill"
    assert initial.step_metadata["sample_prefill_token"] is False
    assert initial.inputs[0].tensor_info == [t0]

    step1 = model.get_partition_forward_pass_args(
        "default", initial.full_metadata, persist_signals={},
    )
    assert step1.full_metadata.graph_walk == "prefill_vision"
    assert step1.full_metadata.is_prefill is True
    assert step1.step_metadata["sample_prefill_token"] is False

    step2 = model.get_partition_forward_pass_args(
        "default", step1.full_metadata, persist_signals={},
    )
    assert step2.full_metadata.graph_walk == "prefill"
    assert step2.step_metadata["sample_prefill_token"] is True
    assert step2.inputs[0].tensor_info == [t1]

    step3 = model.get_partition_forward_pass_args(
        "default", step2.full_metadata, persist_signals={"new_token": [t1]},
    )
    assert step3.full_metadata.graph_walk == "decode"
    assert step3.full_metadata.is_prefill is False
    assert step3.request_done is False


def test_graph_walks_include_prefill_vision_when_vision_configured():
    walks = _make_model().get_graph_walk_graphs()
    assert set(walks) == {"prefill", "decode", "prefill_vision"}
    assert isinstance(walks["prefill_vision"], Sequential)


# --- process_prompt image path ------------------------------------------------


class _Enc:
    def __init__(self, ids):
        self.input_ids = torch.tensor([ids])


class _ImageStubTokenizer:
    """Renders to a fixed marker string, then hands back a canned token-id
    sequence — sidesteps needing a real HF tokenizer for the round trip."""

    def __init__(self, token_ids):
        self._token_ids = token_ids
        self.chat_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return "<rendered>"

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        return _Enc(self._token_ids)


def test_process_image_messages_expands_pad_token_and_splits_text():
    model = _make_model()
    vision_cfg = model.config.vision
    content, pad, end = (
        vision_cfg.media_content_token_id,
        vision_cfg.media_pad_token_id,
        vision_cfg.media_end_token_id,
    )
    model._tokenizer = _ImageStubTokenizer([10, 20, content, pad, end, 30, 40])
    model._tokenizer_mode = "hf"
    image = torch.rand(3, 56, 56)  # 4x4 patch grid at the default patch_size

    result = model.process_prompt(
        None, ["text", "image"], ["text"],
        tensors={"image_inputs": [image]},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "x"}, {"type": "image"},
        ]}],
    )

    assert [t.tolist() for t in result["text_inputs"]] == [[10, 20], [30, 40]]
    assert result["image_grids"][0].tolist() == [4, 4]
    assert result["image_inputs"][0].shape == (16, 3, vision_cfg.patch_size, vision_cfg.patch_size)


def test_process_image_messages_raises_on_segment_count_mismatch():
    model = _make_model()
    vision_cfg = model.config.vision
    content, pad, end = (
        vision_cfg.media_content_token_id,
        vision_cfg.media_pad_token_id,
        vision_cfg.media_end_token_id,
    )
    # Only one wrapped pad run in the rendered prompt, but two images arrived.
    model._tokenizer = _ImageStubTokenizer([content, pad, end])
    model._tokenizer_mode = "hf"

    with pytest.raises(ValueError, match="text segments"):
        model.process_prompt(
            None, ["text", "image", "image"], ["text"],
            tensors={"image_inputs": [torch.rand(3, 56, 56), torch.rand(3, 56, 56)]},
            messages=[{"role": "user", "content": [{"type": "image"}, {"type": "image"}]}],
        )


def test_process_prompt_images_without_messages_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="chat-template"):
        model.process_prompt(
            "hello", ["text", "image"], ["text"],
            tensors={"image_inputs": [torch.rand(3, 4, 4)]},
        )


# --- _create_vision_submodule --------------------------------------------------


class _StubVisionTower(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(2))


class _StubMMProjector(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(2))


def _make_checkpoint_model(tmp_path) -> KimiK2Model:
    model = object.__new__(KimiK2Model)
    model.model_path_hf = str(tmp_path)
    model.cache_dir = None
    model.config = KimiK2Config.k27_code()
    model._submodule_cache = {}
    return model


def test_create_vision_submodule_loads_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mstar.model.kimi_k2_7.components.vision.KimiVisionTower", _StubVisionTower,
    )
    monkeypatch.setattr(
        "mstar.model.kimi_k2_7.components.vision.KimiMMProjector", _StubMMProjector,
    )
    calls = {}

    def fake_load_weights_from_hf_shards(repo_dir, modules, device="cpu"):
        calls["repo_dir"] = repo_dir
        calls["prefixes"] = [m.prefix for m in modules]

    monkeypatch.setattr(
        "mstar.model.utils.load_weights_from_hf_shards", fake_load_weights_from_hf_shards,
    )
    submodule = _make_checkpoint_model(tmp_path)._create_vision_submodule("cpu")

    assert isinstance(submodule, KimiVisionEncoderSubmodule)
    assert calls == {"repo_dir": str(tmp_path), "prefixes": ["vision_tower", "mm_projector"]}


def test_create_vision_submodule_dummy_mode_when_no_checkpoint():
    model = object.__new__(KimiK2Model)
    model.model_path_hf = None
    model.config = KimiK2Config.k27_code()
    assert model._create_vision_submodule("cpu") is None


def test_create_submodule_dispatches_to_vision_creator(monkeypatch):
    model = object.__new__(KimiK2Model)
    model.config = KimiK2Config.k27_code()
    model._submodule_cache = {}
    calls = []
    monkeypatch.setattr(
        model, "_create_vision_submodule", lambda device: calls.append(device) or "SENTINEL",
    )
    assert model._create_submodule(VISION_NODE, "cpu") == "SENTINEL"
    assert calls == ["cpu"]


# --- KimiLLMSubmodule: prefill_vision -------------------------------------------


class _FakeInner(torch.nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden)

    def forward(self, input_ids=None, inputs_embeds=None, label=None):
        return inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)


class _FakeLM(torch.nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.model = _FakeInner(vocab_size, hidden)
        self.lm_head = torch.nn.Identity()


def _reduced_vision_llm_config() -> KimiK2Config:
    cfg = KimiK2Config.reduced()
    cfg.vision = KimiVisionConfig()
    return cfg


def _fwd_info(sample_prefill_token: bool = True) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="r0", graph_walk="prefill_vision", fwd_index=0, random_seed=0,
        max_tokens=1, step_metadata={"sample_prefill_token": sample_prefill_token},
    )


def test_llm_submodule_wraps_vision_embeddings_with_sentinels():
    cfg = _reduced_vision_llm_config()
    vocab_size = cfg.vision.media_end_token_id + 1
    submodule = KimiLLMSubmodule(language_model=_FakeLM(vocab_size, cfg.hidden_size), config=cfg)
    image_embeds = torch.randn(3, cfg.hidden_size)

    node_inputs = submodule.prepare_inputs(
        graph_walk="prefill_vision", fwd_info=_fwd_info(), inputs={"image_embeds": [image_embeds]},
    )

    assert node_inputs.input_embeds.shape == (5, cfg.hidden_size)  # content + 3 + end
    assert node_inputs.input_seq_len == 5
    assert torch.equal(node_inputs.input_embeds[1:4], image_embeds)
    assert node_inputs.resource_step_info is True


def test_llm_submodule_forward_prefill_vision_skips_sampling_when_not_last():
    cfg = _reduced_vision_llm_config()
    lm = _FakeLM(cfg.vocab_size, cfg.hidden_size)
    submodule = KimiLLMSubmodule(language_model=lm, config=cfg)
    info = CurrentForwardPassInfo(
        request_id="r0", graph_walk="prefill_vision", fwd_index=0, random_seed=0,
        max_tokens=10, step_metadata={"sample_prefill_token": False},
    )
    engine_inputs = ModelInputsFromEngine(request_ids=["r0"], per_request_info={"r0": info})

    out = submodule.forward(
        graph_walk="prefill_vision", engine_inputs=engine_inputs,
        input_embeds=torch.randn(5, cfg.hidden_size),
    )

    assert out == {}


class _FakeBatch:
    def __init__(self, graph_walk, per_request_info):
        self.graph_walk = graph_walk
        self.per_request_info = per_request_info


def test_can_batch_never_batches_prefill_vision():
    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    submodule = KimiLLMSubmodule(language_model=language_model, config=KimiK2Config.reduced())
    assert submodule.can_batch(_FakeBatch("prefill_vision", {}), []) is False


def test_can_batch_prefill_requires_every_request_ready_to_sample():
    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    submodule = KimiLLMSubmodule(language_model=language_model, config=KimiK2Config.reduced())
    ready = CurrentForwardPassInfo(
        request_id="a", graph_walk="prefill", fwd_index=0, random_seed=0, max_tokens=1,
        step_metadata={"sample_prefill_token": True},
    )
    not_ready = CurrentForwardPassInfo(
        request_id="b", graph_walk="prefill", fwd_index=0, random_seed=0, max_tokens=1,
        step_metadata={"sample_prefill_token": False},
    )
    assert submodule.can_batch(_FakeBatch("prefill", {"a": ready}), []) is True
    assert submodule.can_batch(_FakeBatch("prefill", {"a": ready, "b": not_ready}), []) is False


def test_cg_key_info_none_when_every_request_is_sampling():
    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    submodule = KimiLLMSubmodule(language_model=language_model, config=KimiK2Config.reduced())
    ready = CurrentForwardPassInfo(
        request_id="a", graph_walk="prefill", fwd_index=0, random_seed=0, max_tokens=1,
        step_metadata={"sample_prefill_token": True},
    )
    assert submodule.cg_key_info("prefill", {"a": ready}) is None


def test_cg_key_info_false_when_one_request_is_not_sampling():
    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    submodule = KimiLLMSubmodule(language_model=language_model, config=KimiK2Config.reduced())
    ready = CurrentForwardPassInfo(
        request_id="a", graph_walk="prefill", fwd_index=0, random_seed=0, max_tokens=1,
        step_metadata={"sample_prefill_token": True},
    )
    not_ready = CurrentForwardPassInfo(
        request_id="b", graph_walk="prefill", fwd_index=0, random_seed=0, max_tokens=1,
        step_metadata={"sample_prefill_token": False},
    )
    assert submodule.cg_key_info("prefill", {"a": ready, "b": not_ready}) is False


def test_declare_step_stamps_cg_key_info_matching_cg_key_info_method():
    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    submodule = KimiLLMSubmodule(language_model=language_model, config=KimiK2Config.reduced())

    all_sampling_inputs = [
        ARNodeInputs(
            input_ids=torch.tensor([1, 2]), input_seq_len=2, resource_step_info=True,
        ),
    ]
    step = submodule.declare_step("prefill", ["r0"], all_sampling_inputs)
    ready = CurrentForwardPassInfo(
        request_id="r0", graph_walk="prefill", fwd_index=0, random_seed=0, max_tokens=1,
        step_metadata={"sample_prefill_token": True},
    )
    assert step.cg_key_info is None
    assert step.cg_key_info == submodule.cg_key_info("prefill", {"r0": ready})

    mixed_inputs = [
        ARNodeInputs(input_ids=torch.tensor([1]), input_seq_len=1, resource_step_info=True),
        ARNodeInputs(input_ids=torch.tensor([2]), input_seq_len=1, resource_step_info=False),
    ]
    step2 = submodule.declare_step("prefill", ["r0", "r1"], mixed_inputs)
    not_ready = CurrentForwardPassInfo(
        request_id="r1", graph_walk="prefill", fwd_index=0, random_seed=0, max_tokens=1,
        step_metadata={"sample_prefill_token": False},
    )
    assert step2.cg_key_info is False
    assert step2.cg_key_info == submodule.cg_key_info("prefill", {"r0": ready, "r1": not_ready})
