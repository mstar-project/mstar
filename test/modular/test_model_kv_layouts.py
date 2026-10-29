"""Each model lays out every label it declares, and caps each growth loop where its stop check ends it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import ForwardPassInfoWrapper
from mstar.engine.resources import SamplingReqConfig
from mstar.model.base import LabelLayout
from mstar.model.submodule_base import InputMetadata

NOT_EOS = 42


def _stops_after(check_stop, loop: str, walk: str, max_iters: int, outputs: dict, **info) -> int:
    resource_configs = info.pop("resource_configs", {"sampler": SamplingReqConfig(ignore_eos=True)})
    for k in range(max_iters):
        request_info = ForwardPassInfoWrapper(
            info=CurrentForwardPassInfo(
                request_id="r", graph_walk=walk, fwd_index=0, random_seed=0,
                resource_configs=resource_configs, **info,
            ),
            metadata=InputMetadata(dynamic_loop_iter_counts={loop: k}),
        )
        if loop in check_stop("r", request_info, outputs):
            return k + 1
    return max_iters


def _orpheus():
    from mstar.model.orpheus.config import OrpheusModelConfig
    from mstar.model.orpheus.orpheus_model import OrpheusModel

    model = object.__new__(OrpheusModel)
    model.config = OrpheusModelConfig()
    model.tokenizer = lambda text, return_tensors=None: SimpleNamespace(
        input_ids=torch.tensor([[ord(c) for c in text]], dtype=torch.long),
    )
    return model


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_orpheus_caps_decode_where_its_stop_check_ends_it(max_tokens):
    from mstar.model.orpheus.submodules import OrpheusLLMSubmodule

    model = _orpheus()
    llm = OrpheusLLMSubmodule(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=None), lm_head=None), config=model.config,
    )

    cap = model.get_loop_caps(max_output_tokens=max_tokens)["decode_loop"]

    assert cap == _stops_after(
        llm.check_stop, "decode_loop", "decode", 2048,
        {"new_token": [torch.tensor([NOT_EOS])]}, max_tokens=max_tokens,
    ), "the cap and the stop check end decode on different iterations"


def _bagel():
    from test_multimodal_ordering import _StubTokenizer

    from mstar.model.bagel.bagel_model import BagelModel

    model = object.__new__(BagelModel)
    model.config = SimpleNamespace(
        think_mode=False, cfg_text_scale=4.0, cfg_img_scale=1.0, cfg_interval=[0.4, 1.0],
        cfg_renorm_type="global", cfg_renorm_min=0.0, vit_config=SimpleNamespace(patch_size=14),
        vit_max_num_patch_per_side=70, latent_downsample=16, max_latent_size=64,
        temperature=0.0, top_k=-1, top_p=1.0, repetition_penalty=1.0, ignore_eos=False, vocab_size=1000,
    )
    model.tokenizer = _StubTokenizer()
    model.boi_token_id = _StubTokenizer.SPECIALS["<|vision_start|>"]
    model.eoi_token_id = _StubTokenizer.SPECIALS["<|vision_end|>"]
    model._has_cfg_parallel = model._has_llm_disaggregation = False
    return model


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_bagel_caps_decode_where_its_stop_check_ends_it(max_tokens):
    from test_bagel_guidance_sink import _llm

    llm = _llm()

    cap = _bagel().get_loop_caps(max_output_tokens=max_tokens)["decode_loop"]

    assert cap == _stops_after(
        llm.check_stop, "decode_loop", "decode", 2048,
        {"new_token": [torch.tensor([NOT_EOS])]}, max_tokens=max_tokens,
    ), "the cap and the stop check end decode on different iterations"


_BAGEL_RUNS = {
    "text chat": (["text"], ["text"], {}, "Hello there"),
    "understanding an image": (["image", "text"], ["text"], {}, "What is this?"),
    "understanding, text first": (["text", "image"], ["text"], {}, "What is this?"),
    "text to image": (["text"], ["image"], {}, "A cat on a mat"),
    "text to image, think": (["text"], ["image"], {"think_mode": True}, "A cat on a mat"),
    "text to image without guidance": (["text"], ["image"], {"cfg_text_scale": 1.0}, "A cat on a mat"),
    "an edit": (["image", "text"], ["image"], {}, "Make it night"),
    "understanding without text": (["image"], ["text"], {}, None),
    "an edit without text": (["image"], ["image"], {}, None),
    "an edit without guidance": (["image", "text"], ["image"], {"cfg_text_scale": 1.0, "cfg_img_scale": 1.0}, "Night"),
}


def _pointer(tensor: torch.Tensor):
    from mstar.graph.base import TensorPointerInfo

    return TensorPointerInfo(
        dims=list(tensor.shape), dtype=tensor.dtype, nbytes=0, address=0, stride=[], uuid=id(tensor),
        source_session_id="", source_entity="",
    )


def _bagel_request(model, prompt, input_modalities, output_modalities, kwargs):
    from mstar.conductor.request_info import DEFAULT_PARTITION
    from mstar.model.bagel.submodules import IMAGE_SENTINELS, VAEEncoderSubmodule, vit_patches, vit_preprocess

    cfg = model.config
    images = [torch.rand(3, 480, 640)] if "image" in input_modalities else []
    out = model.process_prompt(prompt, input_modalities, output_modalities, tensors={"image_inputs": images}, **kwargs)
    tensors = {"text_inputs": out.new_input_tensors.get("text_inputs", []),
               "image_inputs": out.new_input_tensors.get("image_inputs", images)}
    signals = {name: [_pointer(t) for t in by_name] for name, by_name in tensors.items()}
    fwd = model.get_initial_forward_pass_args(DEFAULT_PARTITION, input_modalities, output_modalities, signals, kwargs)
    metadata = fwd.full_metadata
    by_uuid = {id(t): t for by_name in tensors.values() for t in by_name}
    later = []
    while not model.get_partition_forward_pass_args(DEFAULT_PARTITION, metadata, {}).request_done:
        if not metadata.is_prefill:
            later.append(metadata.graph_walk)
            if metadata.graph_walk in ("image_gen", "image_gen_cfg"):
                break
    config = model.get_request_resource_configs({DEFAULT_PARTITION: fwd}, kwargs)["kv"]
    cap = model.get_loop_caps(max_output_tokens=6)["decode_loop"]
    config.apply_conductor_config(
        kv_layout={label: layout.row() for label, layout in out.metadata["kv_layout"]["kv"].items()},
        loop_caps={"decode_loop": cap},
    )
    vae = VAEEncoderSubmodule(
        SimpleNamespace(posterior_noise=lambda *args, **kw: None), None, None, None, latent_patch_size=1,
        latent_channel=16, latent_downsample=cfg.latent_downsample, max_latent_size=cfg.max_latent_size,
    )
    preprocess = metadata.kwargs["image_preprocess"]
    walks = []
    for walk, info in metadata.kwargs["prefill_schedule"]:
        tensor = by_uuid[info.uuid]
        if walk == "prefill_text":
            walks.append((walk, tensor.numel()))
        elif walk == "prefill_vit":
            patches = vit_patches(
                *tensor.shape[-2:], vit_preprocess(preprocess), cfg.vit_config.patch_size,
                cfg.vit_max_num_patch_per_side,
            )
            walks.append((walk, patches + IMAGE_SENTINELS))
        else:
            info = SimpleNamespace(step_metadata={"image_preprocess": preprocess}, random_seed=0)
            encoded = vae.prepare_inputs(walk, info, {"image_inputs": [tensor]})
            walks.append((walk, encoded.kwargs["h"] * encoded.kwargs["w"] + IMAGE_SENTINELS))
    height, width = metadata.kwargs["height"], metadata.kwargs["width"]
    latents = (height // cfg.latent_downsample) * (width // cfg.latent_downsample)
    for walk in later:
        walks += [(walk, 1)] * cap if walk == "decode" else [(walk, latents + IMAGE_SENTINELS)] * 3
    return config, metadata.kwargs["requires_cfg"], walks


class _BagelPool:

    def __init__(self, configs: dict):
        from test_bagel_guidance_sink import _llm

        from mstar.engine.resources import PagedKVConfig, StepRunner
        from mstar.engine.resources.kv.manager import KVManager

        self.kv = KVManager(
            cfg=PagedKVConfig(
                num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=32768, max_num_pages=4096, page_size=16,
            ),
            name="kv", joint_comm_group=None, transfer_engine_info=None, device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.runner = StepRunner({"kv": self.kv}, node_resources={"LLM": ["kv"]})
        for rid, config in configs.items():
            self.runner.ingest_request(rid, {"kv": config})
        self.llm = _llm()

    def step(self, walk: str, rows: dict, guided: dict) -> None:
        from mstar.engine.resources import StepContext
        from mstar.engine.resources.step import SubmoduleStep
        from mstar.model.submodule_base import ARNodeInputs

        inputs = []
        for rid, count in rows.items():
            inputs.append(ARNodeInputs(input_seq_len=count, resource_step_info=guided[rid]))
            if walk not in {"prefill_text", "decode"}:
                positions = torch.zeros(count, dtype=torch.int32)
                inputs[-1].custom_pos_ids = dict.fromkeys(("main", "cfg_text", "cfg_img"), positions)
        kv_step = self.llm.declare_step(walk, list(rows), inputs, per_request_info={}).steps["kv"]
        submodule_step = SubmoduleStep(steps={"kv": kv_step}, segments=list(kv_step.segments))
        submodule_step.set_ctx(StepContext(request_ids=tuple(rows), graph_walk=walk, slot=0, capture=False))
        assert self.runner.admit(submodule_step).outcome.ok
        self.runner.plan(submodule_step)
        self.runner.commit(submodule_step)

    def held(self, rid) -> dict:
        return {label: (stream.peak, stream.footprint) for label, stream in self.kv._streams[rid].items()}


@pytest.fixture
def _bagel_checks(monkeypatch):
    from mstar.engine.resources.kv import manager as manager_mod

    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    monkeypatch.setattr(
        manager_mod, "KVTransferManager", lambda *a, **kw: SimpleNamespace(remove_request=lambda rid: None),
    )
    model = _bagel()
    model.config.num_timesteps = 3
    return model


@pytest.mark.parametrize(
    ("input_modalities", "output_modalities", "kwargs", "prompt"), _BAGEL_RUNS.values(), ids=_BAGEL_RUNS,
)
def test_bagel_peaks_at_its_footprint_on_every_label(
    input_modalities, output_modalities, kwargs, prompt, _bagel_checks,
):
    config, guided, walks = _bagel_request(_bagel_checks, prompt, input_modalities, output_modalities, kwargs)
    pool = _BagelPool({1: config})

    for walk, rows in walks:
        pool.step(walk, {1: rows}, {1: guided})

    held = pool.held(1)
    assert all(peak == footprint for peak, footprint in held.values()), f"(peak, footprint) by label: {held}"


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_chatterbox_caps_decode_where_its_stop_check_ends_it(max_tokens):
    from test_chatterbox_model import _make_model

    from mstar.model.chatterbox.submodules import SPEECH_TOKENS, T3Submodule

    model = _make_model()
    t3 = T3Submodule(None, model.config, builtin_voice=None)
    outputs = {SPEECH_TOKENS: [torch.tensor([model.config.t3.stop_speech_token + 1])]}
    t3.request_state("r").add_all(generated=0, requires_cfg=False)
    t3.postprocess("r", None, outputs)

    def check_stop(request_id, request_info, outputs):
        t3.postprocess(request_id, request_info, outputs)
        return t3.check_stop(request_id, request_info, outputs)

    cap = model.get_loop_caps(max_new_tokens=max_tokens)["decode_loop"]

    assert cap == _stops_after(
        check_stop, "decode_loop", "decode", model.config.t3.max_speech_tokens, outputs,
        max_tokens=max_tokens,
    ), "the cap and the stop check end decode on different iterations"


@pytest.mark.parametrize("variant", ["chatterbox", "turbo"])
def test_a_reference_clip_is_laid_out_at_the_tokens_the_tokenizer_makes_of_it(variant):
    from mstar.model.chatterbox.components.audio_frontend import resample, trim_reference
    from mstar.model.chatterbox.components.s3_tokenizer import S3Tokenizer
    from mstar.model.chatterbox.config import S3_SR, ChatterboxConfig
    from mstar.model.chatterbox.submodules import reference_tokens

    config = ChatterboxConfig.from_variant(variant)
    tokenizer = S3Tokenizer(config.s3_tokenizer).eval()
    for samples in (14_400, 144_001, 720_000):
        wav16 = trim_reference(resample(torch.rand(samples), config.sample_rate, S3_SR), S3_SR, config.enc_cond_seconds)
        _, lens = tokenizer([wav16], max_len=config.t3.speech_cond_prompt_len)

        assert int(lens[0]) == reference_tokens(samples, config), f"a {samples}-sample clip is laid out wrong"


@pytest.mark.parametrize(("cfg_weight", "clip"), [(0.0, False), (0.5, False), (0.5, True)])
def test_chatterbox_lays_out_every_label_it_declares(cfg_weight, clip):
    from test_chatterbox_model import _make_model, _tone

    model = _make_model()
    tensors = {"audio_inputs": [_tone(2.0)]} if clip else None

    output = model.process_prompt("hello", ["text", "audio"] if clip else ["text"], ["audio"],
                                  tensors=tensors, cfg_weight=cfg_weight)

    opened = model.get_request_resource_configs({}, {"cfg_weight": cfg_weight})["t3_kv"].opened_labels()
    assert set(output.metadata["kv_layout"]["t3_kv"]) == opened, "the layout doesn't name the labels the request opens"


class _Cosmos3Tokenizer:

    eos_token_id = 1

    def apply_chat_template(self, conversations, **kwargs):
        return {"input_ids": [ord(c) for c in conversations[-1]["content"]]}

    def convert_tokens_to_ids(self, token):
        return 2


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_cosmos3_caps_the_reasoners_decode_where_its_stop_check_ends_it(max_tokens):
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.cosmos3.submodules import REASONER_DECODE_LOOP, Cosmos3ReasonerSubmodule

    reasoner = SimpleNamespace(eos_token_id=None)

    def check_stop(request_id, request_info, outputs):
        return Cosmos3ReasonerSubmodule.check_stop(reasoner, request_id, request_info, outputs)

    cap = Cosmos3Model.get_loop_caps(object.__new__(Cosmos3Model), max_output_tokens=max_tokens)[REASONER_DECODE_LOOP]

    assert cap == _stops_after(
        check_stop, REASONER_DECODE_LOOP, "reasoner_decode", 2048,
        {"new_token": [torch.tensor([NOT_EOS])]}, max_tokens=max_tokens,
    ), "the cap and the stop check end the reasoner's decode on different iterations"


_COSMOS3_COMMITS = {
    "text to image": ("a fox in the snow", ["image"], {}),
    "text to image without guidance": ("a fox in the snow", ["image"], {"guidance_scale": 1.0}),
    "text to video": ("a fox in the snow", ["video"], {"num_frames": 33}),
    "kv windows": ("a fox in the snow", ["video"], {"window_mode": "kv", "num_frames": 121}),
    "kv windows without context": ("a fox", ["video"], {"window_mode": "kv", "num_frames": 121, "context_frames": 0}),
}


@pytest.mark.parametrize(("prompt", "outputs", "kwargs"), _COSMOS3_COMMITS.values(), ids=_COSMOS3_COMMITS)
def test_cosmos3_lays_out_what_its_dit_commits(tmp_path, prompt, outputs, kwargs):
    from test_cosmos3_edge_model import _model

    from mstar.model.cosmos3.submodules import COND_LABEL, UNCOND_LABEL, Cosmos3DiTSubmodule

    model = _model(tmp_path, enable_windowed_video=True)
    model.tokenizer = _Cosmos3Tokenizer()
    dit = Cosmos3DiTSubmodule(None, model.config, scheduler=None)
    dit.get_device = lambda: torch.device("cpu")
    dit._new_scheduler = lambda *args, **kw: SimpleNamespace(timesteps=torch.arange(4))

    output = model.process_prompt(prompt, ["text"], outputs, tensors={}, **kwargs)
    args = model.get_initial_forward_pass_args("default", ["text"], outputs, {"text_inputs": []}, kwargs)
    walk = args.full_metadata.graph_walk
    info = CurrentForwardPassInfo(request_id="r", graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=1,
                                  rid_handle="r", step_metadata=args.step_metadata)
    dit.prepare_inputs(walk, info, {"text_inputs": output.new_input_tensors.get("text_inputs", [])})

    state, layout = dit.request_states["r"], output.metadata["kv_layout"]["kv"]
    for label, static in ((COND_LABEL, state["cond"]), (UNCOND_LABEL, state["uncond"])):
        laid_out = layout[label]
        if static is None:
            assert laid_out == LabelLayout(), f"{label} is laid out for a branch the DiT doesn't run"
            continue
        assert (laid_out.spans[0].walk, laid_out.spans[0].length) == (walk, static["und_len"]), label
        if state.get("ar_schedule") is None:
            step = static["num_vision_tokens"] + static.get("num_action_tokens", 0) + (state.get("num_sound") or 0)
            assert laid_out.scratch == step, f"{label}'s scratch isn't one denoise step"
            continue
        schedule, per_unit = state["ar_schedule"], state["ar_tokens_per_unit"]
        commits = [(plan.commit_end - plan.commit_start) * per_unit for plan in schedule.windows()]
        assert [span.length for span in laid_out.spans[1:]] == commits, f"{label}'s windows aren't the commits"
        assert [span.scratch for span in laid_out.spans[1:]] == [plan.units * per_unit for plan in schedule.windows()]
        assert laid_out.scratch == 0, f"{label}'s denoise is counted on top of its last window"
        assert laid_out.retention == dit._window_retention(state).get(label), f"{label}'s retention isn't the DiT's"


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_fish_caps_decode_where_its_stop_check_ends_it(max_tokens):
    from test_fish_s2_pro_model import tiny_ar, tiny_model

    ar = tiny_ar()
    outputs = {"new_token": [torch.tensor([4])]}
    ar.request_state("r").add("generated_frames", 0)
    ar.postprocess("r", None, outputs)
    prefill = SimpleNamespace(step_metadata={}, max_tokens=max_tokens, resource_configs={})
    ar.check_stop("r", prefill, outputs)

    cap = tiny_model().get_loop_caps(max_output_tokens=max_tokens)["ar_decode_loop"]

    assert cap == _stops_after(
        ar.check_stop, "ar_decode_loop", "ar_decode", 4096, outputs,
        max_tokens=max_tokens,
    ), "the cap and the stop check end decode on different iterations"


@pytest.mark.parametrize("clone", [False, True])
def test_fish_lays_out_the_rows_its_prefill_builds(clone):
    from test_fish_s2_pro_model import tiny_ar, tiny_model

    from mstar.model.fish_s2_pro.config import AR_KV, PROMPT_LAYOUT_KEY

    model = tiny_model()
    tensors = {"audio_inputs": [torch.zeros(20 * model.config.codec.hop_length + 1)]} if clone else None

    output = model.process_prompt("hi", ["text", "audio"] if clone else ["text"], ["audio"], tensors, ref_text="r")

    layout = output.metadata["kv_layout"]
    [span] = layout[AR_KV]["main"].spans
    assert span.walk == ("ar_prefill_clone" if clone else "ar_prefill")
    prefix_len, frames = output.metadata[PROMPT_LAYOUT_KEY]
    prefill = tiny_ar()._build_prefill(
        torch.zeros_like(output.new_input_tensors["prompt_ids"][0]), prefix_len, frames,
        torch.zeros(frames, 4, dtype=torch.long) if frames else None,
    )
    assert span.length == prefill.shape[0], "the layout and the prefill disagree on its rows"


def _higgs():
    from mstar.model.higgs_audio.config import HiggsAudioModelConfig
    from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel

    model = object.__new__(HiggsAudioModel)
    model.config = HiggsAudioModelConfig()
    model.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [ord(c) for c in text])
    model.feature_extractor = lambda chunks, **kwargs: {
        "input_features": torch.zeros(len(chunks), 128, 3000),
        "attention_mask": torch.stack([torch.arange(3000) < len(chunk) // 160 for chunk in chunks]),
    }
    return model


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_higgs_caps_decode_where_its_stop_check_ends_it(max_tokens):
    from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule

    model = _higgs()
    llm = HiggsAudioLLMSubmodule(None, model.config)

    cap = model.get_loop_caps(max_output_tokens=max_tokens)["decode_loop"]

    assert cap == _stops_after(
        llm.check_stop, "decode_loop", "decode", 1024,
        {"new_token": [torch.tensor([NOT_EOS])]}, max_tokens=max_tokens,
    ), "the cap and the stop check end decode on different iterations"


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_qwen3_5_caps_decode_where_its_stop_check_ends_it(max_tokens):
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    llm = SimpleNamespace(config=SimpleNamespace(stop_token_ids=()), num_draft_tokens=0)

    def check_stop(request_id, request_info, outputs):
        return LLMSubmodule.check_stop(llm, request_id, request_info, outputs)

    cap = Qwen3_5DenseModel.get_loop_caps(object.__new__(Qwen3_5DenseModel), max_output_tokens=max_tokens)

    assert cap["decode_loop"] == _stops_after(
        check_stop, "decode_loop", "decode", 2048,
        {"new_token": [torch.tensor([NOT_EOS])]}, max_tokens=max_tokens,
    ), "the cap and the stop check end decode on different iterations"


def _omni_submodule():
    return SimpleNamespace(config=SimpleNamespace(im_end_token_id=0, talker=SimpleNamespace(codec_eos_token_id=0)))


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_qwen3_omni_caps_the_thinkers_decode_where_its_stop_check_ends_it(max_tokens):
    pytest.importorskip("flashinfer")  # imported by Qwen3OmniModel's submodules
    from mstar.model.qwen3_omni.config import THINKER_SAMPLER
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
    from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

    cap = Qwen3OmniModel.get_loop_caps(object.__new__(Qwen3OmniModel), max_output_tokens=max_tokens)

    assert cap["thinker_decode_loop"] == _stops_after(
        lambda *args: ThinkerSubmodule.check_stop(_omni_submodule(), *args), "thinker_decode_loop", "thinker_decode",
        2048, {"new_token": [torch.tensor([NOT_EOS])]}, max_tokens=max_tokens,
        resource_configs={THINKER_SAMPLER: SamplingReqConfig(ignore_eos=True)},
    ), "the cap and the stop check end decode on different iterations"


@pytest.mark.parametrize("talker_max_tokens", [0, 1, 64])
def test_qwen3_omni_caps_the_talkers_decode_where_its_stop_check_ends_it(talker_max_tokens):
    pytest.importorskip("flashinfer")
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
    from mstar.model.qwen3_omni.submodules import TalkerSubmodule

    cap = Qwen3OmniModel.get_loop_caps(object.__new__(Qwen3OmniModel), talker_max_output_tokens=talker_max_tokens)

    assert cap["talker_decode_loop"] == _stops_after(
        lambda *args: TalkerSubmodule.check_stop(_omni_submodule(), *args), "talker_decode_loop", "talker_decode",
        2048, {"layer0_codes": [torch.tensor([NOT_EOS])]}, max_tokens=2048,
        step_metadata={"talker_max_tokens": talker_max_tokens},
    ), "the cap and the stop check end decode on different iterations"


def _tiny_qwen3_tts_talker():
    from test_qwen3_tts_model import _tiny_model_config

    from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
    from mstar.model.qwen3_tts.submodules import TalkerSubmodule

    config = _tiny_model_config()
    return TalkerSubmodule(Qwen3TTSTalkerModel(config), Qwen3TTSCodePredictor(config), config)


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_qwen3_tts_caps_decode_where_its_stop_check_ends_it(max_tokens):
    from test_qwen3_tts_model import _make_model

    talker = _tiny_qwen3_tts_talker()
    outputs = {"new_token": [torch.tensor(NOT_EOS)]}
    talker.request_state("r").add_all(generated_frames=0)
    talker.postprocess("r", None, outputs)

    def check_stop(request_id, request_info, outputs):
        talker.postprocess(request_id, request_info, outputs)
        return talker.check_stop(request_id, request_info, outputs)

    cap = _make_model().get_loop_caps(max_output_tokens=max_tokens)["talker_decode_loop"]

    assert cap == _stops_after(
        check_stop, "talker_decode_loop", "talker_decode", 2048, outputs,
        max_tokens=max_tokens,
    ), "the cap and the stop check end decode on different iterations"


LONG_TEXT = "one two three four five six seven eight nine ten eleven twelve"
_QWEN3_TTS_SHAPES = {
    "custom voice": ("custom_voice", "hello big world", {"language": "English"}),
    "custom voice, streamed": ("custom_voice", "hello big world", {"non_streaming_mode": False}),
    "voice design": ("voice_design", "hello big world", {"instruct": "a calm low voice"}),
    "x-vector clone": ("base", "hello big world", {"x_vector_only_mode": True}),
    "in-context clone, text past the clip": ("base", LONG_TEXT, {"ref_text": "the reference says"}),
    "in-context clone, text short of the clip": ("base", "hi", {"ref_text": "ok"}),
    "in-context clone, not streamed": ("base", "hello big world", {"ref_text": "ok", "non_streaming_mode": True}),
}


@pytest.mark.parametrize(("variant", "text", "kwargs"), _QWEN3_TTS_SHAPES.values(), ids=_QWEN3_TTS_SHAPES)
def test_qwen3_tts_lays_out_the_rows_its_prefill_builds(variant, text, kwargs):
    from test_qwen3_tts_model import _variant_model

    from mstar.model.qwen3_tts.config import LANGUAGE_ID_KEY, PROMPT_LAYOUT_KEY, SPEAKER_ID_KEY, TALKER_KV

    model = _variant_model(variant)
    clone = variant == "base"
    tensors = {"audio_inputs": [torch.zeros(24000 + 1)]} if clone else None

    output = model.process_prompt(
        text, ["audio", "text"] if clone else ["text"], ["audio"], tensors=tensors, **kwargs,
    )

    layout, metadata = output.metadata["kv_layout"], output.metadata
    [span] = layout[TALKER_KV]["main"].spans
    assert span.walk == ("talker_prefill_clone" if clone else "talker_prefill")
    frames = metadata[PROMPT_LAYOUT_KEY][4]
    prefill = _tiny_qwen3_tts_talker()._build_prefill(
        "r", torch.zeros_like(output.new_input_tensors["text_inputs"][0]), metadata[PROMPT_LAYOUT_KEY],
        speaker_id=40 if metadata[SPEAKER_ID_KEY] >= 0 else -1,
        language_id=41 if metadata[LANGUAGE_ID_KEY] >= 0 else -1,
        speaker_embed=torch.randn(16) if clone else None,
        ref_codes=torch.randint(0, 32, (frames, 4)) if frames else None,
    )
    assert span.length == prefill.shape[0], "the layout and the prefill disagree on its rows"


def _vjepa2(predictor_kind: str = "ac", extrinsics: bool = False):
    from mstar.model.vjepa2.vjepa2_model import VJepa2Model

    model = VJepa2Model("unused", skip_weight_loading=True, predictor_kind=predictor_kind)
    if extrinsics:
        model.config.ac_predictor.use_extrinsics = True
    return model


@pytest.mark.parametrize("horizon", [2, 40])
def test_vjepa2_caps_the_rollout_where_its_stop_check_ends_it(horizon):
    from mstar.model.vjepa2.submodules import VJepa2ACRolloutPredictorSubmodule

    model = _vjepa2()
    kwargs = {"rollout_horizon": horizon}
    step_metadata = model.get_initial_forward_pass_args("p0", ["video"], ["video"], {}, kwargs).step_metadata

    assert model.get_loop_caps(**kwargs)["rollout_loop"] == _stops_after(
        VJepa2ACRolloutPredictorSubmodule(None, model.config).check_stop, "rollout_loop", "prefill_video_rollout",
        64, {}, max_tokens=1, step_metadata=step_metadata,
    ), "the cap and the stop check end the rollout on different iterations"


@pytest.mark.parametrize("extrinsics", [False, True])
def test_vjepa2_lays_out_a_frame_per_rollout_iteration(extrinsics):
    from mstar.model.vjepa2.config import KV_CACHE
    from mstar.model.vjepa2.submodules import VJepa2ACRolloutPredictorSubmodule

    model = _vjepa2(extrinsics=extrinsics)
    kwargs = {"actions": [[0.1] * 7] * 5, "states": [[0.2] * 7] * 5}
    if extrinsics:
        kwargs["extrinsics"] = [[0.3] * 6] * 5

    rollout = model.process_prompt(None, ["video"], ["video"], rollout_horizon=5, **kwargs).metadata["kv_layout"]
    once = model.process_prompt(None, ["video"], ["video"], **kwargs).metadata["kv_layout"]

    step = VJepa2ACRolloutPredictorSubmodule(None, model.config).prepare_inputs(
        "prefill_video_rollout", SimpleNamespace(dynamic_loop_iter_counts={"rollout_loop": 0}), {
            "encoder_hidden": [torch.zeros(1, model.config.grid_size ** 2, model.config.hidden_size)],
            **{name: [torch.tensor(value)] for name, value in kwargs.items()},
        },
    )
    assert rollout[KV_CACHE]["main"].growth == (("rollout_loop", step.input_seq_len),)
    assert once[KV_CACHE]["main"] == LabelLayout(), "a walk that never runs the rollout is laid out"
    assert _vjepa2("masked").process_prompt(None, ["video"], ["video"]).metadata["kv_layout"] == {}


def _whisper():
    from transformers import WhisperFeatureExtractor

    from mstar.model.whisper.config import WhisperModelConfig
    from mstar.model.whisper.whisper_model import WhisperModel

    model = object.__new__(WhisperModel)
    model.config = WhisperModelConfig(lang_to_id={"<|en|>": 50259}, task_to_id={"transcribe": 50360})
    model.feature_extractor = WhisperFeatureExtractor(feature_size=model.config.num_mel_bins)
    return model


@pytest.mark.parametrize("max_tokens", [1, 1000])
def test_whisper_caps_decode_where_its_stop_check_ends_it(max_tokens):
    from mstar.model.whisper.config import SAMPLER
    from mstar.model.whisper.submodules import WhisperDecoderSubmodule

    model = _whisper()
    bound = model.get_max_output_tokens(max_output_tokens=max_tokens)

    assert model.get_loop_caps(max_output_tokens=max_tokens)["decode_loop"] == _stops_after(
        WhisperDecoderSubmodule(None, model.config).check_stop, "decode_loop", "decode", 2048,
        {"new_token": [torch.tensor([NOT_EOS])]}, max_tokens=bound,
        resource_configs={SAMPLER: SamplingReqConfig(ignore_eos=True)},
    ), "the cap and the stop check end decode on different iterations"


@pytest.mark.parametrize("seconds", [1.0, 29.0, 45.0])
def test_whisper_lays_out_the_forced_prompt_and_the_encoders_whole_window(seconds):
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    from mstar.model.whisper.config import CONTEXT_LABEL, CROSS_KV_CACHE, KV_CACHE

    model = _whisper()
    clip = torch.zeros(int(seconds * 16000))

    output = model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [clip]})

    layout = output.metadata["kv_layout"]
    assert [span.length for span in layout[KV_CACHE]["main"].spans] == [len(output.new_input_tensors["text_inputs"][0])]
    encoder = WhisperEncoder(WhisperConfig(
        d_model=16, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32,
        num_mel_bins=model.config.num_mel_bins, max_source_positions=model.config.max_source_positions,
    )).eval()
    with torch.no_grad():
        rows = encoder(output.new_input_tensors["audio_features"][0][None]).last_hidden_state.shape[1]
    assert [span.length for span in layout[CROSS_KV_CACHE][CONTEXT_LABEL].spans] == [rows]
