"""Canary-Qwen's loading on fake inputs: the NeMo encoder names, the LLM from
the checkpoint's PEFT names with its LoRA merged, the config read from NeMo's
and the base LLM's ``config.json``, the prompt, the splice and the OpenAI
adapter. The LLM's forward is ``test_qwen3_lm.py``'s, against HF's."""

import json
import re
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.openai.adapters import CanaryQwenAdapter
from mstar.api_server.openai.protocol import TranscriptionRequest
from mstar.model.canary_qwen.canary_qwen_model import CanaryQwenModel, merge_lora
from mstar.model.canary_qwen.config import CanaryQwenModelConfig
from mstar.model.canary_qwen.submodules import CanaryQwenLLMSubmodule
from mstar.model.components.fastconformer import FastConformerConfig, FastConformerEncoder, nemo_to_hf_name


def test_merge_lora():
    torch.manual_seed(0)
    base, a, b = torch.randn(6, 4), torch.randn(2, 4), torch.randn(6, 2)
    weights = [
        ("layers.0.self_attn.q_proj.base_layer.weight", base),
        ("layers.0.self_attn.k_proj.weight", torch.ones(3, 4)),
        ("layers.0.self_attn.q_proj.lora_A.default.weight", a),
        ("layers.0.self_attn.q_proj.lora_B.default.weight", b),
        ("norm.weight", torch.ones(4)),
    ]
    out = dict(merge_lora(weights, scale=2.0))
    assert set(out) == {"layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight", "norm.weight"}
    assert torch.allclose(out["layers.0.self_attn.q_proj.weight"], base + 2.0 * b @ a)
    with pytest.raises(KeyError):
        list(merge_lora(weights[:3], scale=2.0))


def test_nemo_names_map_onto_hf_names():
    assert nemo_to_hf_name("pre_encode.conv.5.weight") == "subsampling.layers.5.weight"
    assert nemo_to_hf_name("pre_encode.out.bias") == "subsampling.linear.bias"
    assert nemo_to_hf_name("layers.3.self_attn.linear_pos.weight") == "layers.3.self_attn.relative_k_proj.weight"
    assert nemo_to_hf_name("layers.3.self_attn.pos_bias_u") == "layers.3.self_attn.bias_u"
    assert nemo_to_hf_name("layers.3.self_attn.linear_out.bias") == "layers.3.self_attn.o_proj.bias"
    assert nemo_to_hf_name("layers.0.conv.batch_norm.running_var") == "layers.0.conv.norm.running_var"
    assert nemo_to_hf_name("layers.0.feed_forward1.linear1.weight") == "layers.0.feed_forward1.linear1.weight"


def test_nemo_named_encoder_loads_completely():
    """Every parameter of the encoder is reachable from NeMo's names."""
    hf_to_nemo = [(hf, nemo) for nemo, hf in [
        ("pre_encode.conv.", "subsampling.layers."), ("pre_encode.out.", "subsampling.linear."),
        (".self_attn.linear_q.", ".self_attn.q_proj."), (".self_attn.linear_k.", ".self_attn.k_proj."),
        (".self_attn.linear_v.", ".self_attn.v_proj."), (".self_attn.linear_out.", ".self_attn.o_proj."),
        (".self_attn.linear_pos.", ".self_attn.relative_k_proj."), (".self_attn.pos_bias_u", ".self_attn.bias_u"),
        (".self_attn.pos_bias_v", ".self_attn.bias_v"), (".conv.batch_norm.", ".conv.norm."),
    ]]
    cfg = FastConformerConfig(hidden_size=16, num_layers=1, num_heads=2, intermediate_size=32, num_mel_bins=16,
                              subsampling_channels=4)
    enc = FastConformerEncoder(cfg)
    nemo = {}
    for name, p in enc.named_parameters():
        if name.endswith("qkv_proj.weight") or name.endswith("qkv_proj.bias"):
            for i, s in enumerate("qkv"):
                nemo[name.replace("qkv_proj", f"linear_{s}")] = p.detach().chunk(3)[i]
            continue
        if ".conv.pointwise_conv" in name and name.endswith("weight"):
            nemo[name] = p.detach()[..., None]
            continue
        if ".conv.depthwise_conv" in name:
            nemo[name] = p.detach()
            continue
        for hf, n in hf_to_nemo:
            name = name.replace(hf, n)
        nemo[name] = p.detach()
    for i in range(cfg.num_layers):
        for stat, value in (("weight", 1.0), ("bias", 0.0), ("running_mean", 0.0), ("running_var", 1.0)):
            nemo[f"layers.{i}.conv.batch_norm.{stat}"] = torch.full((16,), value)
    fresh = FastConformerEncoder(cfg)
    loaded = fresh.load_hf_weights((nemo_to_hf_name(k), v) for k, v in nemo.items())
    assert loaded == {n for n, _ in fresh.named_parameters()}


def test_config(tmp_path):
    nemo = {
        "pretrained_llm": "Qwen/Qwen3-1.7B", "audio_locator_tag": "<|audioplaceholder|>",
        "lora": {"r": 128, "lora_alpha": 256},
        "perception": {"encoder": {
            "d_model": 1024, "n_layers": 32, "n_heads": 8, "ff_expansion_factor": 4, "conv_kernel_size": 9,
            "feat_in": 128, "subsampling_factor": 8, "subsampling_conv_channels": 256, "xscaling": False,
            "self_attention_model": "rel_pos", "subsampling": "dw_striding", "att_context_size": [-1, -1],
        }, "preprocessor": {
            "sample_rate": 16000, "features": 128, "n_fft": 512, "window_size": 0.025, "window_stride": 0.01,
            "window": "hann", "normalize": "per_feature", "log": True, "frame_splicing": 1, "pad_to": 0,
            "dither": 1e-05,
        }},
    }
    qwen = {"hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 128, "intermediate_size": 6144, "vocab_size": 151936, "rope_theta": 1000000,
            "tie_word_embeddings": True}
    (tmp_path / "nemo").mkdir()
    (tmp_path / "qwen").mkdir()
    (tmp_path / "nemo" / "config.json").write_text(json.dumps(nemo))
    (tmp_path / "qwen" / "config.json").write_text(json.dumps(qwen))
    cfg = CanaryQwenModelConfig.from_pretrained(tmp_path / "nemo", tmp_path / "qwen")
    assert cfg.lora_scale == 2.0 and cfg.llm_repo == "Qwen/Qwen3-1.7B"
    assert cfg.encoder.num_layers == 32 and cfg.encoder.intermediate_size == 4096 and not cfg.encoder.causal
    assert cfg.llm.tie_word_embeddings and cfg.llm.num_key_value_heads == 8
    assert not cfg.encoder.scale_input
    # NeMo's default when the key is absent
    del nemo["perception"]["encoder"]["xscaling"]
    (tmp_path / "nemo" / "config.json").write_text(json.dumps(nemo))
    assert CanaryQwenModelConfig.from_pretrained(tmp_path / "nemo", tmp_path / "qwen").encoder.scale_input
    for section, key, value in (("encoder", "att_context_size", [70, 13]), ("preprocessor", "window_size", 0.02),
                                ("preprocessor", "normalize", "all_features"), ("preprocessor", "features", 80)):
        bad = json.loads(json.dumps(nemo))
        bad["perception"][section][key] = value
        (tmp_path / "nemo" / "config.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            CanaryQwenModelConfig.from_pretrained(tmp_path / "nemo", tmp_path / "qwen")


def test_adapter():
    adapter = CanaryQwenAdapter()
    # OpenAI's prompt is context (or the last window's text), not an instruction
    args = adapter.transcription_to_request(TranscriptionRequest(model="canary_qwen", prompt="ZyntriQix"), "a")
    assert args.text == "" and "initial_prompt" not in args.model_kwargs
    assert adapter.transcription_to_request(TranscriptionRequest(model="canary_qwen", language="en-US"), "")
    for bad in (TranscriptionRequest(model="canary_qwen", language="de"),
                TranscriptionRequest(model="canary_qwen", task="translate"),
                TranscriptionRequest(model="canary_qwen", timestamp_granularities=["segment"])):
        with pytest.raises(ValueError):
            adapter.transcription_to_request(bad, "")


def _model_with_word_tokenizer() -> CanaryQwenModel:
    """``prompt_ids`` over a stand-in tokenizer: one id per special token,
    word or whitespace run."""
    vocab: dict[str, int] = {}

    def encode(text, add_special_tokens=False):
        return [vocab.setdefault(piece, len(vocab)) for piece in re.findall(r"<\|\w+\|>|\w+|[^\w\s]|\s+", text)]

    model = object.__new__(CanaryQwenModel)
    model.config = SimpleNamespace(audio_locator_tag="<|audio|>")
    model.tokenizer = SimpleNamespace(encode=encode)
    model.audio_token_id = encode("<|audio|>")[0]
    return model


def test_prompt_ids():
    model = _model_with_word_tokenizer()
    ids = model.prompt_ids(3)
    at = ids.index(model.audio_token_id)
    assert ids[at:at + 3] == [model.audio_token_id] * 3 and ids.count(model.audio_token_id) == 3
    assert model.tokenizer.encode("<|im_start|>user\nTranscribe the following: ") == ids[:at]
    assert model.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n") == ids[at + 3:]
    for bad in ("say <|audio|> twice", "word " * CanaryQwenModel.MAX_TEXT_TOKENS):
        with pytest.raises(ValueError):
            model.prompt_ids(3, bad)
    # /generate's text (the ASR harnesses send a generic one) leaves the
    # trained instruction alone; model_kwargs' ``instruction`` replaces it
    model.config.encoder = SimpleNamespace(output_lengths=lambda n: 3)
    tensors = {"audio_inputs": [torch.zeros(16000)]}
    generic = "Transcribe the speech in this audio clip."

    def ids(**kwargs):
        out = model.process_prompt(generic, ["audio", "text"], ["text"], tensors=tensors, **kwargs)
        # the prompt holds the clip's frames, so it's laid out as rows (never keyed for prefix reuse)
        span, = out.metadata["kv_layout"]["kv_cache"]["main"].spans
        assert span.kind == "rows" and span.length == len(out.new_input_tensors["text_inputs"][0])
        return out.new_input_tensors["text_inputs"][0]

    assert ids().tolist() == model.prompt_ids(3)
    assert ids(instruction="Write down what is said:").tolist() == model.prompt_ids(3, "Write down what is said:")


def test_output_and_request_limits():
    model = object.__new__(CanaryQwenModel)
    model.config = SimpleNamespace(llm=SimpleNamespace(
        num_hidden_layers=1, num_key_value_heads=1, head_dim=8, num_attention_heads=1, rope_theta=1e6, vocab_size=16,
    ))
    limit = CanaryQwenModel.MAX_OUTPUT_TOKENS
    assert model.get_max_output_tokens_limit() == limit
    assert model.get_max_output_tokens(max_output_tokens=None) == limit
    assert model.get_max_output_tokens(max_output_tokens=5) == 5
    # the request's max_output_tokens caps the loop, not the graph's max_iters
    assert model.get_loop_caps(max_output_tokens=5) == {"decode_loop": 5}
    # the shipped pool holds 64 requests at MAX_SEQ_LEN and the sink, no more
    model.validate_config_yaml({"max_concurrent_requests": 64}, "canary_qwen.yaml")
    with pytest.raises(ValueError, match=r"resources\.kv_cache"):
        model.validate_config_yaml({"max_concurrent_requests": 65}, "canary_qwen.yaml")


def test_splice_puts_frame_k_on_placeholder_k():
    node = object.__new__(CanaryQwenLLMSubmodule)
    torch.nn.Module.__init__(node)
    node.model = SimpleNamespace(embed=torch.nn.Embedding(10, 4))
    node.audio_token_id = 99
    ids = torch.tensor([1, 2, 99, 99, 99, 3])
    audio = torch.randn(3, 4)
    with torch.no_grad():
        out = node.splice_audio(ids, audio)
        text = node.model.embed(torch.tensor([1, 2, 3]))
    assert torch.equal(out[2:5], audio)
    assert torch.equal(out[[0, 1, 5]], text)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync detection needs a GPU")
def test_splice_does_not_sync_the_gpu_thread():
    """It runs in prepare_inputs, on the GPU thread, ahead of the step in
    flight: a host read there stalls the pipeline."""
    node = object.__new__(CanaryQwenLLMSubmodule)
    torch.nn.Module.__init__(node)
    node.model = SimpleNamespace(embed=torch.nn.Embedding(10, 4).cuda())
    node.audio_token_id = 99
    ids = torch.tensor([1, 2, 99, 99, 99, 3], device="cuda")
    audio = torch.randn(3, 4, device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.no_grad():
            out = node.splice_audio(ids, audio)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(out[2:5], audio)


def test_llm_loads_from_the_checkpoints_names(tmp_path):
    """NeMo keeps the token embedding outside the LLM and the q/v LoRA
    adapters unmerged under PEFT names; through ``_llm_weights`` the LLM gets
    exactly the parameters the merged HF names would give it."""
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import save_file

    from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig

    tiny = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                intermediate_size=48, vocab_size=97, tie_word_embeddings=True)
    torch.manual_seed(0)
    hf = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**tiny)).eval()
    merged, nemo = {}, {}
    for name, tensor in hf.model.state_dict().items():
        merged[name] = tensor
        if name == "embed_tokens.weight":
            nemo[name] = tensor
        elif re.search(r"self_attn\.[qv]_proj\.weight$", name):
            stem = "llm.base_model.model.model." + name.removesuffix(".weight")
            a, b = torch.randn(2, tensor.shape[1]), torch.randn(tensor.shape[0], 2)
            nemo |= {f"{stem}.base_layer.weight": tensor, f"{stem}.lora_A.default.weight": a,
                     f"{stem}.lora_B.default.weight": b}
            merged[name] = tensor + 2.0 * b @ a
        else:
            nemo[f"llm.base_model.model.model.{name}"] = tensor
    save_file({k: v.contiguous() for k, v in nemo.items()}, str(tmp_path / "model.safetensors"))

    model = object.__new__(CanaryQwenModel)
    model.local_dir, model.config = tmp_path, SimpleNamespace(lora_scale=2.0)
    config = Qwen3LMConfig.from_hf(tiny)
    from_checkpoint = Qwen3DenseLM(config, attn_key="attn", kv_key="kv", pos_key="rope")
    from_checkpoint.load_weights(model._llm_weights("cpu"))
    from_merged = Qwen3DenseLM(config, attn_key="attn", kv_key="kv", pos_key="rope")
    from_merged.load_weights(merged.items())
    expected = dict(from_merged.named_parameters())
    for name, param in from_checkpoint.named_parameters():
        assert torch.allclose(param, expected[name], atol=1e-6), name


def test_llm_node_caps_captures_and_names():
    """What the shared TokenStepNode gives the LLM node: prefill batches
    capped, decode bounded by its captures, both stop tokens, and the walk,
    loop and sampler names of Canary-Qwen's graph."""
    from mstar.model.canary_qwen.config import SAMPLER
    from mstar.model.canary_qwen.submodules import CanaryQwenLLMSubmodule
    from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig

    config = Qwen3LMConfig.from_hf(dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=1, head_dim=8, intermediate_size=32, vocab_size=20))
    llm = Qwen3DenseLM(config, attn_key="attn", kv_key="kv", pos_key="rope")
    node = CanaryQwenLLMSubmodule(llm, audio_token_id=19, stop_token_ids={11, 12})
    assert node.max_batch_size("prefill") == 16 and node.max_batch_size("decode") is None
    (capture,) = node.get_accelerator_graph_configs(torch.device("cpu"))
    assert capture.capture_graph_walk == "decode" and capture.capture_batch_sizes == [1, 2, 4, 8, 16, 32, 64]
    info = SimpleNamespace(
        resource_configs={SAMPLER: SimpleNamespace(ignore_eos=False)}, dynamic_loop_iter_counts={}, max_tokens=100,
    )
    assert [node.check_stop("a", info, {"new_token": [torch.tensor([t])]}) for t in (11, 12, 3)] == [
        {"decode_loop"}, {"decode_loop"}, set(),
    ]
    walks = object.__new__(CanaryQwenModel).get_graph_walk_graphs()
    assert walks[node.DECODE_WALK].name == node.DECODE_LOOP and node.SAMPLER == SAMPLER


def _qwen_tokenizer():
    """A tiny byte-level BPE with Qwen3's end-of-turn tokens, as the base LLM's repo ships them."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["the cat sat on the mat"], trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ))
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", clean_up_tokenization_spaces=False,
    )


def test_a_transcript_passes_the_data_worker_with_its_stop_ids(tmp_path, monkeypatch):
    """The data worker drops the stop ids Canary-Qwen's ``text_output`` declares from the
    transcript, and tags the chunk that ends it."""
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    import queue

    from mstar.api_server.data_worker import InlineTokens, PreprocessWorkerThread, RequestOutputState

    tokenizer = _qwen_tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(
        CanaryQwenModelConfig, "from_pretrained", lambda *a: SimpleNamespace(audio_locator_tag="<|audioplaceholder|>"),
    )
    (tmp_path / "llm").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"pretrained_llm": str(tmp_path / "llm")}))
    model = CanaryQwenModel(str(tmp_path))

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.out_queue, wt.model, wt.request_model_kwargs = queue.Queue(), model, {}
    wt._draining_rids, wt._reads_done_sent, wt.in_flight_requests = set(), set(), set()
    wt.request_output_state = {"r": RequestOutputState()}
    tokens = [*tokenizer.encode(" the cat", add_special_tokens=False), tokenizer.convert_tokens_to_ids("<|im_end|>")]
    for token in tokens:  # one frame per decode step
        wt._emit_inline_tokens(InlineTokens(["r"], [token], "text"))
    chunks = [wt.out_queue.get_nowait() for _ in tokens]
    assert [c.modality for c in chunks] == ["text"] * len(tokens)
    assert b"".join(c.data for c in chunks) == b" the cat"
    assert [c.metadata["stop_token"] for c in chunks] == [False] * (len(tokens) - 1) + [True]
