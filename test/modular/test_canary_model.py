"""Canary (canary-1b-v2) on a tiny fake snapshot: the task prompt, what
requests it refuses, its graph, SentencePiece detokenization and the OpenAI
adapter; the decoder against HF's on CPU, with the engine's caches stood in
for by dense fakes; and the decoder node's context spans and stop checks. The
encoder's parity is in test_fastconformer.py."""

import json

import pytest
import torch

from mstar.api_server.openai.adapters import CanaryAdapter
from mstar.api_server.openai.protocol import TranscriptionRequest
from mstar.model.canary.canary_model import CanaryModel
from mstar.model.canary.config import LANGUAGES, PROMPT_LENGTH, CanaryModelConfig, check_task
from mstar.model.canary.decoder import CanaryDecoderModel, sinusoid_positions

SPECIALS = [
    "<unk>", "<|nospeech|>", "<pad>", "<|endoftext|>", "<|startoftranscript|>", "<|pnc|>", "<|nopnc|>",
    "<|startofcontext|>", "<|itn|>", "<|noitn|>", "<|timestamp|>", "<|notimestamp|>", "<|diarize|>",
    "<|nodiarize|>", "<|emo:undefined|>", *(f"<|{lang}|>" for lang in LANGUAGES),
]
PIECES = ["▁he", "llo", "▁caf", "<0xC3>", "<0xA9>", "!"]


@pytest.fixture
def snapshot(tmp_path):
    vocab = {tok: i for i, tok in enumerate(SPECIALS + PIECES)}
    tokenizer = {
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "byte_fallback": True},
        "added_tokens": [{"id": vocab[t], "content": t, "special": True} for t in SPECIALS],
    }
    encoder = dict(
        model_type="parakeet_encoder", hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=64, conv_kernel_size=9, num_mel_bins=128, subsampling_factor=8,
        subsampling_conv_channels=8, subsampling_conv_kernel_size=3, subsampling_conv_stride=2,
        attention_bias=True, convolution_bias=True, scale_input=False,
    )
    decoder = dict(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
        vocab_size=len(vocab), max_position_embeddings=64, hidden_act="relu",
    )
    config = {"model_type": "canary", "eos_token_id": 3, "encoder_config": encoder, "decoder_config": decoder}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer))
    return tmp_path


def test_prompt_tokens(snapshot):
    """The tokens ``chat_template.jinja`` renders for a transcription (and a
    translation), in its order."""
    cfg = CanaryModelConfig.from_pretrained(snapshot)
    ids = cfg.special_ids
    assert cfg.prompt_ids("en") == [ids[t] for t in (
        "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", "<|en|>", "<|en|>",
        "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
    )]
    translate = cfg.prompt_ids("DE", "en", punctuation=False)
    assert translate[3:6] == [ids["<|de|>"], ids["<|en|>"], ids["<|nopnc|>"]]
    # the output limit counts on it
    assert len(translate) == PROMPT_LENGTH


@pytest.mark.parametrize("source,target", [("xx", "en"), ("en", "zh"), ("de", "fr")])
def test_refuses_what_canary_cannot_do(source, target):
    with pytest.raises(ValueError):
        check_task(source, target)
    assert check_task("FR", "en") == ("fr", "en")
    assert check_task("en-US", "de_DE") == ("en", "de")


@pytest.mark.parametrize("bad", [1, True, ["de"]])
def test_non_string_language_is_refused(bad):
    with pytest.raises(ValueError):
        check_task("en", bad)


def test_process_prompt_and_its_limits(snapshot):
    model = CanaryModel(str(snapshot))
    out = model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(16_000)]},
                               language="de", target_language="en")
    assert out.new_input_tensors["audio"][0].shape == (16_000,)
    assert out.new_input_tensors["text_inputs"][0].tolist() == model.config.prompt_ids("de", "en")
    # the decoder writes the prompt, then a slot per step; the context is the clip's encoder frames
    layout = out.metadata["kv_layout"]
    assert set(layout) == set(model.kv_labels()) and all(set(layout[k]) == set(v) for k, v in model.kv_labels().items())
    assert layout["kv_cache"]["main"].spans[0].length == PROMPT_LENGTH
    assert layout["cross_kv_cache"]["main"].spans[0].length == model.config.encoder.output_lengths(100)
    # 40.96 s, the largest encoder graph, is the limit
    assert model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(655_360)]})
    for bad in (torch.zeros(655_361), torch.zeros(100)):
        with pytest.raises(ValueError):
            model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [bad]})
    with pytest.raises(ValueError):
        model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(16_000)]},
                             punctuation="yes")
    with pytest.raises(ValueError):
        model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": []})


def test_graph_and_limits(snapshot):
    model = CanaryModel(str(snapshot))
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "decode"}
    assert model.nodes == ["audio_encoder", "decoder"]
    # the position table bounds prompt + transcript
    assert model.get_max_output_tokens() == 64 - 9
    assert model.get_max_output_tokens(max_output_tokens=10) == 10
    assert model.get_max_output_tokens(max_output_tokens=None) == 64 - 9
    # published, so the conductor refuses a larger request instead of cutting it short
    assert model.get_max_output_tokens_limit() == 64 - 9
    # the request's max_output_tokens caps the loop, not the graph's max_iters
    assert model.get_loop_caps(max_output_tokens=10) == {"decode_loop": 10}


def test_request_cap_must_fit_the_kv_pools(snapshot):
    model = CanaryModel(str(snapshot))
    # the default caches hold MAX_CONCURRENT_REQUESTS at full length: one
    # self-attention page per request here, and 4 context pages per 40.96 s clip
    cap = CanaryModel.MAX_CONCURRENT_REQUESTS
    model.validate_config_yaml({"max_concurrent_requests": cap}, "canary.yaml")
    with pytest.raises(ValueError, match=r"resources\.kv_cache "):
        model.validate_config_yaml({"max_concurrent_requests": cap + 1}, "canary.yaml")
    roomy = {"resources": {"kv_cache": {"max_num_pages": 4096}}}
    with pytest.raises(ValueError, match=r"resources\.cross_kv_cache"):
        model.validate_config_yaml({"max_concurrent_requests": cap + 1, **roomy}, "canary.yaml")


def test_detokenizes_pieces_and_byte_fallback(snapshot):
    model = CanaryModel(str(snapshot))
    vocab = {tok: i for i, tok in enumerate(SPECIALS + PIECES)}
    ids = [vocab[t] for t in ("<|en|>", "▁he", "llo", "▁caf", "<0xC3>", "<0xA9>", "!", "<|endoftext|>")]
    per_token = b"".join(model.postprocess(torch.tensor([i]), "text") for i in ids)
    assert per_token.decode("utf-8") == " hello café!"


def test_decoder_weight_names():
    assert CanaryDecoderModel.remap("model.decoder.layers.3.encoder_attn.o_proj.weight") == \
        "layers.3.encoder_attn.out_proj.weight"
    assert CanaryDecoderModel.remap("model.decoder.layers.0.self_attn.q_proj.bias") == "layers.0.self_attn.q_proj.bias"
    assert CanaryDecoderModel.remap("proj_out.bias") == "proj_out.bias"
    assert CanaryDecoderModel.remap("model.encoder.layers.0.conv.norm.weight") is None


def test_decoder_positions_match_hf():
    canary = pytest.importorskip("transformers.models.canary.modeling_canary", reason="transformers without Canary")
    pos = torch.arange(64)
    assert torch.allclose(sinusoid_positions(pos, 32), canary.CanaryPositionalEmbedding(64, 32)(pos), atol=1e-6)


def test_adapter_maps_and_refuses():
    adapter = CanaryAdapter()
    req = TranscriptionRequest(model="canary", language="DE", prompt="ignored", target_language="en")
    args = adapter.transcription_to_request(req, "/tmp/a.wav")
    assert args.model_kwargs["language"] == "de" and args.model_kwargs["target_language"] == "en"
    assert "initial_prompt" not in args.model_kwargs and args.file_paths == {"audio": ["/tmp/a.wav"]}
    assert adapter.transcription_to_request(TranscriptionRequest(model="canary"), "").model_kwargs["language"] == "en"
    # Whisper's task=translate is into English
    req = TranscriptionRequest(model="canary", language="de-AT", task="translate")
    mk = adapter.transcription_to_request(req, "").model_kwargs
    assert (mk["language"], mk["target_language"]) == ("de", "en") and "task" not in mk
    # refused before the upload is read, so a stream never opens on a bad field
    for bad in (
        TranscriptionRequest(model="canary", language="zh"),
        TranscriptionRequest(model="canary", language="de", target_language="fr"),
        TranscriptionRequest(model="canary", target_language=1),
        TranscriptionRequest(model="canary", task="summarize"),
        TranscriptionRequest(model="canary", punctuation="maybe"),
        TranscriptionRequest(model="canary", timestamp_granularities=["word"]),
    ):
        with pytest.raises(ValueError):
            adapter.transcription_to_request(bad, "")
    # a timestamped format falls back to one whole-clip segment, not an error
    mk = adapter.transcription_to_request(TranscriptionRequest(model="canary", response_format="verbose_json"), "")
    assert "timestamps" not in mk.model_kwargs


class _FakeSelfAttention:
    """Dense stand-in for the paged KV + attention pair, one request: the K/V
    written so far are its pages, and causal SDPA is the kernel."""

    requires_kv_write = True

    def __init__(self):
        self._label, self._layer = "main", 0
        self._kv: dict[int, tuple[list, list]] = {}
        self._pending = None

    @property
    def default_label(self):
        return self._label

    def set_default_label(self, label):
        self._label = label

    def set_default_layer_idx(self, idx):
        self._layer = idx

    def layer_view(self):
        return self._layer

    def write_kv(self, k, v):
        self._pending = (k, v)

    def run(self, q, kv_cache_layer, k=None, v=None):
        ks, vs = self._kv.setdefault(kv_cache_layer, ([], []))
        ks.append(self._pending[0])
        vs.append(self._pending[1])
        k_all, v_all = torch.cat(ks), torch.cat(vs)
        mask = torch.ones(q.shape[0], k_all.shape[0], dtype=torch.bool).tril(diagonal=k_all.shape[0] - q.shape[0])
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k_all.transpose(0, 1), v_all.transpose(0, 1), attn_mask=mask,
        )
        return out.transpose(0, 1)


class _FakeContext:
    """The cross-attention context cache: the clip's K/V per layer, written once."""

    def __init__(self):
        self._layer, self.kv = 0, {}

    def set_default_layer_idx(self, idx):
        self._layer = idx

    def layer_view(self):
        return self._layer

    def write_kv(self, k, v, label):
        self.kv[self._layer] = (k, v)


class _FakeCross:
    """Cross-attention over the whole context, no mask."""

    def __init__(self, context: _FakeContext):
        self.context = context

    def set_default_label(self, label):
        pass

    def run(self, q, kv_cache_layer):
        k, v = self.context.kv[kv_cache_layer]
        out = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return out.transpose(0, 1)


def test_decoder_matches_hf():
    """Prefill of a 9-token prompt over a clip's encoder output, then two
    decode steps: the decoder's hidden states against HF's ``CanaryDecoder``
    at every position, loaded through the serving loader."""
    transformers = pytest.importorskip("transformers")
    modeling = pytest.importorskip("transformers.models.canary.modeling_canary")
    from mstar.model.canary.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE
    from mstar.model.components.fastconformer import FastConformerConfig
    from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights
    from mstar.model.utils import check_loaded

    hf_config = transformers.CanaryDecoderConfig(
        vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, head_dim=8, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    reference = modeling.CanaryDecoder(hf_config).eval()
    config = CanaryModelConfig(
        encoder=FastConformerConfig(), hidden_size=32, num_layers=2, num_heads=4, intermediate_size=64,
        vocab_size=50, max_position_embeddings=64,
    )
    ours = CanaryDecoderModel(config).eval()
    weights = [(f"model.decoder.{name}", t) for name, t in reference.state_dict().items()]
    weights += [("proj_out.weight", torch.randn(50, 32)), ("proj_out.bias", torch.randn(50))]
    check_loaded(ours, load_hf_weights(
        ours, weights, stacked_params=WHISPER_STACKED_PARAMS, name_remapper=CanaryDecoderModel.remap,
    ), "decoder")
    self_attn, context = _FakeSelfAttention(), _FakeContext()
    for module in ours.modules():
        if hasattr(module, "bind_resources") and module is not ours:
            module.bind_resources({ATTN: self_attn, KV_CACHE: self_attn, CROSS_ATTN: _FakeCross(context),
                                   CROSS_KV_CACHE: context})

    torch.manual_seed(1)
    encoder_states = torch.randn(13, 32)
    tokens = torch.randint(0, 50, (9,))
    with torch.no_grad():
        ours.write_cross_kv(encoder_states)
        hidden = ours(ours.embed(tokens, torch.arange(9)), label="main")
        expected = reference(tokens[None], position_ids=torch.arange(9)[None],
                             encoder_hidden_states=encoder_states[None]).last_hidden_state[0]
        assert torch.allclose(hidden, expected, atol=1e-4), (hidden - expected).abs().max()
        for step in range(2):
            nxt = torch.randint(0, 50, (1,))
            tokens = torch.cat([tokens, nxt])
            position = tokens.numel() - 1
            hidden = ours(ours.embed(nxt, torch.tensor([position])), label="main")
            expected = reference(tokens[None], position_ids=torch.arange(position + 1)[None],
                                 encoder_hidden_states=encoder_states[None]).last_hidden_state[0, -1:]
            assert torch.allclose(hidden, expected, atol=1e-4), (step, (hidden - expected).abs().max())


def _tiny_node():
    from mstar.model.canary.submodules import CanaryDecoderSubmodule
    from mstar.model.components.fastconformer import FastConformerConfig

    config = CanaryModelConfig(
        encoder=FastConformerConfig(), hidden_size=32, num_layers=1, num_heads=4, intermediate_size=64,
        vocab_size=50, max_position_embeddings=64, eos_token_id=3,
    )
    return CanaryDecoderSubmodule(CanaryDecoderModel(config), config)


def test_prefill_reserves_each_clips_context_once():
    """The cross-attention context is written at prefill: each request's
    segment spans its encoder frames, in the order the frames are concatenated
    for the write; on decode the span is zero."""
    from mstar.model.canary.config import CROSS_ATTN, CROSS_KV_CACHE
    from mstar.model.submodule_base import ARNodeInputs

    node = _tiny_node()
    clips = [torch.randn(13, 32), torch.randn(7, 32)]
    prefill = [ARNodeInputs(input_ids=torch.arange(9), input_seq_len=9, tensor_inputs={"encoder_states": c})
               for c in clips]
    step = node.declare_step("prefill", ["a", "b"], prefill)
    for key in (CROSS_KV_CACHE, CROSS_ATTN):
        assert [(s.request_id, s.span) for s in step.steps[key].segments] == [("a", 13), ("b", 7)]
    assert [s.span for s in step.segments] == [9, 9]
    assert torch.equal(node.preprocess("prefill", None, prefill)["encoder_states"], torch.cat(clips))
    decode = [ARNodeInputs(input_ids=torch.tensor([5]), input_seq_len=1) for _ in clips]
    step = node.declare_step("decode", ["a", "b"], decode)
    assert [s.span for s in step.steps[CROSS_KV_CACHE].segments] == [0, 0]
    assert "encoder_states" not in node.preprocess("decode", None, decode)


def test_batched_stop_check_agrees_with_the_per_request_one():
    """Rows come in the order the forward ran the requests, which needn't be
    the order the worker asks in; EOS, ignore_eos and the token limit."""
    from types import SimpleNamespace

    from mstar.model.canary.config import SAMPLER
    from mstar.model.submodule_base import HostRows

    node = _tiny_node()

    def info(ignore_eos=False, decoded=0, max_tokens=100):
        return SimpleNamespace(resource_configs={SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
                               dynamic_loop_iter_counts={"decode_loop": decoded}, max_tokens=max_tokens)

    infos = {"eos": info(), "ignored": info(ignore_eos=True), "limit": info(decoded=9, max_tokens=10), "go": info()}
    rows = ("go", "limit", "eos", "ignored")
    tokens = torch.tensor([[7], [7], [3], [3]])
    asked = ["eos", "ignored", "limit", "go"]
    # called as the engine calls it, with the step's walk
    batched = node.check_stop_batched(
        asked, infos, HostRows(request_ids=rows, buffers={"new_token": tokens}), step_walk="decode",
    )
    one_by_one = {rid: node.check_stop(rid, infos[rid], {"new_token": [tokens[rows.index(rid)]]}) for rid in asked}
    assert batched == {rid: stop for rid, stop in one_by_one.items() if stop} == {
        "eos": {"decode_loop"}, "limit": {"decode_loop"},
    }


def test_decoder_node_caps_captures_and_names(snapshot):
    """What the shared TokenStepNode gives the decoder node: prefill batches
    capped, decode bounded by its captures, the sampled token fed back, and
    the walk, loop and sampler names of Canary's graph."""
    from mstar.model.canary.config import SAMPLER

    node = _tiny_node()
    assert node.max_batch_size("prefill") == 32 and node.max_batch_size("decode") is None
    (capture,) = node.get_accelerator_graph_configs(torch.device("cpu"))
    assert capture.capture_graph_walk == "decode" and capture.capture_batch_sizes == [1, 2, 4, 8, 16, 32, 64]
    outputs = {"new_token": [torch.tensor([5])]}
    node.postprocess("a", None, outputs)
    assert outputs["text_inputs"] is outputs["new_token"]
    walks = CanaryModel(str(snapshot)).get_graph_walk_graphs()
    assert walks[node.DECODE_WALK].name == node.DECODE_LOOP and node.SAMPLER == SAMPLER
