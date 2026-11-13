"""Canary (canary-1b-v2) on a tiny fake snapshot: the task prompt, what
requests it refuses, its graph, SentencePiece detokenization and the OpenAI
adapter; the decoder against HF's on CPU, with the engine's caches stood in
for by dense fakes; and the decoder node's context spans and stop checks. The
encoder's parity is in test_fastconformer.py."""

import json
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.openai.adapters import CanaryAdapter
from mstar.api_server.openai.protocol import TranscriptionRequest
from mstar.model.canary.canary_model import CanaryModel
from mstar.model.canary.config import LANGUAGES, MEL_SETTINGS, PROMPT_LENGTH, CanaryModelConfig, check_task
from mstar.model.canary.decoder import CanaryDecoderModel, sinusoid_positions

SPECIALS = [
    "<unk>", "<|nospeech|>", "<pad>", "<|endoftext|>", "<|startoftranscript|>", "<|pnc|>", "<|nopnc|>",
    "<|startofcontext|>", "<|itn|>", "<|noitn|>", "<|timestamp|>", "<|notimestamp|>", "<|diarize|>",
    "<|nodiarize|>", "<|emo:undefined|>", *(f"<|{lang}|>" for lang in LANGUAGES),
]
PIECES = ["▁", "▁he", "llo", "▁caf", "<0xC3>", "<0xA9>", "!"]


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
    mel = {**MEL_SETTINGS, "feature_extractor_type": "ParakeetFeatureExtractor", "feature_size": 128}
    (tmp_path / "processor_config.json").write_text(json.dumps({"feature_extractor": mel}))
    return tmp_path


def test_refuses_a_mel_front_end_it_does_not_compute(snapshot):
    path = snapshot / "processor_config.json"
    processor = json.loads(path.read_text())
    processor["feature_extractor"]["hop_length"] = 320
    path.write_text(json.dumps(processor))
    with pytest.raises(ValueError, match="hop_length"):
        CanaryModelConfig.from_pretrained(snapshot)


def test_prompt_tokens(snapshot):
    """The tokens NeMo's transcribe feeds the decoder for a transcription (and a
    translation), in its order: SentencePiece's ``▁``, then the task tokens."""
    cfg = CanaryModelConfig.from_pretrained(snapshot)
    ids = cfg.special_ids
    assert cfg.prompt_ids("en") == [cfg.word_boundary_id] + [ids[t] for t in (
        "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", "<|en|>", "<|en|>",
        "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
    )]
    assert cfg.word_boundary_id == len(SPECIALS)  # the vocab's "▁", not an added token
    translate = cfg.prompt_ids("DE", "en", punctuation=False)
    assert translate[4:7] == [ids["<|de|>"], ids["<|en|>"], ids["<|nopnc|>"]]
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
    layout = out.metadata["state_layout"]
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
    assert model.get_max_output_tokens() == 64 - PROMPT_LENGTH
    assert model.get_max_output_tokens(max_output_tokens=10) == 10
    assert model.get_max_output_tokens(max_output_tokens=None) == 64 - PROMPT_LENGTH
    # published, so the conductor refuses a larger request instead of cutting it short
    assert model.get_max_output_tokens_limit() == 64 - PROMPT_LENGTH


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
    assert b"".join(model.postprocess_inline(ids, "text", [None] * len(ids))) == per_token


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
    """Dense stand-in for the self-attention array's handle, one request: the
    K/V added so far are its pages, and causal SDPA is the kernel. It hands the
    node the step's position ids, and keeps the hidden states it picks each
    row's last token from."""

    label = "main"

    def __init__(self):
        self._kv: dict[int, tuple[list, list]] = {}
        self._pending = None
        self.positions = None
        self.seen = None

    def _check_open(self, call):
        pass

    def at_last_token(self, hidden, *, label=None):
        self.seen = hidden
        return hidden[-1:]

    def _add_read(self, layer, q, *, keep, k, v):
        assert keep is None, "this fake keeps every add"
        self.add(layer, k=k, v=v)
        return self.read(layer, q)

    def add(self, layer, *, k, v):
        self._pending = (k, v)

    def read(self, layer, q):
        ks, vs = self._kv.setdefault(int(layer), ([], []))
        ks.append(self._pending[0])
        vs.append(self._pending[1])
        k_all, v_all = torch.cat(ks), torch.cat(vs)
        mask = torch.ones(q.shape[0], k_all.shape[0], dtype=torch.bool).tril(diagonal=k_all.shape[0] - q.shape[0])
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k_all.transpose(0, 1), v_all.transpose(0, 1), attn_mask=mask,
        )
        return out.transpose(0, 1)


class _FakeSampler:
    """The sampler's handle for one row, handing back the logits it's given."""

    _open, _drawn, rows = True, 0, ("a",)
    sampler = SimpleNamespace(draws=2 ** 31)
    resource = SimpleNamespace(sample=lambda rows, logits: logits)


class _FakeContext:
    """The context array's handle: the clip's K/V per layer, added once and read
    whole with no mask."""

    def __init__(self):
        self.kv = {}

    def add(self, layer, *, k, v):
        self.kv[int(layer)] = (k, v)

    def read(self, layer, q):
        k, v = self.kv[int(layer)]
        out = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return out.transpose(0, 1)


def test_decoder_matches_hf():
    """Prefill of a 9-token prompt over a clip's encoder output, then two
    decode steps, through the decoder node's forward: the decoder's hidden
    states against HF's ``CanaryDecoder`` at every prefill position, and the
    logits it samples from at each step, loaded through the serving loader."""
    transformers = pytest.importorskip("transformers")
    modeling = pytest.importorskip("transformers.models.canary.modeling_canary")
    from mstar.model.canary.config import CROSS_KV_CACHE, KV_CACHE, SAMPLER
    from mstar.model.canary.submodules import CanaryDecoderSubmodule
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
    handles = {KV_CACHE: _FakeSelfAttention(), CROSS_KV_CACHE: _FakeContext(), SAMPLER: _FakeSampler()}
    for module in ours.modules():
        if hasattr(module, "bind_state"):
            module.bind_state(handles)
    node, kv = CanaryDecoderSubmodule(ours, config, max_context=13), handles[KV_CACHE]
    node.bind_state(handles)

    def run(walk, inputs, positions):
        """One step as the engine runs it, with a sampler that hands back its logits."""
        kv.positions = SimpleNamespace(pos_ids=lambda label: positions)
        engine_inputs = SimpleNamespace(request_ids=["a"])
        rows = [node.prepare_inputs(walk, None, inputs)]
        return node.forward_batched(walk, engine_inputs, **node.preprocess(walk, engine_inputs, rows))

    torch.manual_seed(1)
    encoder_states = torch.randn(13, 32)
    tokens = torch.randint(0, 50, (9,))
    with torch.no_grad():
        out = run("prefill", {"text_inputs": [tokens], "encoder_states": [encoder_states]}, torch.arange(9))
        expected = reference(tokens[None], position_ids=torch.arange(9)[None],
                             encoder_hidden_states=encoder_states[None]).last_hidden_state[0]
        assert torch.allclose(kv.seen, expected, atol=1e-4), (kv.seen - expected).abs().max()
        assert torch.allclose(out.row_outputs["new_token"], ours.logits(expected[-1:]), atol=1e-3)
        for step in range(2):
            nxt = torch.randint(0, 50, (1,))
            tokens = torch.cat([tokens, nxt])
            position = tokens.numel() - 1
            logits = run("decode", {"text_inputs": [nxt]}, torch.tensor([position])).row_outputs["new_token"]
            expected = reference(tokens[None], position_ids=torch.arange(position + 1)[None],
                                 encoder_hidden_states=encoder_states[None]).last_hidden_state[0, -1:]
            expected = ours.logits(expected)
            assert torch.allclose(logits, expected, atol=1e-3), (step, (logits - expected).abs().max())


def _tiny_node():
    from mstar.model.canary.submodules import CanaryDecoderSubmodule
    from mstar.model.components.fastconformer import FastConformerConfig

    config = CanaryModelConfig(
        encoder=FastConformerConfig(), hidden_size=32, num_layers=1, num_heads=4, intermediate_size=64,
        vocab_size=50, max_position_embeddings=64, eos_token_id=3,
    )
    return CanaryDecoderSubmodule(CanaryDecoderModel(config), config, max_context=13)


def test_prefill_reserves_each_clips_context_once():
    """The cross-attention context is added at prefill: each request's
    segment spans its encoder frames, in the order the frames are concatenated
    for the add; on decode the span is zero."""
    from mstar.model.canary.config import CROSS_KV_CACHE
    from mstar.model.submodule_base import ARNodeInputs

    node = _tiny_node()
    clips = [torch.randn(13, 32), torch.randn(7, 32)]
    prefill = [ARNodeInputs(input_ids=torch.arange(9), input_seq_len=9, tensor_inputs={"encoder_states": c})
               for c in clips]
    step = node.declare_step("prefill", ["a", "b"], prefill)
    assert [(s.request_id, s.span) for s in step.state[CROSS_KV_CACHE].segments] == [("a", 13), ("b", 7)]
    assert [s.span for s in step.segments] == [9, 9]
    # added in `preprocess`, outside the prefill's graph, which is sized by the prompt alone
    assert step.state[CROSS_KV_CACHE].eager_add
    written = []
    node.model.add_context = written.append
    assert "encoder_states" not in node.preprocess("prefill", None, prefill)
    assert len(written) == 1 and torch.equal(written[0], torch.cat(clips))
    decode = [ARNodeInputs(input_ids=torch.tensor([5]), input_seq_len=1) for _ in clips]
    step = node.declare_step("decode", ["a", "b"], decode)
    assert [s.span for s in step.state[CROSS_KV_CACHE].segments] == [0, 0]
    assert not step.state[CROSS_KV_CACHE].eager_add
    assert "encoder_states" not in node.preprocess("decode", None, decode) and len(written) == 1


def test_a_padded_prefill_replay_projects_only_the_real_rows_context():
    """A replay pads its rows to the bucket's batch size with rows of no
    tokens; they carry no context, so the eager context add projects the
    real clips' frames and nothing for the padding."""
    from collections import namedtuple
    from types import SimpleNamespace

    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
    from mstar.model.canary.config import CROSS_KV_CACHE
    from mstar.model.submodule_base import ARNodeInputs

    node = _tiny_node()
    _, prefill = node.get_accelerator_graph_configs(torch.device("cpu"))
    clips = [torch.randn(13, 32), torch.randn(7, 32)]
    real = [ARNodeInputs(input_ids=torch.arange(PROMPT_LENGTH), input_seq_len=PROMPT_LENGTH,
                         tensor_inputs={"encoder_states": c}) for c in clips]
    # the runner's own padding of a 2-row step to a 5-row bucket
    bucket = namedtuple("Bucket", "bs")(5)
    runner = SimpleNamespace(_buckets={bucket: SimpleNamespace(config=prefill)})
    rows = AcceleratorGraphRunner.pad_inputs(runner, SimpleNamespace(bucket=bucket), real)
    assert len(rows) == 5 and rows[:2] == real
    step = node.declare_step("prefill", ["a", "b", "p0", "p1", "p2"], rows)
    assert [s.span for s in step.state[CROSS_KV_CACHE].segments] == [13, 7, 0, 0, 0]
    written = []
    node.model.add_context = written.append
    node.preprocess("prefill", None, rows)
    assert len(written) == 1 and torch.equal(written[0], torch.cat(clips))


@pytest.mark.parametrize("width", [1, 3])
def test_batched_stop_check_agrees_with_the_per_request_one(width):
    """Rows come in the order the forward ran the requests, which needn't be
    the order the worker asks in; EOS, ignore_eos, a row at its token
    limit, which the loop's cap ends rather than the check, and the
    prefill's EOS, which ends the reply before the loop exists. Both read a
    row's first element, also from a buffer wider than one token."""
    from types import SimpleNamespace

    from mstar.conductor.request_info import FINAL_PREFILL
    from mstar.model.canary.config import SAMPLER
    from mstar.model.submodule_base import END_PARTITION, HostRows

    node = _tiny_node()

    def info(ignore_eos=False, decoded=0, max_tokens=100, walk="decode"):
        # the walk plan marks the prefill, the step before decode, FINAL_PREFILL
        return SimpleNamespace(graph_walk=walk, resource_configs={SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
                               dynamic_loop_iter_counts={"decode_loop": decoded}, max_tokens=max_tokens,
                               step_metadata={FINAL_PREFILL: walk == "prefill"})

    infos = {"eos": info(), "ignored": info(ignore_eos=True), "limit": info(decoded=9, max_tokens=10), "go": info(),
             "prefill_eos": info(walk="prefill")}
    rows = ("go", "limit", "eos", "ignored", "prefill_eos")
    tokens = torch.tensor([[7], [7], [3], [3], [3]]).repeat(1, width)
    asked = ["eos", "ignored", "limit", "go", "prefill_eos"]
    batched = node.check_stop_batched(asked, infos, HostRows(request_ids=rows, buffers={"new_token": tokens}))
    one_by_one = {rid: node.check_stop(rid, infos[rid], {"new_token": [tokens[rows.index(rid)]]}) for rid in asked}
    assert batched == {rid: stop for rid, stop in one_by_one.items() if stop} == {
        "eos": {"decode_loop"}, "prefill_eos": {END_PARTITION},
    }


def test_decoder_node_caps_captures_and_names(snapshot):
    """What the shared decoder gives Canary's node: prefill batches capped,
    decode captured on its ladder and prefill packed at the prompt's length per
    row, the sampled token fed back, the walk, loop and sampler names of
    Canary's graph, and positions advanced on the array the model declares,
    under its key."""
    from mstar.engine.state.adapters import node_resources
    from mstar.model.canary.config import KV_CACHE, SAMPLER
    from mstar.model.submodule_base import ARNodeInputs

    node = _tiny_node()
    assert node.max_batch_size("prefill") == 32 and node.max_batch_size("decode") is None
    capture, prefill = node.get_accelerator_graph_configs(torch.device("cpu"))
    assert capture.capture_graph_walk == "decode" and capture.capture_batch_sizes == [1, 2, 4, 8, 16, 32, 64]
    assert prefill.capture_graph_walk == "prefill" and prefill.capture_batch_sizes == [1, 2, 4, 8, 16, 32]
    assert [prefill.get_total_tokens(bs) for bs in (1, 4, 32)] == [[PROMPT_LENGTH], [4 * PROMPT_LENGTH], [320]]
    # template rows reserve the longest clip's context
    rows = prefill.get_node_inputs(4, 4 * PROMPT_LENGTH)
    assert [(r.input_seq_len, r.tensor_inputs["encoder_states"].shape[0]) for r in rows] == [(PROMPT_LENGTH, 13)] * 4
    outputs = {"new_token": [torch.tensor([5])]}
    node.postprocess("a", None, outputs)
    assert outputs["text_inputs"] is outputs["new_token"]
    model = CanaryModel(str(snapshot))
    walks = model.get_graph_walk_graphs()
    assert walks[node.DECODE_WALK].name == node.DECODE_LOOP and node.SAMPLER == SAMPLER
    # repeats need every output of the step on the device: one signal, not a second for the client
    assert {e.name for e in walks[node.DECODE_WALK].section.outputs} == {"text_inputs"}
    step = node.declare_step("decode", ["a"], [ARNodeInputs(input_ids=torch.tensor([5]), input_seq_len=1)])
    # the resource key YAML overrides name
    assert KV_CACHE in step.state and "rope" in {s.resource_key for s in node_resources(model)}


def test_canary_decodes_on_the_device_and_resident_tables_by_default():
    """Decode loops its token back on the device and runs on resident tables
    without a deployment asking; ``step_repeats:`` leaves that as it is."""
    from types import SimpleNamespace

    from mstar.model.canary.config import SAMPLER
    from mstar.model.submodule_base import DeviceLoopback
    from mstar.testing_utils.engine import make_engine, make_mgmt

    node = _tiny_node()
    loopback = DeviceLoopback(node.DECODE_WALK, "text_inputs", SAMPLER, "new_token")
    assert node.device_loopback == loopback and node.resident_walks() == {node.DECODE_WALK}
    sampler = SimpleNamespace(
        enable_device_loopback_reader=lambda: None,
        lean_capture=False, force_double_buffer=False, supports_preplan=False,
    )
    engine = make_engine(_submodules={"decoder": make_mgmt(node, resources={SAMPLER: sampler})})

    engine.enable_repeats("decoder", {node.DECODE_WALK})

    assert node.device_loopback == loopback
    assert node.resident_walks() == {node.DECODE_WALK} and node.get_stop_policies(node.DECODE_WALK)
