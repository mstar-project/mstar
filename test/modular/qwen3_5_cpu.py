"""A tiny random-weight Qwen3.5 (three gated-delta-net layers and one gated
full-attention layer), with or without its MTP head drafting, on the real
`Engine` with its real KV, position, recurrent, linear-attention and sampler
resources, on the CPU.

The kernels with no CPU path are `qwen3_cpu`'s (paged attention, RoPE, the
prompt's draw) and `gdn_cpu`'s (the delta rule and the short conv). The
tokenizer, which the model reads only for its eos id and chat template, is a
stub. Everything runs in float32 (the GDN pool's state dtype), so parity
checks use the margin rule: where two runs differ, the reference's top-2
margin there must be a near-tie.

`reference_greedy` decodes the same weights with no engine and no state kept
between tokens, and `reference_drafts` runs the MTP chain from scratch on the
target's hidden states of the whole sequence: what a verify step's drafts must
be, given what it kept.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from qwen3_cpu import PAGE, CPUAttentionSpec, _DenseAttention, _DensePositions, init_random

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources.attn.config import AttentionSpec
from mstar.engine.resources.draft.drafter import VERIFY_ACCEPTED, VERIFY_STOPPED
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec
from mstar.engine.resources.recurrent.config import RecurrentStateSpec
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.step import StepContext
from mstar.model.base import LabelLayout, Span
from mstar.model.submodule_base import HostRows

NODE = "LLM"
VOCAB = 97
STOP = 96
DTYPE = torch.float32
LINEAR, FULL = "linear_attention", "full_attention"
TEXT = dict(
    hidden_size=32, num_hidden_layers=4, layer_types=[LINEAR, LINEAR, LINEAR, FULL],
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48,
    linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
    linear_conv_kernel_dim=4, vocab_size=VOCAB, rms_norm_eps=1e-6, max_position_embeddings=512,
    attn_output_gate=True, tie_word_embeddings=False, eos_token_id=STOP, mtp_num_hidden_layers=1,
    mamba_ssm_dtype="float32", dtype="float32",
    # a rotary half of each head, its four frequency pairs split over the three MRoPE axes
    rope_parameters={"rope_theta": 10_000.0, "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1],
                     "mrope_interleaved": True},
)
# what FlashInfer's, FA3's and the GDN kernels take: head dims of 128, bf16 weights on an fp32 state
CUDA_TEXT = dict(
    TEXT, hidden_size=256, num_attention_heads=2, num_key_value_heads=1, head_dim=128, intermediate_size=512,
    linear_key_head_dim=128, linear_value_head_dim=128, vocab_size=512, dtype="bfloat16",
    rope_parameters={"rope_theta": 10_000.0, "partial_rotary_factor": 0.25, "mrope_section": [6, 5, 5],
                     "mrope_interleaved": True},
)
# the released checkpoints' full-attention head size, which the verify kernels also run at
CUDA_TEXT_256 = dict(
    CUDA_TEXT, head_dim=256,
    rope_parameters={"rope_theta": 10_000.0, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                     "mrope_interleaved": True},
)


VISION_START, VISION_END = 90, 91
# the image pad the tokenizer names (`Qwen3_5DenseModel._pad_ids`); the toy has no video pad
IMAGE_PAD = 92


class _Tokenizer:
    """What the model reads of a tokenizer outside ``process_prompt``."""
    eos_token_id = STOP
    unk_token_id = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return {"<|image_pad|>": IMAGE_PAD}.get(token)


def write_checkpoint(root: Path, **text) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
        "vision_start_token_id": VISION_START, "vision_end_token_id": VISION_END,
        "text_config": {**TEXT, **text},
    }))
    return root


def make_model(monkeypatch, root: Path, k: int | None = None, mode: str = "synchronous", **kwargs):
    from mstar.model.qwen3_5 import qwen3_5_model

    monkeypatch.setattr(qwen3_5_model, "AutoTokenizer", _Tokenizer)
    draft = None if k is None else {"method": "mtp", "num_draft_tokens": k, "mode": mode}
    return qwen3_5_model.Qwen3_5DenseModel(model_path_hf=str(root), draft=draft, **kwargs)


def build_lm(model, seed: int = 0, with_mtp: bool | None = None, round_to: torch.dtype | None = None):
    """The text stack (and MTP head), random from ``seed``, as the checkpoint
    would fill it; the same seed gives the same weights with or without the
    head. ``round_to``: the weights rounded through that dtype (a bf16 device
    model's, for a float32 reference of it)."""
    from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM

    mtp = model.draft is not None if with_mtp is None else with_mtp
    lm = model._build(lambda: Qwen3_5ForCausalLM(model.config, None, with_mtp=True), DTYPE, "cpu")
    init_random(lm, seed)
    with torch.no_grad():
        for name, p in lm.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                p.mul_(0.3)
            if round_to is not None:
                p.copy_(p.to(round_to))
    if not mtp:
        lm.mtp = None
    return lm.eval()


def cpu_specs(model, max_num_pages: int = 512, prefix_cache: bool = False):
    specs = []
    for spec in model.get_node_resources():
        if isinstance(spec, AttentionSpec):
            spec = CPUAttentionSpec(resource_key=spec.resource_key, nodes=spec.nodes, config=spec.config)
        if isinstance(spec, RecurrentStateSpec):
            # the conv window in the activations' dtype (a deployment's is bf16, as its activations)
            blocks = {name: dataclasses.replace(block, dtype=DTYPE) if name == "conv" else block
                      for name, block in spec.config.blocks.items()}
            # prompt snapshots at page boundaries, which a prefix hit restores
            prefix_slots = 8 if prefix_cache else 0
            spec = dataclasses.replace(spec, config=dataclasses.replace(
                spec.config, blocks=blocks, prefix_slots=prefix_slots,
            ))
        if isinstance(spec, KVSpec):
            spec.config.page_size = PAGE
            spec.config.max_num_pages = max_num_pages
            if prefix_cache:
                spec.config.prefix_cache_salt = "qwen3_5"
        specs.append(spec)
    return specs


def engine(model, seed: int = 0, prefix_cache: bool = False, device: str = "cpu", captured: bool = False) -> Engine:
    if device != "cpu":
        return _device_engine(model, seed, device, captured)
    lm = build_lm(model, seed)
    sub = model.make_llm_submodule(lm)
    # no autocast: the norms would round through bfloat16 (`mstar::rms_norm`)
    eng = Engine(graph_runtime=None, autocast_dtype=None)
    eng.load_model(
        {NODE: sub}, cpu_specs(model, prefix_cache=prefix_cache), WorkerParallelGroups(num_workers=1, global_rank=0),
        torch.device("cpu"), transfer_engine_info=None, kv_cache_type=DTYPE, model=model,
    )
    # what a capture's buffers would build: the sampler's slot masters, the loop-back slots
    eng._resources["sampler"].build_accelerator_graph_buffers([], 8, 256)
    return eng


def _device_engine(model, seed: int, device: str, captured: bool) -> Engine:
    """The node on a device with its own kernels, bf16 under autocast, captured
    when ``captured``; small pages so short prompts cross them."""
    lm = build_lm(model, seed).to(device=device, dtype=torch.bfloat16)
    sub = model.make_llm_submodule(lm)
    specs = model.get_node_resources()
    for spec in specs:
        if isinstance(spec, KVSpec):
            spec.config.page_size = 16
            spec.config.max_num_pages = 512
    eng = Engine(graph_runtime=None, autocast_dtype=torch.bfloat16)
    eng.load_model(
        {NODE: sub}, specs, WorkerParallelGroups(num_workers=1, global_rank=0), _indexed(device),
        transfer_engine_info=None, kv_cache_type=torch.bfloat16, model=model,
    )
    if captured:
        eng.warmup()
    else:
        eng._resources["sampler"].build_accelerator_graph_buffers([], 8, 256)
    return eng


def _info(rid, walk, max_tokens, sampling, metadata):
    return SimpleNamespace(
        rid=rid, request_id=rid, rid_handle=rid, graph_walk=walk, max_tokens=max_tokens, step_metadata=metadata,
        resource_configs={"sampler": sampling}, update_publish_info=lambda published: None,
        dynamic_loop_iter_counts={},
    )


class Driver:
    """Plays the worker, synchronously: a prompt step, then decode (or verify)
    steps, each built after the last was post-processed."""

    def __init__(self, eng: Engine, model):
        self.eng, self.model = eng, model
        self.out: dict[int, list[int]] = {}
        self.max: dict[int, int] = {}
        self.sampling: dict[int, SamplingReqConfig] = {}
        self.verifies = model.draft is not None
        self.accepted: list[int] = []
        self.device = eng._device

    def add(self, rid: int, prompt: list[int], max_tokens: int, keyed: bool = False, ignore_eos: bool = False) -> None:
        from mstar.engine.resources.kv.keys import chain

        scratch = model_slack(self.model)
        kv = KVReqConfig()
        whole = len(prompt) // PAGE
        kv.apply_conductor_config(
            kv_layout={"main": LabelLayout(
                spans=(Span("ids", "prefill_text", len(prompt), len(prompt), "text_inputs"),),
                growth=(("decode_loop", 1),), scratch=scratch,
            ).row()},
            loop_caps={"decode_loop": max_tokens},
            **({"prefix_keys": {"main": chain([prompt[at: at + PAGE] for at in range(0, whole * PAGE, PAGE)])},
                "prefix_tail": {"main": prompt[whole * PAGE:]}} if keyed else {}),
        )
        sampling = SamplingReqConfig(temperature=0.0, top_k=0, top_p=1.0, ignore_eos=ignore_eos)
        sampling.apply_conductor_config(seed=rid)
        self.sampling[rid] = sampling
        self.eng.add_request(rid, {"kv_cache": kv, "sampler": sampling})
        sampler = self.eng._resources["sampler"]
        if sampler.ask(rid, NODE, "prefill_text") is not None:
            sampler.reserve(rid, NODE, "prefill_text", None)
        self.out[rid], self.max[rid] = [], max_tokens

    def prefill(self, rids: list[int], prompts: dict[int, list[int]], prefix: bool = False) -> int:
        """One packed prompt step over ``rids``; with ``prefix``, each first takes
        its prefix-cache hit (the tokens it hit, of the last row, returned)."""
        from mstar.model.submodule_base import ARNodeInputs

        hit = 0
        if prefix:
            for rid in rids:
                inputs = ARNodeInputs(input_seq_len=len(prompts[rid]))
                found = self.eng._runner.resolve_cached_prefix(rid, NODE, "prefill_text", inputs)
                hit = 0 if found is None else found.tokens
                self.eng._runner.apply_cached_prefix(rid, NODE, "prefill_text", inputs, found)
        batch = self._batch("prefill_text", {rid: torch.tensor(prompts[rid], device=self.device) for rid in rids},
                            {"is_prefill": True, "last_prefill": True})
        self.eng.prepare_inputs(batch)
        out = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        tokens = out.check_stop_buffers["new_token"].reshape(-1).tolist()
        for i, rid in enumerate(rids):
            self.out[rid].append(int(tokens[i]))
        return hit

    def prefill_chunks(self, rid: int, prompt: list[int], bounds: list[int]) -> None:
        """``rid``'s prompt in chunks ending at each of ``bounds``, one step
        each, as the scheduler cuts a long prompt."""
        from mstar.engine.chunking import ChunkRange

        start = 0
        for end in bounds:
            batch = self._batch("prefill_text", {rid: torch.tensor(prompt, device=self.device)},
                                {"is_prefill": True, "last_prefill": end == len(prompt)})
            batch.chunk_ranges = {rid: ChunkRange(start, end, len(prompt), end - start if start else end)}
            self.eng.prepare_inputs(batch)
            out = self.eng.exec_and_postprocess(batch)
            assert batch.admit_error is None, batch.admit_error
            start = end
        self.out[rid].append(int(out.check_stop_buffers["new_token"].reshape(-1)[0]))

    def add_image_prompt(self, rid: int, texts: list[list[int]], image: torch.Tensor, grid, max_tokens: int):
        """A prompt of ``texts[0]``, one image (``image``: its merged embeddings,
        ``grid`` its (t, h, w) patches), then ``texts[1]``, as the LLM node
        gets it from the encoder: the image's pads between the vision
        sentinels, and the walk's media layout. Returns the walk's row as the
        reference reads it: its length, the spliced embeddings, their 3D
        positions and the counter's advance (``mrope_advance``)."""
        from mstar.model.multimodal import MEDIA_LAYOUT, splice_media

        sub = self.eng._submodules[NODE].submodule
        if sub._merge_size is None:
            sub._merge_size = sub.vision_config.spatial_merge_size
        n = image.shape[0]
        ids = torch.tensor([*texts[0], VISION_START, *[IMAGE_PAD] * n, VISION_END, *texts[1]])
        layout = {"offset": [len(texts[0]) + 1], "tokens": [n], "grid": [tuple(grid)], "output": ["vision_embeds"]}
        metadata = {"is_prefill": True, "last_prefill": True, MEDIA_LAYOUT: layout}
        self.out[rid], self.max[rid] = [], max_tokens
        sampling = SamplingReqConfig(temperature=0.0, top_k=0, top_p=1.0)
        sampling.apply_conductor_config(seed=rid)
        self.sampling[rid] = sampling
        inputs = {"vision_embeds": [image], "text_inputs": [ids]}
        info = _info(rid, "prefill_vision", max_tokens, sampling, metadata)
        # the row as the node builds it, for its length and positions (a new request starts at 0)
        self.eng.add_request(rid, {"kv_cache": KVReqConfig(), "sampler": sampling})
        media_row = sub._media_inputs(info, inputs)
        self.eng.remove_request(rid)
        length = media_row.input_seq_len
        advance = int(media_row.resource_step_info)
        with torch.no_grad():
            embeds = splice_media(sub.model.embed(ids), ids == IMAGE_PAD, image)
        row = SimpleNamespace(
            input_seq_len=length, input_embeds=embeds, custom_pos_ids=media_row.custom_pos_ids,
            tensor_inputs={"mrope_advance": advance},
        )
        kv = KVReqConfig()
        kv.apply_conductor_config(
            kv_layout={"main": LabelLayout(
                spans=(Span("rows", "prefill_vision", length, advance, "text_inputs"),),
                growth=(("decode_loop", 1),), scratch=model_slack(self.model),
            ).row()},
            loop_caps={"decode_loop": max_tokens},
        )
        self.eng.add_request(rid, {"kv_cache": kv, "sampler": sampling})
        sampler = self.eng._resources["sampler"]
        if sampler.ask(rid, NODE, "prefill_vision") is not None:
            sampler.reserve(rid, NODE, "prefill_vision", None)
        batch = ExecutingBatch(
            node_name=NODE, per_request_info={rid: info},
            step_context=StepContext(request_ids=(rid,), graph_walk="prefill_vision", slot=0, capture=False),
            per_request_input_tensors={rid: inputs},
        )
        self.eng.prepare_inputs(batch)
        out = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        self.out[rid].append(int(out.check_stop_buffers["new_token"].reshape(-1)[0]))
        return row

    def kv_rows(self, rid: int, layer: int, n: int) -> torch.Tensor:
        """``rid``'s K/V of layer ``layer`` (of the KV pool) at its first ``n`` slots."""
        kv = self.eng._resources["kv_cache"]
        pages = list(kv._streams[rid]["main"].page_indices)
        pos = torch.arange(n)
        return kv.kv_cache.read_tokens(
            layer_idx=layer, page_idx=torch.tensor(pages)[pos // PAGE], cache_idx=pos % PAGE,
        )

    def _batch(self, walk, ids, metadata):
        rids = list(ids)
        return ExecutingBatch(
            node_name=NODE,
            per_request_info={
                rid: _info(rid, walk, self.max[rid], self.sampling[rid], dict(metadata)) for rid in rids
            },
            step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
            per_request_input_tensors={rid: {"text_inputs": [row]} for rid, row in ids.items()},
        )

    def step(self, rids: list[int]) -> dict[int, bool]:
        batch = self._batch("decode", {rid: torch.tensor([self.out[rid][-1]], device=self.device) for rid in rids}, {})
        self.eng.prepare_inputs(batch)
        outputs = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        stopped = {}
        if self.verifies:
            host = HostRows(tuple(outputs.row_request_ids or rids), {
                name: t.cpu() for name, t in (outputs.check_stop_buffers or {}).items()
            })
            self.eng.cut_verify_step(batch, outputs, host)
            for i, rid in enumerate(host.request_ids):
                self.out[rid] += outputs.per_rid_outputs[rid]["new_token"][0].reshape(-1).tolist()
                stopped[rid] = bool(host.buffers[VERIFY_STOPPED][i])
            self.accepted += host.buffers[VERIFY_ACCEPTED].tolist()[: len(rids)]
            return stopped
        tokens = outputs.check_stop_buffers["new_token"].reshape(-1).tolist()
        for i, rid in enumerate(rids):
            self.out[rid].append(int(tokens[i]))
            stopped[rid] = tokens[i] == STOP or len(self.out[rid]) >= self.max[rid]
        return stopped

    def run(self, rids: list[int], max_steps: int = 400, remove: bool = True) -> None:
        live = [rid for rid in rids if self.out[rid][-1] != STOP and len(self.out[rid]) < self.max[rid]]
        for _ in range(max_steps):
            if not live:
                break
            stopped = self.step(live)
            live = [rid for rid in live if not stopped[rid]]
        assert not live, "rows still running"
        if remove:
            for rid in rids:
                self.eng.remove_request(rid)


def model_slack(model) -> int:
    return 0 if model.draft is None else model.draft.kv_slack


def decode(model, prompts: dict[int, list[int]], max_tokens: int, seed: int = 0, batched: bool = True,
           eng: Engine | None = None) -> Driver:
    driver = Driver(eng or engine(model, seed), model)
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, max_tokens)
    for group in ([list(prompts)] if batched else [[rid] for rid in prompts]):
        driver.prefill(group, prompts)
        driver.run(group)
    return driver


# ── the reference: no engine, every state from zeros ──


class _DensePool:
    def block(self, name, layer):
        return None


def dense_lm(model, seed: int = 0, round_to: torch.dtype | None = None):
    """The node's weights with dense stand-ins for its resources: each call runs
    a whole sequence from zero state."""
    from hybrid_toy import _DenseGDN

    from mstar.model.qwen3_5.config import ATTN, GDN_STATE, KV_CACHE, LINEAR_ATTN, ROPE

    lm = build_lm(model, seed, with_mtp=True, round_to=round_to)
    gdn, attn = _DenseGDN(), _DenseAttention()
    resources = {LINEAR_ATTN: gdn, GDN_STATE: _DensePool(), ATTN: attn, KV_CACHE: attn, ROPE: _DensePositions()}
    for module in [*lm.model.layers, *lm.mtp.layers]:
        module.self_attn.bind_resources(resources)
    # no engine binds the head here: its layer goes where the pool's join puts it, after the stack's
    lm.mtp.kv_layer_idx = len(model.config.full_layer_indices)
    return lm


def _cos_sin(lm, n: int):
    return lm.model.build_cos_sin(torch.arange(n).unsqueeze(0).expand(3, -1), dtype=DTYPE)


@torch.no_grad()
def target_hidden(lm, seq: list[int]) -> torch.Tensor:
    ids = torch.tensor(seq)
    return lm.model(lm.model.embed_tokens(ids), label="main", cos_sin=_cos_sin(lm, len(seq)))


@torch.no_grad()
def reference_greedy(model, prompt: list[int], max_tokens: int, seed: int = 0,
                     margins: list | None = None, round_to: torch.dtype | None = None) -> list[int]:
    lm = dense_lm(model, seed, round_to)
    seq, out = list(prompt), []
    while len(out) < max_tokens:
        logp = lm.lm_head(target_hidden(lm, seq))[-1].log_softmax(-1)
        top = logp.topk(2)
        token = int(top.indices[0])
        if margins is not None:
            margins.append(float(top.values[0] - top.values[1]))
        out.append(token)
        seq.append(token)
        if token == STOP:
            break
    return out


@torch.no_grad()
def reference_drafts(model, seq: list[int], k: int, seed: int = 0, margins: list | None = None,
                     round_to: torch.dtype | None = None) -> list[int]:
    """The ``k`` drafts after ``seq`` (its last token the step's input, at
    position ``len(seq) - 1``): `reference_drafts_embeds` of its embeddings at
    text positions."""
    lm = dense_lm(model, seed, round_to)
    embeds = lm.model.embed_tokens(torch.tensor(seq))
    positions = torch.arange(len(seq)).unsqueeze(0).expand(3, -1)
    return reference_drafts_embeds(model, embeds, positions, k, seed, margins, round_to, lm=lm)


@torch.no_grad()
def reference_drafts_embeds(model, embeds: torch.Tensor, positions: torch.Tensor, k: int, seed: int = 0,
                            margins: list | None = None, round_to: torch.dtype | None = None, lm=None) -> list[int]:
    """The ``k`` drafts after a sequence given as embeddings and 3D positions
    (its last row the step's input): the MTP head over every row on the
    target's hidden states, each row's following input the next row's
    embedding, at the row's own position; its last row gives the first draft;
    then the chain, each row on the last output and draft, one text position
    on. ``margins`` takes each draft's top-2 log-prob margin. It runs the same
    head on the same conventions, so it catches state and caching bugs, not a
    wrong input convention (which only acceptance on real weights shows)."""
    lm = lm or dense_lm(model, seed, round_to)
    n = embeds.shape[0]
    cos_sin = lm.model.build_cos_sin(positions, dtype=DTYPE)
    hidden = lm.model(embeds, label="main", cos_sin=cos_sin)[: n - 1]
    following = embeds[1:]
    rows = positions[:, : n - 1]
    drafts = []
    for _ in range(k):
        out = lm.mtp(hidden, following, cos_sin=lm.model.build_cos_sin(rows, dtype=DTYPE))
        logp = lm.lm_head(out[-1]).log_softmax(-1)
        top = logp.topk(2)
        if margins is not None:
            margins.append(float(top.values[0] - top.values[1]))
        draft = int(top.indices[0])
        drafts.append(draft)
        hidden = torch.cat([hidden, out[-1:]])
        following = torch.cat([following, lm.model.embed_tokens(torch.tensor([draft]))])
        rows = torch.cat([rows, rows[:, -1:] + 1], dim=1)
    return drafts


@torch.no_grad()
def forced_draft_logprobs(model, seq: list[int], drafts: list[int], seed: int = 0,
                          bf16: bool = False) -> list[torch.Tensor]:
    """The reference's log-probs at each of ``drafts``, the chain teacher-forced
    on the drafts before it (as `reference_drafts`, but each pass reads the
    given draft, not its own top-1), on the bf16-rounded weights; ``bf16``
    computes under bf16 autocast, the precision control a device check is
    judged against."""
    lm = dense_lm(model, seed, torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        embeds = lm.model.embed_tokens(torch.tensor(seq))
        n = embeds.shape[0]
        positions = torch.arange(n).unsqueeze(0).expand(3, -1)
        hidden = lm.model(embeds, label="main", cos_sin=lm.model.build_cos_sin(positions, dtype=DTYPE))[: n - 1]
        following, rows = embeds[1:], positions[:, : n - 1]
        logprobs = []
        for draft in drafts:
            out = lm.mtp(hidden, following, cos_sin=lm.model.build_cos_sin(rows, dtype=DTYPE))
            logprobs.append(lm.lm_head(out[-1]).float().log_softmax(-1))
            hidden = torch.cat([hidden, out[-1:]])
            following = torch.cat([following, lm.model.embed_tokens(torch.tensor([draft]))])
            rows = torch.cat([rows, rows[:, -1:] + 1], dim=1)
    return logprobs


@torch.no_grad()
def reference_greedy_embeds(model, embeds: torch.Tensor, positions: torch.Tensor, max_tokens: int, seed: int = 0,
                            margins: list | None = None) -> list[int]:
    """`reference_greedy` from a prompt given as embeddings and 3D positions
    (an image prompt): each reply token is text, one position past the last."""
    lm = dense_lm(model, seed)
    out = []
    while len(out) < max_tokens:
        cos_sin = lm.model.build_cos_sin(positions, dtype=DTYPE)
        logp = lm.lm_head(lm.model(embeds, label="main", cos_sin=cos_sin))[-1].log_softmax(-1)
        top = logp.topk(2)
        token = int(top.indices[0])
        if margins is not None:
            margins.append(float(top.values[0] - top.values[1]))
        out.append(token)
        if token == STOP:
            break
        embeds = torch.cat([embeds, lm.model.embed_tokens(torch.tensor([token]))])
        positions = torch.cat([positions, positions[:, -1:] + 1], dim=1)
    return out


def vision_config(merge: int = 2):
    from mstar.model.qwen3_5.config import Qwen3_5VisionConfig

    return Qwen3_5VisionConfig(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2, in_channels=3, patch_size=2,
        temporal_patch_size=1, spatial_merge_size=merge, num_position_embeddings=16,
        out_hidden_size=TEXT["hidden_size"],
    )


def _indexed(device) -> torch.device:
    """The device as a worker passes it: a GPU with its index (capture sets the device by index)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device
