"""Per-request LoRA declared from a model's modules (``module_lora_spec``), on real
weights and real or random adapters, on GPU. Nothing here runs in CI.

Per model (``MODELS``): the LoRA node's submodule is built by the model itself (its
checkpoint, its loader, bf16 on CUDA), its LoRA pool on CUDA (Triton kernels, bf16
slots) is bound by the generic path, and one step is declared through
``declared_step`` over a batch mixing every adapter and the base model. Phases, each
writing JSON under ``--out``:

``merged``
    The node's stack (attention as causal SDPA per row, no RoPE on either side) on that
    mixed batch, inside the step's compute scope, against the same stack with each
    row's adapter merged into the weights (``W + scaling * B @ A`` in fp32, cast to
    bf16) and nothing bound: the Triton kernels, the real shapes and the mixed-batch
    addressing against a merge. Rule, per row: the last token's top-1 is equal, the
    logits' cosine similarity is at least 0.999, and (control) each adapter row moves
    its logits from the base by more than ten times the row's error.
``hf``
    Models with a transformers reference for the LoRA node (``MODELS[...].hf``):
    Orpheus (``LlamaForCausalLM`` + ``PeftModel``), Higgs Audio's LLM (``Qwen3ForCausalLM``
    over the checkpoint's text weights), the Qwen3-Omni Thinker (transformers' thinker)
    and Chatterbox T3 (transformers' ``LlamaModel`` under the speech head); the adapter
    goes in through PEFT (``inject_adapter_in_model`` where the reference is not a
    ``PreTrainedModel`` whose layout the adapter names). The stub applies the
    reference's own RoPE. Rule: top-1 in the reference's top-5, cosine at least 0.99,
    and the same control against the reference's base.
``serve``
    The model served with its stock config plus a ``resources: lora:`` block (CUDA
    graphs and compile as shipped), against the same server started on a checkpoint
    with the first adapter merged in (``model_kwargs: model_path_hf``, no ``lora``
    block). Per adapter: the request succeeds and differs from the base model's; the
    first adapter's output is compared with the merged server's (text: the common
    prefix; audio: length and correlation, judged loosely: bf16 greedy decoding may part
    from a merge after a near tie). Then every adapter and the base model at once, rows
    on different adapters in one batch: all succeed.

    python -m benchmark.lora.module_parity --model orpheus --out ~/lora_module_parity/orpheus
    python -m benchmark.lora.module_parity --model chatterbox --phase merged --out ...

Adapters: Orpheus has a PEFT ecosystem on the Hub (``DEFAULT_ADAPTERS``: rank 64 alpha
32 on the released fine-tune, an unsloth fine-tune at rank 64, an rsLoRA one at rank 32
and a rank-16 one on the pretrained base). The others have none known, so ``--synthetic
N`` (the default for them) writes N random PEFT adapters of the real shapes over every
target but the vocabulary-sized ones, each with an ``adapter_config.json`` (rank 16,
alpha 32), which every phase reads alike. Every target the model declares is served,
the opt-in experts targets included (the Qwen3-Omni Thinker's routed experts: a synthetic
adapter there holds an A and B for every expert, about 3.3 GB in fp32 at rank 16).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ADAPTERS = {
    "orpheus": {
        "sft": "Tachyeon/orpheus-3b-sft-lora",
        "jarvis": "derekurban2001/orpheus-jarvis-voice-lora",
        "rslora": "ErisGrey/orpheus-3b-0.1-ft-lora-ft_20250810_232018",
        "bengali": "kawshikbuet17/bengali-orpheus-tts-adapter",
    },
}
TEXTS = [
    "Hello, how are you doing today?",
    "The meeting is moved to Thursday afternoon.",
    "Please read the following numbers slowly: four, eight, fifteen, sixteen.",
]
BASE = "<base>"


@dataclass
class ModelSpec:
    """How the script drives one model's LoRA node."""

    node: str
    prompts: Callable          # (model) -> list of 1-D id tensors
    run: Callable              # (submodule, ids, lengths, rope) -> last-token logits, a row each
    declare: Callable          # (submodule, rids, ids) -> SubmoduleStep, via declared_step
    config: str                # the stock serving config
    request: dict              # serve: /generate fields (output_modalities, model_kwargs, audio)
    hf: Callable | None = None  # (model) -> HfReference


@dataclass
class HfReference:
    forward: Callable          # (model, ids) -> last-token logits
    model: object
    rope: Callable | None
    pretrained: bool = False   # the adapter's own layout: PeftModel; else inject by name
    adapter_name: Callable = lambda module: module  # adapter module -> the reference's module


# --- the stack, stubbed only where it holds no adapter ------------------------------------

class _Attend:
    """Causal GQA attention within each row (SDPA): the cursor API the shared attention
    layers call."""

    label = "main"

    def __init__(self, lengths):
        self.lengths = lengths

    def bind_step(self, label):
        del label

    def set_layer_idx(self, idx):
        del idx

    def __call__(self, q, k, v):
        import torch

        out, at = [], 0
        group = q.shape[1] // k.shape[1]
        for n in self.lengths:
            qs = q[at:at + n].transpose(0, 1)
            ks = k[at:at + n].repeat_interleave(group, dim=1).transpose(0, 1)
            vs = v[at:at + n].repeat_interleave(group, dim=1).transpose(0, 1)
            out.append(torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=True).transpose(0, 1))
            at += n
        return torch.cat(out)


def _stub_attention(root, lengths, rope=None) -> None:
    """Every attention layer under ``root``: the SDPA stub, and ``rope(q, k)`` in place of
    the engine's position resource (None leaves positions out on both sides)."""
    for module in root.modules():
        if hasattr(module, "_attn_key") and hasattr(module, "qkv_proj"):
            module.attend = _Attend(lengths)
            if rope is not None:
                module._apply_rope = lambda q, k, label, rope=rope: rope(q, k)


def _last(hidden, lengths):
    import torch

    return hidden[torch.tensor(lengths, device=hidden.device).cumsum(0) - 1]


def _ar_inputs(ids):
    from mstar.model.submodule_base import ARNodeInputs

    return [ARNodeInputs(input_ids=i, input_seq_len=len(i)) for i in ids]


def _random_ids(vocab: int, lengths=(17, 5, 33)):
    import torch

    g = torch.Generator().manual_seed(0)
    return [torch.randint(1, vocab, (n,), generator=g) for n in lengths]


# --- RoPE as the references compute it ------------------------------------------------------

_POSITIONS: dict = {}


def _set_positions(lengths) -> None:
    import torch

    _POSITIONS["positions"] = torch.cat([torch.arange(n) for n in lengths])


def _rotate(q, k, cos, sin):
    import torch

    def half(x):
        n = x.shape[-1] // 2
        return torch.cat((-x[..., n:], x[..., :n]), dim=-1)

    return q * cos + half(q) * sin, k * cos + half(k) * sin


def _hf_rope(rotary, three_d: bool = False):
    """``rope(q, k)`` for ``[tokens, heads, dim]`` over positions 0.. of each row, with
    transformers' ``rotary`` (Llama 3 scaling and all; a text-only MRoPE's three
    position rows are equal, which makes it the 1D RoPE)."""
    def rope(q, k):
        positions = _POSITIONS["positions"].to(q.device)[None]
        if three_d:
            positions = positions[None].expand(3, 1, -1)
        cos, sin = rotary(q, positions)
        return _rotate(q, k, cos[0][:, None, :].to(q.dtype), sin[0][:, None, :].to(q.dtype))
    return rope


def _load_hf(module, snapshot: str, rename, strict: bool = True, files=None) -> None:
    """The checkpoint's tensors into ``module`` under ``rename`` (None skips one)."""
    import torch
    from safetensors.torch import load_file

    state = {}
    for path in sorted(Path(snapshot).glob("*.safetensors")):
        if files and path.name not in files:
            continue
        for name, tensor in load_file(str(path)).items():
            target = rename(name)
            if target is not None:
                state[target] = tensor.to(torch.bfloat16)
    missing, unexpected = module.load_state_dict(state, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    if strict and (missing or unexpected):
        raise SystemExit(f"reference load: missing {missing[:5]}, unexpected {unexpected[:5]}")


def _snapshot(model) -> str:
    # the base Model's checkpoint_path() is None (no prefix-cache manifest), not a path
    return model.checkpoint_path() or model.local_dir


# --- per model ----------------------------------------------------------------------------

def _orpheus() -> ModelSpec:
    from mstar.model.submodule_base import declared_step

    def prompts(model):
        import torch

        cfg = model.config
        out = []
        for i, text in enumerate(TEXTS):
            voice = cfg.available_voices[i % len(cfg.available_voices)]
            ids = model.tokenizer(f"{voice}: {text}", return_tensors="pt").input_ids[0]
            out.append(torch.cat([torch.tensor([cfg.start_token_id]), ids, torch.tensor(cfg.end_token_ids)]))
        return out

    def run(sub, ids, lengths, rope=None):
        _stub_attention(sub.language_model, lengths, rope)
        hidden = sub.language_model(sub.embed_tokens(ids), label="main")
        return sub.lm_head(_last(hidden, lengths)).float()

    def hf(model):
        import torch
        from transformers import AutoModelForCausalLM

        ref = AutoModelForCausalLM.from_pretrained(_snapshot(model), torch_dtype=torch.bfloat16, device_map="cuda")
        return HfReference(
            forward=lambda m, ids: m(ids[None].cuda()).logits[0, -1].float(),
            model=ref.eval(), rope=_hf_rope(ref.model.rotary_emb), pretrained=True,
        )

    return ModelSpec(
        "LLM", prompts, run, lambda sub, rids, ids: declared_step(sub, "prefill", rids, _ar_inputs(ids)),
        config="configs/orpheus_colocated.yaml",
        request={"output_modalities": "audio", "model_kwargs": {"voice": "tara", "temperature": 0.0}},
        hf=hf,
    )


def _higgs() -> ModelSpec:
    from mstar.model.submodule_base import declared_step

    def run(sub, ids, lengths, rope=None):
        _stub_attention(sub.model, lengths, rope)
        hidden = sub.model(input_embeds=sub.model.embed_tokens(ids), label="main")
        return sub.model.lm_head(_last(hidden, lengths)).float()

    def hf(model):
        """transformers' Qwen3 over the checkpoint's text weights: the LLM Higgs runs."""
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from mstar.model.higgs_audio.components.llm import higgs_llm_remap

        c = model.config
        ref = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=c.vocab_size, hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
            num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads, head_dim=c.head_dim, rms_norm_eps=c.rms_norm_eps,
            rope_theta=c.rope_theta, tie_word_embeddings=False,
        )).to(torch.bfloat16)

        def rename(name):
            name = higgs_llm_remap(name)
            if name is None:
                return None
            return name if name.startswith("lm_head") else f"model.{name}"

        _load_hf(ref, _snapshot(model), rename)
        return HfReference(
            forward=lambda m, ids: m(ids[None].cuda()).logits[0, -1].float(),
            model=ref.cuda().eval(), rope=_hf_rope(ref.model.rotary_emb),
            adapter_name=lambda module: rename(module + ".weight").removesuffix(".weight"),
        )

    return ModelSpec(
        "LLM", lambda model: [model.tokenizer(t, return_tensors="pt").input_ids[0] for t in TEXTS], run,
        lambda sub, rids, ids: declared_step(sub, "prefill_text", rids, _ar_inputs(ids)),
        config="configs/higgs_audio.yaml",
        request={"output_modalities": "text", "model_kwargs": {"temperature": 0.0}, "audio": True},
        hf=hf,
    )


def _qwen3_omni() -> ModelSpec:
    from mstar.model.submodule_base import declared_step

    def run(sub, ids, lengths, rope=None):
        _stub_attention(sub.model, lengths, rope)
        hidden, _, _ = sub.model(input_embeds=sub.model.model.embed_tokens(ids), label="main")
        return sub.model.lm_head(_last(hidden, lengths)).float()

    def hf(model):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration

        thinker = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            _snapshot(model), torch_dtype=torch.bfloat16, device_map="cuda",
        ).thinker
        return HfReference(
            forward=lambda m, ids: m(input_ids=ids[None].cuda()).logits[0, -1].float(),
            model=thinker.eval(), rope=_hf_rope(thinker.model.rotary_emb, three_d=True),
            adapter_name=lambda module: module.removeprefix("thinker."),
        )

    return ModelSpec(
        "Thinker", lambda model: _random_ids(150000), run,
        lambda sub, rids, ids: declared_step(sub, "prefill_text", rids, _ar_inputs(ids)),
        config="configs/qwen3omni_colocated.yaml",
        request={"output_modalities": "text", "model_kwargs": {"temperature": 0.0}},
        hf=hf,
    )


def _qwen3_tts() -> ModelSpec:
    from mstar.model.submodule_base import declared_step

    def run(sub, ids, lengths, rope=None):
        _stub_attention(sub.model, lengths, rope)
        hidden = sub.model(sub.model.model.codec_embedding(ids), label="main")
        return sub.model.codec_head(_last(hidden, lengths)).float()

    return ModelSpec(
        "Talker", lambda model: _random_ids(2048), run,
        lambda sub, rids, ids: declared_step(sub, "talker_prefill", rids, _ar_inputs(ids)),
        config="configs/qwen3tts.yaml",
        request={"output_modalities": "audio", "model_kwargs": {"temperature": 0.0}},
    )


def _chatterbox() -> ModelSpec:
    from mstar.model.submodule_base import declared_step

    def run(sub, ids, lengths, rope=None):
        _stub_attention(sub.model, lengths, rope)
        hidden = sub.model.hidden(sub.model.speech_emb(ids), label="main")
        return sub.model.logits(_last(hidden, lengths)).float()

    def declare(sub, rids, ids):
        from mstar.model.submodule_base import ARNodeInputs

        # without guidance, a row per request (the CPU test covers guidance's two)
        return declared_step(sub, "prefill", rids, [
            ARNodeInputs(input_seq_len=len(i), input_embeds=sub.model.speech_emb(i), resource_step_info=False)
            for i in ids
        ])

    def hf(model):
        """transformers' ``LlamaModel`` (T3's ``tfmr``) under the speech head."""
        import torch
        from torch import nn
        from transformers import LlamaConfig, LlamaModel

        t3 = model.config.t3
        bb = t3.backbone
        cfg = LlamaConfig(
            vocab_size=8, hidden_size=bb.hidden_size, intermediate_size=bb.intermediate_size,
            num_hidden_layers=bb.num_hidden_layers, num_attention_heads=bb.num_attention_heads,
            num_key_value_heads=bb.num_key_value_heads, head_dim=bb.head_dim, rms_norm_eps=bb.norm_eps,
            rope_theta=bb.rope_theta, rope_scaling=dict(bb.rope_scaling),
            max_position_embeddings=bb.max_position_embeddings,
        )

        class T3Reference(nn.Module):
            def __init__(self):
                super().__init__()
                self.tfmr = LlamaModel(cfg)
                self.speech_emb = nn.Embedding(t3.speech_vocab_size, bb.hidden_size)
                self.speech_head = nn.Linear(bb.hidden_size, t3.speech_vocab_size, bias=t3.speech_head_bias)

            def forward(self, ids):
                hidden = self.tfmr(inputs_embeds=self.speech_emb(ids)[None]).last_hidden_state
                return self.speech_head(hidden[0, -1])

        ref = T3Reference().to(torch.bfloat16)
        keep = ("tfmr.", "speech_emb.", "speech_head.")
        _load_hf(ref, _snapshot(model),
                 lambda n: n if n.startswith(keep) and not n.startswith("tfmr.embed_tokens.") else None,
                 strict=False, files=(model.config.t3_weights,))
        return HfReference(
            forward=lambda m, ids: m(ids.cuda()).float(), model=ref.cuda().eval(),
            rope=_hf_rope(ref.tfmr.rotary_emb),
            # mstar names T3's Llama stack `backbone`; the reference keeps the checkpoint's `tfmr`
            adapter_name=lambda module: (
                "tfmr." + module.removeprefix("backbone.") if module.startswith("backbone.") else module),
        )

    return ModelSpec(
        "T3", lambda model: _random_ids(model.config.t3.speech_vocab_size - 10), run, declare,
        config="configs/chatterbox.yaml",
        request={"output_modalities": "audio", "model_kwargs": {"temperature": 0.0}},
        hf=hf,
    )


MODELS = {
    "orpheus": _orpheus, "higgs_audio": _higgs, "qwen3_omni": _qwen3_omni,
    "qwen3_tts": _qwen3_tts, "chatterbox": _chatterbox,
}


def _model(name: str):
    from mstar.model.registry import get_model_class, model_init_kwargs

    return get_model_class(name)(**model_init_kwargs(name))


def _lora_spec(model):
    """The model's LoRA spec, declared beside its other resources."""
    from mstar.engine.resources.lora.serving import lora_spec

    return lora_spec(model.get_node_resources())


def _submodule(model, node: str):
    import torch

    from mstar.distributed.communication import CommGroup

    sub = model.get_submodule(node, device="cuda", tp_group=CommGroup.trivial(), autocast_dtype=torch.bfloat16)
    return sub.to(torch.bfloat16).requires_grad_(False).eval()


# --- adapters -----------------------------------------------------------------------------

def _adapter_file(spec: str) -> str:
    if os.path.isfile(spec):
        return spec
    if os.path.isdir(spec):
        return str(Path(spec) / "adapter_model.safetensors")
    from huggingface_hub import snapshot_download

    return str(Path(snapshot_download(spec, allow_patterns=["adapter_model.safetensors", "adapter_config.json"]))
               / "adapter_model.safetensors")


def _synthetic(spec, directory: Path, count: int, rank: int = 16, alpha: float = 32.0,
               std: float = 0.02) -> dict[str, str]:
    """Random PEFT adapters, each in a directory with its ``adapter_config.json``, over
    every target but the vocabulary-sized ones (heads, embeddings), named as the
    checkpoint names its modules."""
    import torch
    from safetensors.torch import save_file

    module_map = spec.module_map
    out = {}
    for n in range(count):
        g = torch.Generator().manual_seed(1000 + n)
        tensors, modules = {}, set()
        for path, ref in module_map.bindings.items():
            target = _declared(spec)[ref.target]
            if target.kind == "embedding" or target.per_segment:
                continue
            for name, rows in _checkpoint_names(module_map, path, target):
                key = f"base_model.model.{name}"
                tensors[f"{key}.lora_A.weight"] = torch.randn(rank, target.in_features, generator=g) * std
                tensors[f"{key}.lora_B.weight"] = torch.randn(rows, rank, generator=g) * std
                modules.add(name.rsplit(".", 1)[-1])
        folder = directory / f"synthetic{n}"
        folder.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(folder / "adapter_model.safetensors"))
        (folder / "adapter_config.json").write_text(json.dumps({
            "r": rank, "lora_alpha": alpha, "target_modules": sorted(modules), "peft_type": "LORA",
        }))
        out[f"synthetic{n}"] = str(folder / "adapter_model.safetensors")
    return out


def _checkpoint_names(module_map, path: str, target) -> list[tuple[str, int]]:
    """The checkpoint modules that load ``path``'s slices: each loader rule turned back
    (and the checkpoint prefix restored), kept when the map places it on this path. An
    experts target's are the per-expert ``experts.<N>.<slice>_proj`` modules (the HF MoE
    layout), each slice of each expert."""
    prefix = module_map.rules.prefix
    if target.experts > 1:
        stem = path.rsplit(".", 1)[0]
        stems = [(f"{stem}.{e}.{name}_proj", f"{name}:{e}")
                 for e in range(target.experts) for name in target.slice_names]
    elif target.num_slices == 1:
        stems = [(path, None)]
    else:
        stems = [
            (path.replace(rule.target_suffix, rule.source_suffix), rule.shard_id)
            for rule in module_map.rules.stacked_params if path.endswith(rule.target_suffix)
        ]
    names = []
    for stem, shard in stems:
        for candidate in (prefix + stem, stem):
            ref = module_map(candidate)
            if ref is not None and ref.shard_id == shard:
                names.append((candidate, sum(target.out_slices[i] for i in target.module_slices(shard))))
                break
    if not names:
        raise SystemExit(f"no checkpoint name places {path}; pass real adapters for this model")
    return names


def _declared(spec) -> dict:
    """Every target the model declares, the opt-in experts targets (``dropped_targets``)
    included."""
    return {**spec.targets, **spec.dropped_targets}


def _experts_opt_in(spec) -> dict:
    """The ``lora:`` block's ``targets``, naming every target when the model declares opt-in
    experts targets; else nothing, so the API server narrows to what the adapters touch."""
    if spec.dropped_targets:
        return {"targets": sorted(_declared(spec))}
    return {}


def _max_rank(adapters: dict[str, str]) -> int:
    from safetensors import safe_open

    rank = 16
    for path in adapters.values():
        with safe_open(path, "pt") as f:
            for key in f.keys():
                if key.endswith(("lora_A.weight", "lora_embedding_A")):
                    rank = max(rank, f.get_slice(key).get_shape()[0])
    return rank


# --- phases -------------------------------------------------------------------------------

def _rows(adapters: dict[str, str]) -> list[tuple[str, str | None]]:
    return [(name, None if name == BASE else name) for name in (BASE, *adapters)]


def _row(rows, adapter, j) -> str:
    return next(r for r, a, _ in rows if a == adapter and r.endswith(f"_{j}"))


def _lora_logits(spec_: ModelSpec, model, adapters: dict[str, str], prompts, rope=None):
    """Every (adapter, prompt) in one mixed step through the generic path, inside the
    step's compute scope."""
    import torch

    from mstar.engine.resources.lora import LORA_RESOURCE_KEY, LoraReqConfig
    from mstar.engine.resources.lora.pool import LoraAdapterPool
    from mstar.engine.resources.lora.store import check_adapters
    from mstar.engine.resources.step import StepContext

    spec = _lora_spec(model)
    # every target the model declares, the opt-in experts targets included
    spec.apply_yaml_overrides(max_slots=len(adapters), max_rank=_max_rank(adapters), adapters=adapters,
                              targets=sorted(_declared(spec)))
    spec.adapter_digests = check_adapters(spec)  # as the API server pins them
    spec.dtype = torch.bfloat16
    pool = LoraAdapterPool(spec, "cuda")
    sub = _submodule(model, spec_.node)
    sub.bind_node_resources({LORA_RESOURCE_KEY: pool})
    rows = [(f"r{i}_{j}", adapter, ids) for i, (_, adapter) in enumerate(_rows(adapters))
            for j, ids in enumerate(prompts)]
    rids = [r for r, _, _ in rows]
    step = spec_.declare(sub, rids, [ids.cuda() for _, _, ids in rows])
    for rid, adapter, _ in rows:
        pool.ingest_request(rid, None if adapter is None else LoraReqConfig(adapter))
    ctx = StepContext(request_ids=rids, graph_walk="prefill", slot=0, capture=False)
    assert pool.admit(step.get(LORA_RESOURCE_KEY), ctx).ok
    pool.plan(step.get(LORA_RESOURCE_KEY), ctx)
    lengths = [len(i) for _, _, i in rows]
    _set_positions(lengths)
    pool.enter_compute()
    try:
        with torch.no_grad():
            logits = spec_.run(sub, torch.cat([i for _, _, i in rows]).cuda(), lengths, rope)
    finally:
        pool.exit_compute()
    return {rid: logits[k] for k, rid in enumerate(rids)}, rows


def _merge_into(sub, spec, path: str) -> None:
    """``W += scaling * B @ A`` in fp32 for every module of the adapter at ``path``, on
    the module the spec binds it to, at its slices' rows (one rank: no sharding)."""
    import torch

    from mstar.engine.resources.lora.checkpoint import read_lora_file

    module_map = spec.module_map
    by_ref = {(ref.target, ref.layer): p for p, ref in module_map.bindings.items()}
    modules = dict(sub.named_modules())
    prefix = f"{module_map.root_path}." if module_map.root_path else ""
    for name, layer in read_lora_file(path, convert_keys=spec.convert_keys).layers.items():
        if spec.ignored_modules is not None and spec.ignored_modules(name):
            continue
        ref = module_map(name)
        target = _declared(spec)[ref.target]
        bound = by_ref[(ref.target, ref.layer)]
        if ref.param is None:
            weight = modules[prefix + bound].weight
        else:  # one expert's matrix of a stacked parameter
            stacked = getattr(modules[prefix + bound.removesuffix(f".{ref.param}")], ref.param)
            weight = stacked[target.expert_of(ref.shard_id)]
        a, b = layer.a.float().reshape(layer.a.shape[0], -1), layer.b.float().reshape(layer.b.shape[0], -1)
        delta = (layer.alpha / a.shape[0]) * (b.cuda() @ a.cuda())
        with torch.no_grad():
            if target.kind == "embedding":
                weight += delta.t().to(weight.dtype)
                continue
            at = 0
            for i in target.module_slices(ref.shard_id):
                start, rows = sum(target.out_slices[:i]), target.out_slices[i]
                part = (weight[start:start + rows].float() + delta[at:at + rows]).to(weight.dtype)
                weight[start:start + rows] = part.view_as(weight[start:start + rows])
                at += rows


def _judge(name: str, got, ref, base, cos_min: float, topk: int) -> dict:
    import torch

    cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
    in_topk = got.argmax().item() in ref.topk(topk).indices.tolist()
    err, moved = (got - ref).norm().item(), (ref - base).norm().item()
    return {
        "cos": cos, "top1_in_ref_topk": in_topk, "err": err, "adapter_moved": moved,
        "pass": cos >= cos_min and in_topk and (name == BASE or moved > 10 * err),
    }


def merged(args, spec_: ModelSpec) -> int:
    import torch

    model = _model(args.model)
    prompts = spec_.prompts(model)
    got, rows = _lora_logits(spec_, model, args.adapters, prompts)
    report, failed, base = {}, [], {}
    for name, adapter in _rows(args.adapters):
        # a model instance of its own: a model may cache its submodules, and the cached one
        # has the pool bound (and, after a merge, the previous adapter merged in)
        sub = _submodule(_model(args.model), spec_.node)
        if adapter is not None:
            _merge_into(sub, _lora_spec(model), args.adapters[adapter])
        for j, ids in enumerate(prompts):
            _set_positions([len(ids)])
            with torch.no_grad():
                ref = spec_.run(sub, ids.cuda(), [len(ids)])[0]
            if adapter is None:
                base[j] = ref
            report[f"{name}/{j}"] = r = _judge(name, got[_row(rows, adapter, j)], ref, base[j], 0.999, 1)
            if not r["pass"]:
                failed.append(f"{name}/{j}")
        del sub
        torch.cuda.empty_cache()
    return _report(args, "merged", report, failed)


def _attach(reference: HfReference, adapters: dict[str, str]):
    """Every adapter on the reference through PEFT, under its own name; returns the
    model and ``activate(name | None)``."""
    import peft
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file

    if reference.pretrained:
        names = list(adapters)
        model = peft.PeftModel.from_pretrained(reference.model, str(Path(adapters[names[0]]).parent),
                                               adapter_name=names[0])
        for name in names[1:]:
            model.load_adapter(str(Path(adapters[name]).parent), adapter_name=name)
    else:
        model = reference.model
        for name, path in adapters.items():
            config = json.loads((Path(path).parent / "adapter_config.json").read_text())
            state = {}
            for key, tensor in load_file(path).items():
                module, _, kind = key.removeprefix("base_model.model.").partition(".lora_")
                state[f"{reference.adapter_name(module)}.lora_{kind}"] = tensor.cuda()
            lora = peft.LoraConfig(
                r=config["r"], lora_alpha=config["lora_alpha"], use_rslora=config.get("use_rslora", False),
                target_modules=sorted({k.rsplit(".lora_", 1)[0] for k in state}),
            )
            model = peft.inject_adapter_in_model(lora, model, adapter_name=name)
            result = peft.set_peft_model_state_dict(model, state, adapter_name=name)
            if getattr(result, "unexpected_keys", None):
                raise SystemExit(f"{name}: PEFT did not place {result.unexpected_keys[:5]}")

    def activate(name: str | None) -> None:
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                on = name is not None and name in module.lora_A
                module.enable_adapters(on)
                if on:
                    module.set_adapter(name)

    return model, activate


def hf(args, spec_: ModelSpec) -> int:
    """transformers + PEFT on the same ids; the stub applies the reference's RoPE."""
    import torch

    if spec_.hf is None:
        raise SystemExit(f"{args.model} has no transformers reference for its LoRA node; run --phase merged")
    model = _model(args.model)
    prompts = spec_.prompts(model)
    reference = spec_.hf(model)
    got, rows = _lora_logits(spec_, model, args.adapters, prompts, rope=reference.rope)
    ref_model, activate = _attach(reference, args.adapters)
    report, failed, base = {}, [], {}
    for name, adapter in _rows(args.adapters):
        activate(adapter)
        for j, ids in enumerate(prompts):
            _set_positions([len(ids)])
            with torch.no_grad():
                ref = reference.forward(ref_model, ids)
            if adapter is None:
                base[j] = ref
            report[f"{name}/{j}"] = r = _judge(name, got[_row(rows, adapter, j)], ref, base[j], 0.99, 5)
            if not r["pass"]:
                failed.append(f"{name}/{j}")
    return _report(args, "hf", report, failed)


def serve(args, spec_: ModelSpec) -> int:
    """The model served with the adapters, against a merged checkpoint served alike."""
    import yaml

    from benchmark.eval.server import ManagedServer

    if spec_.request.get("audio") and not args.audio:
        raise SystemExit(f"serve: {args.model} takes speech in; pass --audio PATH")
    base_config = yaml.safe_load((Path(__file__).resolve().parents[2] / spec_.config).read_text())
    failed, report = [], {}
    first = next(iter(args.adapters))
    names = (BASE, *args.adapters)
    with tempfile.TemporaryDirectory(prefix="lora-module-parity-") as tmp:
        lora_config = {**base_config, "resources": {
            **(base_config.get("resources") or {}),
            "lora": {"max_slots": max(1, len(args.adapters) - 1), "max_rank": _max_rank(args.adapters),
                     "adapters": dict(args.adapters), **_experts_opt_in(_lora_spec(_model(args.model)))},
        }}
        path = Path(tmp) / "lora.yaml"
        path.write_text(yaml.safe_dump(lora_config))
        with ManagedServer(args.model, config=str(path), log_path=args.out / f"server_{args.model}_lora.log") as s:
            single = {n: _generate(s.url, spec_, args, None if n == BASE else n, TEXTS[0]) for n in names}
            jobs = [(n, text) for n in names for text in TEXTS]
            with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
                mixed = list(pool.map(lambda j: _generate(s.url, spec_, args, None if j[0] == BASE else j[0], j[1]),
                                      jobs))
        for name, out in single.items():
            if out["status"] != 200 or not out["output"]:
                failed.append(f"single/{name}: HTTP {out['status']}")
            elif name != BASE and out["output"] == single[BASE]["output"]:
                failed.append(f"control/{name}: identical to the base model's output")
        failed += [f"mixed/{n}/{t[:12]}: HTTP {o['status']}" for (n, t), o in zip(jobs, mixed, strict=True)
                   if o["status"] != 200 or not o["output"]]
        merged_dir = _merged_checkpoint(args, Path(tmp) / "merged", args.adapters[first])
        merged_config = {**base_config, "model_kwargs": {
            **(base_config.get("model_kwargs") or {}), "model_path_hf": str(merged_dir),
        }}
        path = Path(tmp) / "merged.yaml"
        path.write_text(yaml.safe_dump(merged_config))
        with ManagedServer(args.model, config=str(path), log_path=args.out / f"server_{args.model}_merged.log") as s:
            reference = _generate(s.url, spec_, args, None, TEXTS[0])
        report["vs_merged"] = _similar(spec_, single[first]["output"], reference["output"])
        report["vs_base"] = _similar(spec_, single[first]["output"], single[BASE]["output"])
        if not report["vs_merged"]["close"]:
            failed.append(f"{first}: served output far from the merged checkpoint's ({report['vs_merged']})")
    return _report(args, "serve", report, failed)


def _generate(url: str, spec_: ModelSpec, args, adapter: str | None, text: str) -> dict:
    import base64

    import requests

    kwargs = dict(spec_.request.get("model_kwargs", {}))
    if adapter is not None:
        kwargs["lora_adapter"] = adapter
    data = {"output_modalities": spec_.request["output_modalities"], "model_kwargs": json.dumps(kwargs)}
    files = None
    if spec_.request.get("audio"):
        files = [("files", open(args.audio, "rb"))]  # noqa: SIM115 - requests reads and closes it
    else:
        data["text"] = text
    with requests.post(f"{url}/generate", data=data, files=files, stream=True, timeout=600) as r:
        if r.status_code != 200:
            return {"status": r.status_code, "output": None}
        chunks = [json.loads(line) for line in r.iter_lines() if line]
    modality = spec_.request["output_modalities"]
    parts = [m.get("data", "") for m in chunks if m.get("modality") == modality]
    output = b"".join(base64.b64decode(p) for p in parts) if modality == "audio" else "".join(parts)
    return {"status": 200, "output": output}


def _similar(spec_: ModelSpec, a, b) -> dict:
    """Loose: greedy bf16 decoding may part from a merge after a near tie."""
    if a is None or b is None:
        return {"close": False}
    if spec_.request["output_modalities"] == "text":
        common = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        return {"common_prefix": common, "close": a == b or common >= min(32, len(a), len(b))}
    import numpy as np

    x, y = (np.frombuffer(v[: len(v) // 2 * 2], dtype=np.int16).astype(np.float32) for v in (a, b))
    n = min(len(x), len(y))
    ratio = len(x) / max(len(y), 1)
    corr = float(np.corrcoef(x[:n], y[:n])[0, 1]) if n > 1 else 0.0
    # where the waveforms part (a code token differs): float noise alone stays far below this
    apart = np.nonzero(np.abs(x[:n] - y[:n]) > 1000)[0]
    return {"length_ratio": ratio, "correlation": corr, "close": 0.8 <= ratio <= 1.25 and corr > 0.5,
            "samples": [len(x), len(y)], "first_apart_sample": int(apart[0]) if len(apart) else None}


def _checkpoint_names_of(module_map, module: str) -> list[str]:
    """The names ``module`` may have in the checkpoint (its root and aliases rewritten) that
    land on a target; a merged checkpoint adds the delta to whichever one a shard holds."""
    names = []
    for candidate in module_map._candidates(module):
        found = module_map._native(candidate)
        if found is not None and found[0] in module_map.bindings:
            names.append(candidate)
    return names


def _merged_checkpoint(args, out: Path, adapter_path: str) -> Path:
    """The model's snapshot with the adapter merged into its checkpoint tensors (named as
    the module map reads them), other files linked."""
    from safetensors.torch import load_file, save_file

    from mstar.engine.resources.lora.checkpoint import read_lora_file

    model = _model(args.model)
    spec = _lora_spec(model)
    snapshot = Path(_snapshot(model))
    deltas, alias = {}, {}
    for module, layer in read_lora_file(adapter_path, convert_keys=spec.convert_keys).layers.items():
        names = _checkpoint_names_of(spec.module_map, module)
        if not names:
            raise SystemExit(f"{module}: no checkpoint tensor to merge into")
        a, b = layer.a.float().reshape(layer.a.shape[0], -1), layer.b.float().reshape(layer.b.shape[0], -1)
        deltas[f"{names[0]}.weight"] = (layer.alpha / a.shape[0]) * (b @ a)
        # keyed by the native name the loader reads each checkpoint tensor as, so a
        # checkpoint that names the module otherwise (Chatterbox's `tfmr.` for `backbone.`) matches
        alias.update({k: f"{names[0]}.weight" for k in (spec.module_map.rules.native(f"{n}.weight") for n in names)
                      if k is not None})
    out.mkdir(parents=True, exist_ok=True)
    for path in snapshot.rglob("*"):
        target = out / path.relative_to(snapshot)
        if path.is_dir():
            target.mkdir(exist_ok=True)
            continue
        tensors = load_file(str(path)) if path.suffix == ".safetensors" else {}
        native = spec.module_map.rules.native
        hit = [k for k in tensors if alias.get(native(k)) in deltas]
        if not hit:
            os.symlink(path.resolve(), target)
            continue
        for k in hit:
            delta = deltas.pop(alias[native(k)])
            tensors[k] = (tensors[k].float() + delta.view_as(tensors[k])).to(tensors[k].dtype)
        save_file(tensors, str(target))
    if deltas:
        raise SystemExit(f"merge: {sorted(deltas)[:5]} are in no checkpoint shard")
    return out


def _report(args, phase: str, report: dict, failed: list[str]) -> int:
    path = args.out / f"{args.model}_{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"report": report, "failed": failed}, indent=1, default=str))
    print(f"[{phase}] {len(failed)} failure(s)", *failed, sep="\n  ")
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="orpheus", choices=sorted(MODELS))
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=REPO_DIR_OR_FILE")
    p.add_argument("--synthetic", type=int, default=None, help="random adapters (default 2 without Hub ones)")
    p.add_argument("--synthetic-std", type=float, default=0.02,
                   help="std of the random adapters' A and B (each delta is 2 * B @ A at alpha/r = 2)")
    p.add_argument("--audio", help="serve: a speech clip, for a model that takes speech in")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("all", "merged", "hf", "serve"), default="all")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec_ = MODELS[args.model]()
    chosen = dict(a.split("=", 1) for a in args.adapter) or DEFAULT_ADAPTERS.get(args.model, {})
    args.adapters = {name: _adapter_file(s) for name, s in chosen.items()}
    if not args.adapters or args.synthetic:
        spec = _lora_spec(_model(args.model))
        args.adapters.update(_synthetic(spec, args.out / "adapters", args.synthetic or 2,
                                        std=args.synthetic_std))
    phases = {"merged": merged, "hf": hf, "serve": serve}
    if args.phase != "all":
        return phases[args.phase](args, spec_)
    code = merged(args, spec_)
    if spec_.hf is not None:
        code |= hf(args, spec_)
    if not spec_.request.get("audio") or args.audio:
        code |= serve(args, spec_)
    return code


if __name__ == "__main__":
    sys.exit(main())
