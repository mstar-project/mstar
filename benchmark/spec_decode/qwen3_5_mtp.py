"""Qwen3.5 on Hugging Face transformers as the target, and the checkpoint's MTP head as the draft.

transformers' Qwen3.5 model ignores the ``mtp.*`` tensors, so the head is assembled here from the same building blocks
(one full-attention decoder layer, the fc projection and its two input norms, a final norm) and loaded straight from
the safetensors shards of the cached snapshot. It shares the embedding and the lm_head with the base model: the
checkpoints set ``mtp_use_dedicated_embeddings: false`` and ship no ``mtp.lm_head``.

Draft input at position p is fc(cat(norm_e(embed(x_{p+1})), norm_h(h_p))), with h_p the target's hidden state after its
final norm (the tensor its lm_head reads) at position p; a chained draft feeds the head's own output hidden state
back as h. Both conventions are the vLLM / SGLang ones for this head; the CLI's calibration compares them against the
pre-norm hidden state and the other concatenation order on a few prompts before trusting them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

MTP_PREFIX = "mtp."


def snapshot_dir(repo_id: str) -> Path:
    """The cached snapshot of a Hub repo, found without touching the network."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    candidates = [p for p in repo.iterdir() if (p / "config.json").exists()] if repo.exists() else []
    if not candidates:
        raise FileNotFoundError(f"no cached snapshot with config.json for {repo_id} under {hub}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


class BaseModel:
    """The text stack of a loaded checkpoint: the decoder, its lm_head, embedding and rotary module."""

    def __init__(self, model: nn.Module):
        self.model = model
        if hasattr(model.model, "language_model"):  # Qwen3_5ForConditionalGeneration
            self.text = model.model.language_model
        else:  # Qwen3_5ForCausalLM
            self.text = model.model
        self.lm_head = model.lm_head
        self.embed = self.text.embed_tokens
        self.rotary = self.text.rotary_emb
        self.config = self.text.config
        self.device = next(self.text.parameters()).device
        self.dtype = next(self.text.parameters()).dtype


def load_base(repo_id: str, device: str | torch.device, dtype: torch.dtype = torch.bfloat16) -> tuple[BaseModel, Path]:
    snapshot = snapshot_dir(repo_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=dtype, attn_implementation="sdpa")
    except Exception as exc:  # the text-only wrapper is not available for every transformers build
        from transformers import Qwen3_5ForConditionalGeneration

        print(f"AutoModelForCausalLM failed ({exc!r}); loading Qwen3_5ForConditionalGeneration")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    return BaseModel(model), snapshot


def load_tokenizer(snapshot: Path):
    return AutoTokenizer.from_pretrained(snapshot)


class MtpHead(nn.Module):
    """The one-layer multi-token-prediction head of the Qwen3.5 checkpoints."""

    def __init__(self, text_config):
        super().__init__()
        import copy

        config = copy.deepcopy(text_config)
        config.layer_types = ["full_attention"]
        config.num_hidden_layers = 1
        config._attn_implementation = "sdpa"
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden, eps=eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden, eps=eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, 0)])
        self.norm = Qwen3_5RMSNorm(hidden, eps=eps)

    @classmethod
    def from_snapshot(cls, snapshot: Path, text_config, device, dtype: torch.dtype) -> MtpHead:
        weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        by_shard: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            if name.startswith(MTP_PREFIX):
                by_shard.setdefault(shard, []).append(name)
        if not by_shard:
            raise ValueError(f"{snapshot} has no {MTP_PREFIX}* tensors")
        state = {}
        for shard, names in by_shard.items():
            with safe_open(str(snapshot / shard), framework="pt", device="cpu") as f:
                for name in names:
                    state[name[len(MTP_PREFIX):]] = f.get_tensor(name)
        head = cls(text_config)
        head.load_state_dict(state, strict=True)
        return head.to(device=device, dtype=dtype).eval()

    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())


class HFTarget:
    """Qwen3.5 driven one token at a time through transformers' cache. hidden_source selects which hidden state
    the draft sees: "post" (after the final norm) or "pre" (the last decoder layer's output)."""

    def __init__(self, base: BaseModel, hidden_source: str = "post"):
        if hidden_source not in ("post", "pre"):
            raise ValueError(hidden_source)
        self.base = base
        self.hidden_source = hidden_source
        self.cache: DynamicCache | None = None
        self._pre_norm: torch.Tensor | None = None
        self._hook = None
        if hidden_source == "pre":
            self._hook = base.text.norm.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        self._pre_norm = args[0]

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.base.text(input_ids=ids, past_key_values=self.cache, use_cache=True)
        post = out.last_hidden_state[0]
        hidden = self._pre_norm[0] if self.hidden_source == "pre" else post
        logits = self.base.lm_head(post[-1]).float()
        return logits, hidden

    @torch.inference_mode()
    def prefill(self, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        self.cache = DynamicCache(config=self.base.config)
        ids = torch.tensor([tokens], dtype=torch.long, device=self.base.device)
        return self._forward(ids)

    @torch.inference_mode()
    def step(self, token: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor([[token]], dtype=torch.long, device=self.base.device)
        logits, hidden = self._forward(ids)
        return logits, hidden[-1]


class HFDraft:
    """The MTP head as a draft model with its own KV cache. fc_order "emb_first" concatenates the normed embedding
    before the normed hidden state; "hidden_first" the other way round."""

    def __init__(self, head: MtpHead, base: BaseModel, fc_order: str = "emb_first"):
        if fc_order not in ("emb_first", "hidden_first"):
            raise ValueError(fc_order)
        self.head = head
        self.base = base
        self.fc_order = fc_order
        self.cache = DynamicCache()
        self.length = 0
        self.last_hidden: torch.Tensor | None = None

    def reset(self) -> None:
        self.cache = DynamicCache()
        self.length = 0
        self.last_hidden = None

    @torch.inference_mode()
    def _run(self, hiddens: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[0]
        emb = self.head.pre_fc_norm_embedding(self.base.embed(tokens))
        hid = self.head.pre_fc_norm_hidden(hiddens.to(self.base.dtype))
        parts = (emb, hid) if self.fc_order == "emb_first" else (hid, emb)
        x = self.head.fc(torch.cat(parts, dim=-1))[None]  # (1, n, hidden)
        positions = torch.arange(self.length, self.length + n, device=x.device)
        cos, sin = self.base.rotary(x, positions.view(1, 1, n).expand(3, 1, n))
        mask = None
        if n > 1 and self.length > 0:
            # Several new queries over an existing cache: sdpa's is_causal flag aligns top-left, so spell the mask
            # out, as an additive mask so that the eager attention path reads it too.
            total = self.length + n
            allowed = (torch.arange(total, device=x.device)[None, :] <= positions[:, None]).view(1, 1, n, total)
            mask = torch.zeros(allowed.shape, device=x.device, dtype=x.dtype)
            mask.masked_fill_(~allowed, torch.finfo(x.dtype).min)
        y = self.head.layers[0](
            x,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            position_ids=positions[None],
            past_key_values=self.cache,
            use_cache=True,
        )
        g = self.head.norm(y)[0]
        self.length += n
        self.last_hidden = g[-1]
        return self.base.lm_head(g[-1]).float()

    def extend(self, hiddens: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self._run(hiddens.to(self.base.device), tokens.to(self.base.device))

    def chain(self, token: int) -> torch.Tensor:
        tokens = torch.tensor([token], dtype=torch.long, device=self.base.device)
        return self._run(self.last_hidden[None], tokens)

    def rollback(self, n: int) -> None:
        if n <= 0:
            return
        if n > self.length:
            raise ValueError(f"rollback of {n} positions from a cache of {self.length}")
        self.cache.crop(-n)
        self.length -= n
