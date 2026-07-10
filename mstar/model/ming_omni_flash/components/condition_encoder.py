"""Ming-flash-omni-2.0 condition encoder for image generation (step 9b).

Native mstar port of vllm-omni's ``condition_encoder.py``. Encodes the thinker
hidden states (sliced at the learnable ``<imagePatch>`` query-token positions)
into the DiT's ``cap_feats`` conditioning:

    thinker hidden states [B, N, 4096]
              │ proj_in (Linear, bias)        -> [B, N, 1536]
              │ Qwen2 connector (bidirectional, non-causal)
              │ proj_out (Linear, bias)       -> [B, N, 2560]
              │ F.normalize(dim=-1) × 1000    (text_encoder_norm)
              ▼
       cap_feats consumed by ZImageTransformer2DModel

Only transformers is required (the connector is a small Qwen2 backbone loaded
via ``Qwen2ForCausalLM.from_pretrained``); there is no diffusers dependency, so
the forward path is unit-testable with a stub connector.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class MingConditionEncoder(nn.Module):
    """Qwen2 connector + proj_in/out + L2-normalize×1000 → DiT condition embeds.

    The connector runs bidirectionally (``is_causal=False``) since it encodes a
    fixed block of query-token hidden states rather than decoding
    autoregressively. ``proj_in`` / ``proj_out`` / connector are populated by
    :meth:`load_from_checkpoint`; before that the module is cheap to construct
    (Identity projections), which keeps dummy-init and unit tests light.

    Args:
        image_gen_config: an ``ImageGenConfig`` (mstar) exposing
            ``connector_subfolder`` / ``mlp_subfolder`` /
            ``diffusion_c_input_dim`` / ``text_encoder_norm`` /
            ``use_identity_mlp``.
        thinker_hidden_size: hidden size of the thinker (BailingMoeV2); 4096 on
            the released checkpoint.
        device / dtype: optional placement applied after loading.
    """

    def __init__(
        self,
        image_gen_config,
        *,
        thinker_hidden_size: int = 4096,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = image_gen_config
        self.thinker_hidden_size = thinker_hidden_size
        self._target_device = torch.device(device) if device is not None else None
        self._target_dtype = dtype

        self.connector: nn.Module | None = None
        self.connector_hidden_size: int | None = None
        self.proj_in: nn.Module = nn.Identity()
        self.proj_out: nn.Module = nn.Identity()
        self.norm: nn.Module = nn.Identity()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_from_checkpoint(self, model_path: str | Path) -> None:
        """Load the Qwen2 connector + proj_in/proj_out weights from disk."""
        from transformers import AutoConfig, Qwen2ForCausalLM

        model_path = Path(model_path)
        connector_path = model_path / self.config.connector_subfolder
        logger.info("[MingConditionEncoder] loading connector from %s", connector_path)

        connector_cfg = AutoConfig.from_pretrained(connector_path, trust_remote_code=True, local_files_only=True)
        connector_cfg.is_decoder = False
        self.connector_hidden_size = int(connector_cfg.hidden_size)

        connector = Qwen2ForCausalLM.from_pretrained(
            connector_path,
            config=connector_cfg,
            torch_dtype=self._target_dtype,
            local_files_only=True,
        )
        # Force bidirectional attention defensively — some transformers versions
        # read ``self_attn.is_causal`` in forward.
        for module in connector.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False

        self.connector = getattr(connector, "model", connector)  # base encoder, no LM head

        self.proj_in = nn.Linear(self.thinker_hidden_size, self.connector_hidden_size, bias=True)
        # text_encoder_norm = L2 normalize on the final cap_feats (NOT an
        # intermediate RMSNorm); applied explicitly in forward(). Keep
        # self.norm as Identity.
        self.norm = nn.Identity()
        self.proj_out = nn.Linear(self.connector_hidden_size, self.config.diffusion_c_input_dim, bias=True)

        mlp_path = model_path / self.config.mlp_subfolder
        mlp_cfg_path = mlp_path / "config.json"
        if mlp_cfg_path.exists() and not json.loads(mlp_cfg_path.read_text()).get("use_identity_mlp", False):
            raise NotImplementedError(f"{mlp_cfg_path} has use_identity_mlp=False; ToClipMLP path not implemented.")
        self._load_optional_mlp_weights(mlp_path)

        if self._target_device is not None:
            self.to(self._target_device)
        if self._target_dtype is not None:
            self.to(dtype=self._target_dtype)

    def _load_optional_mlp_weights(self, mlp_path: Path) -> None:
        """Copy proj_in / proj_out (+ optional norm) weights from ``mlp/``.

        Expected keys (inclusionAI/Ming-flash-omni-2.0): ``proj_in.{weight,bias}``
        [1536,4096]/[1536], ``proj_out.{weight,bias}`` [2560,1536]/[2560], and
        ``query_tokens_dict.16x16`` [256,4096] which is consumed on the thinker
        side (skipped here). Missing proj weights are logged as errors — the
        conditioning is meaningless without them.
        """
        if not mlp_path.exists():
            logger.warning("[MingConditionEncoder] mlp/ missing at %s — proj/norm stay random-init", mlp_path)
            return

        from safetensors.torch import load_file

        candidates = sorted(mlp_path.glob("*.safetensors")) or sorted(mlp_path.glob("*.bin"))
        if not candidates:
            logger.warning("[MingConditionEncoder] no weight files under %s", mlp_path)
            return

        state: dict[str, torch.Tensor] = {}
        for p in candidates:
            if p.suffix == ".safetensors":
                state.update(load_file(str(p)))
            else:
                state.update(torch.load(str(p), map_location="cpu"))

        handled: set[str] = set()

        def _copy(dst: torch.Tensor, src_key: str) -> bool:
            src = state.get(src_key)
            if src is None:
                logger.error("[MingConditionEncoder] mlp/ missing key %r", src_key)
                return False
            if tuple(src.shape) != tuple(dst.shape):
                logger.error(
                    "[MingConditionEncoder] mlp/%s shape mismatch: ckpt=%s module=%s",
                    src_key,
                    tuple(src.shape),
                    tuple(dst.shape),
                )
                return False
            with torch.no_grad():
                dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
            handled.add(src_key)
            return True

        ok = all(
            [
                _copy(self.proj_in.weight, "proj_in.weight"),
                _copy(self.proj_in.bias, "proj_in.bias"),
                _copy(self.proj_out.weight, "proj_out.weight"),
                _copy(self.proj_out.bias, "proj_out.bias"),
            ]
        )
        if not ok:
            logger.error("[MingConditionEncoder] proj_in/proj_out NOT fully loaded; conditioning will be garbage.")

        if "norm.weight" in state and hasattr(self.norm, "weight"):
            _copy(self.norm.weight, "norm.weight")

        for k in state:
            if k.startswith("query_tokens_dict"):
                handled.add(k)  # thinker-side; not loaded here

        leftover = set(state.keys()) - handled
        if leftover:
            logger.warning("[MingConditionEncoder] mlp/ unhandled keys: %s", sorted(leftover))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        thinker_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``[B, N, thinker_hidden_size]`` → ``[B, N, diffusion_c_input_dim]``."""
        if self.connector is None:
            raise RuntimeError("MingConditionEncoder.load_from_checkpoint() must run before forward().")
        if thinker_hidden_states.dim() != 3:
            raise ValueError(f"expected [B, N, H], got shape {tuple(thinker_hidden_states.shape)}")

        b, n, _ = thinker_hidden_states.shape
        x = self.proj_in(thinker_hidden_states)

        # Ming passes a 4D all-ones mask [B, 1, N, N] to force full bidirectional
        # self-attention over the query positions.
        if attention_mask is None:
            attention_mask = torch.ones((b, 1, n, n), dtype=x.dtype, device=x.device)
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask.to(x.dtype)[:, None, None, :].expand(b, 1, n, n)

        out = self.connector(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        cap_feats = self.proj_out(hidden)

        cap_feats = F.normalize(cap_feats, dim=-1)
        if self.config.text_encoder_norm:
            cap_feats = cap_feats * 1000.0
        return cap_feats

    @torch.no_grad()
    def zero_negative(self, cap_feats: torch.Tensor) -> torch.Tensor:
        """Zero tensor shaped like ``cap_feats`` for CFG negatives."""
        return torch.zeros_like(cap_feats)

    def extra_repr(self) -> str:
        return (
            f"thinker_hidden_size={self.thinker_hidden_size}, "
            f"connector_hidden_size={self.connector_hidden_size}, "
            f"diffusion_c_input_dim={self.config.diffusion_c_input_dim}, "
            f"text_encoder_norm={self.config.text_encoder_norm}"
        )


__all__ = ["MingConditionEncoder"]
