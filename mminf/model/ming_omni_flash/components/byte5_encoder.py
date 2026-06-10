"""ByT5 glyph/text encoder for Ming-flash-omni-2.0 image generation.

Native mminf port of vllm-omni's ``byte5_encoder.py``. Bundles the byt5
tokenizer + HF T5 encoder + :class:`T5EncoderBlockByT5Mapper`. The released
checkpoint's byt5 weights were trained with per-language font/color special
tokens, so we replicate that vocabulary extension before loading — otherwise
``byt5_model.pt`` shape-mismatches at the embedding table.

Typical forward: a list of prompt strings (optionally carrying
``<cn-font-N>`` / ``<color-N>`` markers) → ``[B, byt5_max_length,
diffusion_c_input_dim]`` features, padded positions zeroed so the downstream
``torch.cat`` onto cap_feats injects no garbage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from mminf.model.ming_omni_flash.components.t5_block_mapper import (
    T5EncoderBlockByT5Mapper,
)

logger = logging.getLogger(__name__)


def _add_multilingual_special_tokens(
    tokenizer,
    text_encoder: nn.Module,
    font_ann_path: Path,
    color_ann_path: Path,
    add_font: bool,
    add_color: bool,
    add_align: bool = False,
) -> None:
    """Extend the byt5 vocab with per-language font + color markers.

    Mirrors ``add_special_token_multilingual`` in Ming's bizgen utils. The token
    set must match what the checkpoint was trained with, otherwise the resized
    embedding table won't line up with the shipped weights.
    """
    idx_font_dict = json.loads(Path(font_ann_path).read_text())
    idx_color_dict = json.loads(Path(color_ann_path).read_text())

    font_tokens: list[str] = []
    for font_code in idx_font_dict:
        prefix = font_code[:3]
        if prefix in ("cn-", "en-", "jp-", "kr-"):
            font_tokens.append(f"<{prefix}font-{idx_font_dict[font_code]}>")
        else:
            font_tokens.append(f"<font-{idx_font_dict[font_code]}>")
    color_tokens = [f"<color-{i}>" for i in range(len(idx_color_dict))]
    align_tokens = [f"<align-{i}>" for i in range(3)]

    extra: list[str] = []
    if add_color:
        extra += color_tokens
    if add_font:
        extra += font_tokens
    if add_align:
        extra += align_tokens
    tokenizer.add_tokens(extra, special_tokens=True)
    text_encoder.resize_token_embeddings(len(tokenizer))


class MingByT5Encoder(nn.Module):
    """Bundles byt5 tokenizer + T5 encoder + :class:`T5EncoderBlockByT5Mapper`.

    Build via :meth:`from_checkpoint` when the checkpoint ships byt5 weights;
    otherwise callers can skip this and the pipeline falls back to no-byt5
    conditioning.
    """

    def __init__(
        self,
        tokenizer,
        text_encoder: nn.Module,
        mapper: T5EncoderBlockByT5Mapper,
        max_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.mapper = mapper
        self.max_length = max_length

    @classmethod
    def from_checkpoint(
        cls,
        byte5_dir: Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MingByT5Encoder:
        """Load tokenizer + encoder + mapper from the checkpoint's ``byt5`` dir.

        Wrapped in ``torch.random.fork_rng`` so any ``nn.init`` inside
        ``from_pretrained`` / vocab-resize cannot advance the default generator
        — otherwise the diffusion pipeline's seeded noise becomes
        order-dependent across requests (same-seed replays would diverge).
        """
        cuda_devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devs, enabled=True):
            return cls._from_checkpoint_impl(byte5_dir, device=device, dtype=dtype)

    @classmethod
    def _from_checkpoint_impl(
        cls,
        byte5_dir: Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MingByT5Encoder:
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        byte5_dir = Path(byte5_dir)
        # Ming checkpoint uses ``byt5`` (no 'e') in filenames and JSON keys;
        # the ``byte5_`` variable spelling below is kept for readability.
        cfg_raw = json.loads((byte5_dir / "byt5.json").read_text())
        cfg = SimpleNamespace(**cfg_raw)
        byte5_config = cfg.byt5_config
        mapper_config = cfg.byt5_mapper_config
        max_length = int(cfg.byt5_max_length)

        # ---- Tokenizer + T5 encoder (base).
        ckpt_key = byte5_config.get("byt5_ckpt_path")
        byte5_ckpt_path = byte5_dir / ckpt_key.lstrip("./")
        tokenizer = AutoTokenizer.from_pretrained(byte5_ckpt_path, local_files_only=True)
        text_encoder = T5ForConditionalGeneration.from_pretrained(
            byte5_ckpt_path, local_files_only=True
        ).get_encoder()

        # ---- Extend vocab with font/color markers so the shipped weights load.
        if byte5_config.get("special_token"):
            if not byte5_config.get("multilingual", True):
                raise NotImplementedError(
                    "Non-multilingual byt5 vocab extension is not ported; "
                    "the released Ming checkpoint uses multilingual=True."
                )
            _add_multilingual_special_tokens(
                tokenizer,
                text_encoder,
                font_ann_path=byte5_dir / byte5_config["font_ann_path"].lstrip("./"),
                color_ann_path=byte5_dir / byte5_config["color_ann_path"].lstrip("./"),
                add_font=bool(byte5_config.get("font_special_token")),
                add_color=bool(byte5_config.get("color_special_token")),
            )

        # ---- Load byt5 text-encoder weights. base.pt wraps the backbone in a
        # trainable-module container (module.text_tower.encoder.*); byt5_model.pt
        # carries the top-level encoder state. Follow Ming's two-step load.
        base_state = torch.load(byte5_dir / "byt5_model" / "base.pt", map_location="cpu", weights_only=False)
        prefix = "module.text_tower.encoder."
        base_filtered = {
            name[len(prefix):]: state
            for name, state in base_state["state_dict"].items()
            if name.startswith(prefix)
        }
        text_encoder.load_state_dict(base_filtered, strict=True)
        del base_state, base_filtered

        encoder_state = torch.load(byte5_dir / "byt5_model" / "byt5_model.pt", map_location="cpu", weights_only=False)
        text_encoder.load_state_dict(encoder_state)
        del encoder_state

        text_encoder.to(device=device, dtype=dtype).eval()

        # ---- Mapper (stock HF T5Block layout ⇒ direct state_dict load).
        mapper = T5EncoderBlockByT5Mapper(
            byte5_config=text_encoder.config,
            num_layers=int(mapper_config["num_layers"]),
            sdxl_channels=int(mapper_config["sdxl_channels"]),
        )
        mapper_state = torch.load(byte5_dir / "byt5_mapper" / "byt5_mapper.pt", map_location="cpu", weights_only=False)
        mapper.load_weights(mapper_state.items())
        del mapper_state
        mapper.to(device=device, dtype=dtype).eval()

        logger.info(
            "[MingByT5Encoder] ready: d_model=%d mapper_layers=%d sdxl_channels=%d max_length=%d vocab=%d",
            text_encoder.config.d_model,
            mapper_config["num_layers"],
            mapper_config["sdxl_channels"],
            max_length,
            len(tokenizer),
        )
        return cls(tokenizer, text_encoder, mapper, max_length)

    @torch.inference_mode()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """Tokenize → T5 encode → mapper; zeroes padded positions.

        Returns ``[B, max_length, sdxl_channels]``.
        """
        device = next(self.text_encoder.parameters()).device
        dtype = next(self.text_encoder.parameters()).dtype

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        encoder_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask.float(),
        )
        hidden_states = encoder_out[0]
        feats = self.mapper(hidden_states, attention_mask)
        feats = feats * attention_mask.unsqueeze(-1).to(dtype=feats.dtype)
        return feats.to(dtype=dtype)


__all__ = ["MingByT5Encoder"]
