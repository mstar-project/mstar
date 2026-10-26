"""Transducer (RNN-T / TDT) head and batched greedy decoding, as NeMo decodes.

A joint network scores the next token or blank from the prediction network and
the encoder frame under each request's cursor; TDT also predicts how many
frames to skip. The loop steps per emitted token, as NeMo's label looping does:
the prediction network only changes on an emission, so a step scores a
window of frames at once and runs the LSTM once, for the first frame that
emits. ``advance`` runs ``CHUNK`` steps on a batch's state; the transducer
node runs it once per iteration of its decode loop, with each request's state
kept in a recurrent state slot between iterations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.recurrent.config import RecurrentBlockConfig

# greedy steps per decode iteration: one graph replay each
CHUNK = 16
# frames an RNN-T step scores at once: wider windows cost more joint work, over
# a 13k vocabulary, than the blank steps they save. TDT scores one: its
# durations already skip most blank frames.
RNNT_WINDOW = 4


@dataclass
class TransducerConfig:
    encoder_hidden_size: int = 1024
    hidden_size: int = 640
    num_layers: int = 2
    vocab_size: int = 8193  # including blank
    blank_token_id: int = 8192
    start_token_id: int = 8192
    # TDT skip options; empty for plain RNN-T
    durations: list[int] = field(default_factory=list)
    max_symbols_per_step: int = 10
    # language prompt (Nemotron ASR): a one-hot slot per frame, mixed in by an
    # MLP before the encoder projection; 0 for none
    num_prompts: int = 0
    prompt_intermediate_size: int = 0

    @classmethod
    def from_hf(cls, hf: dict, start_token_id: int) -> TransducerConfig:
        return cls(
            encoder_hidden_size=hf["encoder_config"]["hidden_size"],
            hidden_size=hf["decoder_hidden_size"],
            num_layers=hf["num_decoder_layers"],
            vocab_size=hf["vocab_size"],
            blank_token_id=hf["blank_token_id"],
            start_token_id=start_token_id,
            durations=list(hf.get("durations") or []),
            max_symbols_per_step=hf.get("max_symbols_per_step", 10),
            num_prompts=hf.get("num_prompts", 0),
            prompt_intermediate_size=hf.get("prompt_intermediate_size", 0),
        )

    def state_blocks(self) -> dict[str, RecurrentBlockConfig]:
        """One request's decoding state between iterations, as recurrent pool
        blocks: the LSTM state and output in fp32, the cursor and the tokens
        emitted at its frame."""
        layers = (self.num_layers, self.hidden_size)
        return {
            "h": RecurrentBlockConfig(layers, torch.float32),
            "c": RecurrentBlockConfig(layers, torch.float32),
            "dec_out": RecurrentBlockConfig((self.hidden_size,), torch.float32),
            "cursor": RecurrentBlockConfig((), torch.int64),
            "symbols": RecurrentBlockConfig((), torch.int64),
        }

    def decode_iterations(self, num_frames: int) -> int:
        """The most decode iterations a clip of ``num_frames`` can take: a frame
        takes at most ``max_symbols_per_step`` steps, so this never binds."""
        return max(1, math.ceil(self.max_symbols_per_step * num_frames / CHUNK))


class LSTMWeights(nn.Module):
    """``nn.LSTM``'s parameters under its names, without the module, which
    torch.compile refuses to trace; ``PredictionNetwork`` writes the cell out."""

    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        for layer in range(num_layers):
            for kind, shape in (("weight", (4 * hidden_size, hidden_size)), ("bias", (4 * hidden_size,))):
                for gate in ("ih", "hh"):
                    self.register_parameter(f"{kind}_{gate}_l{layer}", nn.Parameter(torch.empty(shape)))


class PredictionNetwork(nn.Module):
    """Embedding + LSTM + projection over the emitted tokens; HF's
    ``decoder.*`` names."""

    def __init__(self, config: TransducerConfig):
        super().__init__()
        h = config.hidden_size
        self.embedding = nn.Embedding(config.vocab_size, h)
        self.lstm = LSTMWeights(h, config.num_layers)
        self.decoder_projector = nn.Linear(h, h)

    def forward(
        self, tokens: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step for ``(B,)`` tokens with ``(B, layers, hidden)`` state ->
        ``(B, hidden)`` and the new state: ``nn.LSTM``'s cell, written out so
        torch.compile can trace it."""
        x = self.embedding(tokens)
        hs, cs = [], []
        for layer in range(self.lstm.num_layers):
            gates = F.linear(x, getattr(self.lstm, f"weight_ih_l{layer}"), getattr(self.lstm, f"bias_ih_l{layer}"))
            gates = gates + F.linear(
                h[:, layer], getattr(self.lstm, f"weight_hh_l{layer}"), getattr(self.lstm, f"bias_hh_l{layer}"),
            )
            i, f, g, o = gates.chunk(4, dim=-1)
            cell = torch.sigmoid(f) * c[:, layer] + torch.sigmoid(i) * torch.tanh(g)
            x = torch.sigmoid(o) * torch.tanh(cell)
            hs.append(x)
            cs.append(cell)
        return self.decoder_projector(x), torch.stack(hs, dim=1), torch.stack(cs, dim=1)


class PromptProjector(nn.Module):
    def __init__(self, config: TransducerConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.encoder_hidden_size + config.num_prompts, config.prompt_intermediate_size)
        self.linear_2 = nn.Linear(config.prompt_intermediate_size, config.encoder_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.relu(self.linear_1(x)))


class TransducerHead(nn.Module):
    """Encoder projection, prediction network and joint (and the language
    prompt's projector, where the checkpoint has one); parameter names match
    HF's ``ParakeetForTDT`` / ``Nemotron3_5AsrForRNNT``."""

    def __init__(self, config: TransducerConfig):
        super().__init__()
        self.config = config
        self.prompt_projector = PromptProjector(config) if config.num_prompts else None
        self.encoder_projector = nn.Linear(config.encoder_hidden_size, config.hidden_size)
        self.decoder = PredictionNetwork(config)
        self.joint = nn.Module()
        self.joint.head = nn.Linear(config.hidden_size, config.vocab_size + len(config.durations))
        # TDT's skip table; a buffer so it follows the head to its device
        self.register_buffer("durations", torch.tensor(config.durations, dtype=torch.long), persistent=False)
        self.window = 1 if config.durations else RNNT_WINDOW

    def project_frames(self, encoder_states: torch.Tensor, prompt_ids: torch.Tensor | None) -> torch.Tensor:
        if self.prompt_projector is not None:
            one_hot = F.one_hot(prompt_ids, self.config.num_prompts).to(encoder_states.dtype)
            one_hot = one_hot[:, None, :].expand(-1, encoder_states.shape[1], -1)
            encoder_states = self.prompt_projector(torch.cat([encoder_states, one_hot], dim=-1))
        return self.encoder_projector(encoder_states)

    @torch.no_grad()
    def initial_state(self) -> dict[str, torch.Tensor]:
        """A request's state before its first step, one tensor per
        ``state_blocks`` entry: the prediction network after the start token,
        the cursor at frame 0. The same for every request."""
        cfg = self.config
        device = self.encoder_projector.weight.device
        zeros = torch.zeros(1, cfg.num_layers, cfg.hidden_size, device=device)
        dec_out, h, c = self.decoder(torch.tensor([cfg.start_token_id], device=device), zeros, zeros)
        count = torch.zeros((), dtype=torch.long, device=device)
        return {"h": h[0], "c": c[0], "dec_out": dec_out[0], "cursor": count, "symbols": count.clone()}

    def step(self, s: dict[str, torch.Tensor]) -> torch.Tensor:
        """One greedy step on every row of ``advance``'s state, in place and
        with no host reads: each active row emits one token or moves past a
        window of blanks -> the step's tokens (-1 where a row emitted none)."""
        cfg = self.config
        frames, cursor, symbols, active, lengths = s["frames"], s["cursor"], s["symbols"], s["active"], s["lengths"]
        rows, t, hidden = frames.shape
        width = self.window
        offsets = torch.arange(width, device=frames.device)
        at = cursor[:, None] + offsets
        valid = at < lengths[:, None]
        window = frames.gather(1, at.clamp(max=t - 1)[:, :, None].expand(-1, -1, hidden))
        logits = self.joint.head(F.relu(window + s["dec_out"][:, None]))
        tokens = logits[..., :cfg.vocab_size].argmax(dim=-1)
        blank = tokens == cfg.blank_token_id
        if cfg.durations:
            skips = self.durations[logits[..., cfg.vocab_size:].argmax(dim=-1)]
        else:
            skips = torch.zeros_like(tokens)
        # where each frame leads: a blank moves on by its skip (at least one
        # frame), an emission or the clip's end stays put, and the slots past
        # the window keep whatever reaches them
        reach = max(cfg.durations, default=1) or 1
        nxt = torch.where(blank & valid, offsets + skips.clamp(min=1), offsets)
        nxt = torch.cat([nxt, torch.arange(width, width + reach, device=frames.device).expand(rows, -1)], dim=1)
        # pointer doubling: each round doubles the blanks a chain has followed
        for _ in range(math.ceil(math.log2(width))):
            nxt = nxt.gather(1, nxt)
        first = nxt[:, 0]
        at_first = first.clamp(max=width - 1)[:, None]
        emit = active & (first < width) & (valid & ~blank).gather(1, at_first)[:, 0]
        token = tokens.gather(1, at_first)[:, 0]
        skip = skips.gather(1, at_first)[:, 0]
        # NeMo's guard: after max_symbols tokens at one frame, move on. A blank
        # before the emission starts the frame's count over.
        count = torch.where(first > 0, 0, symbols)
        count = torch.where(emit & (skip == 0), count + 1, 0)
        forced = count >= cfg.max_symbols_per_step
        skip = skip.masked_fill(forced, 1)
        symbols.copy_(count.masked_fill(forced, 0))
        # every row steps the LSTM and only emitting rows keep it: cheaper
        # than asking the host whether any row emitted
        new_out, h, c = self.decoder(token, s["h"], s["c"])
        s["dec_out"].copy_(torch.where(emit[:, None], new_out, s["dec_out"]))
        s["h"].copy_(torch.where(emit[:, None, None], h, s["h"]))
        s["c"].copy_(torch.where(emit[:, None, None], c, s["c"]))
        cursor.add_(torch.where(active, torch.where(emit, first + skip, first), 0))
        active.copy_(cursor < lengths)
        return torch.where(emit, token, -1)

    def advance(
        self, frames: torch.Tensor, lengths: torch.Tensor, state: dict[str, torch.Tensor],
        chunk: int = CHUNK, step=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``chunk`` greedy steps (``step``, by default :meth:`step`) over
        ``(B, T, hidden)`` projected frames and per-row frame counts, updating
        ``state`` (``state_blocks``' names, batch-first) in place -> the steps'
        ``(B, chunk)`` tokens and which rows are still decoding."""
        step = step or self.step
        s = {**state, "frames": frames, "lengths": lengths, "active": state["cursor"] < lengths}
        tokens = torch.stack([step(s) for _ in range(chunk)], dim=1)
        return tokens, s["active"]
