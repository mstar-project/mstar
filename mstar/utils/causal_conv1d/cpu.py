"""Torch stand-ins for the two conv kernels on a CPU tensor, with their signatures and
results, so a recurrent resource's paths run in a test without a GPU. Like the kernels,
they read the input in the conv state's dtype and update the state pool in place."""

import torch
import torch.nn.functional as F


def _activate(out: torch.Tensor, activation: str | None) -> torch.Tensor:
    return F.silu(out) if activation in ("silu", "swish") else out


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = -1,
    null_block_id: int = 0,
    **unused,
) -> torch.Tensor:
    """``x`` ``[dim, tokens]``; each sequence of ``query_start_loc`` from its slot's window (zeros
    unless ``has_initial_state``), the slot's window left as the sequence's last inputs. A sequence
    on ``pad_slot_id`` or ``null_block_id`` is skipped. The kernel writes its output into
    ``empty_like`` memory, so tokens no sequence covers come back ``inf`` here rather than as
    found."""
    assert unused.get("block_idx_first_scheduled_token") is None and unused.get("initial_state_idx") is None, (
        "the CPU conv takes no prefix-cache block pointers"
    )
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    dtype = x.dtype
    x = x.to(conv_states.dtype)
    out = torch.full_like(x, float("inf"))
    taps = weight.shape[1] - 1
    bounds = query_start_loc.tolist()
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        slot = i if cache_indices is None else int(cache_indices[i])
        if slot in (pad_slot_id, null_block_id) and cache_indices is not None:
            continue
        carried = has_initial_state is not None and bool(has_initial_state[i])
        window = conv_states[slot] if carried else conv_states.new_zeros(x.shape[0], taps)
        seq = torch.cat([window, x[:, start:end]], dim=1)
        y = F.conv1d(seq.float()[None], weight.float()[:, None], bias.float() if bias is not None else None,
                     groups=x.shape[0])[0]
        out[:, start:end] = _activate(y, activation).to(out.dtype)
        conv_states[slot] = seq[:, seq.shape[1] - taps:]
    return out.to(dtype)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    null_block_id: int = 0,
    out: torch.Tensor | None = None,
    **unused,
) -> torch.Tensor:
    """One token a row, ``x`` ``[batch, dim]``: each row's slot window shifted by its token, in
    place; a row on ``null_block_id`` is left as found. Written into ``out``, or ``x``."""
    assert unused.get("query_start_loc") is None and unused.get("num_accepted_tokens") is None, (
        "the CPU conv update takes one token a row"
    )
    assert x.dim() == 2, "the CPU conv update takes x [batch, dim]"
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    dtype = x.dtype
    x = x.to(conv_state.dtype)
    if out is None:
        out = x
    for row in range(x.shape[0]):
        slot = row if conv_state_indices is None else int(conv_state_indices[row])
        if conv_state_indices is not None and slot == null_block_id:
            continue
        seq = torch.cat([conv_state[slot], x[row, :, None]], dim=1)
        y = (seq.float() * weight.float()).sum(-1)
        if bias is not None:
            y = y + bias.float()
        out[row] = _activate(y, activation).to(out.dtype)
        conv_state[slot] = seq[:, 1:]
    return out.to(dtype)
