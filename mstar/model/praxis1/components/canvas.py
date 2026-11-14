"""The observation canvas: camera views to the single image the VAE encodes.

The checkpoint was trained on one canvas per timestep, built by laying the
rig's views side by side at a fixed tile size. Which pixels reach the model
is part of the weights' contract, so this is a port of the reference rule
rather than a reasonable resize: a different crop or a different rounding is
a different observation, and the policy's output moves with it.

Two rules exist, chosen by ``video.antialias`` of the checkpoint config, and
they crop to *different* aspects:

* off: centre-crop to the **tile** aspect, one bilinear resize to the tile,
  and a single round at the end over the whole stack.
* on (the fit rule): centre-crop to the **native view** aspect, one
  antialiased resize to the fit size, rounding **per chunk**, then a centre
  trim to the tile. A view already at the fit size is not resized at all.

The rounding difference is not cosmetic. The off rule rounds once after the
frames are concatenated; the fit rule rounds each chunk back to uint8 before
concatenating, so a frame near a .5 boundary can land on a different level
under the two. Keep each where the reference put it.

The two rigs do not share the fit rule. Single-arm fits against one hardcoded
native view; bimanual fits each view against its own native size, forces the
views to a common height and trims to the patch grid, and refuses a view that
does not land exactly on the tile.
"""

from __future__ import annotations

import torch

# Frames per resize call, which bounds the float32 memory of the resize.
TILE_CHUNK_FRAMES = 32

# The aspect tolerance within which a centre crop is a no-op.
ASPECT_TOL = 0.03

# The spatial patch of the model (VAE stride 16 times patch 2). The bimanual
# fit rule trims every view to this grid.
SPATIAL_PATCH = 32

# The longest side the bimanual fit rule scales a view into.
RESTRICT_SIDE = 416


def aspect_crop_box(
    src_h: int, src_w: int, tile_h: int, tile_w: int, tol: float = ASPECT_TOL
) -> tuple[int, int, int, int]:
    """The centre crop ``(y0, y1, x0, x1)`` taking ``src`` to the tile aspect.

    The box is the whole frame when the two aspects already agree within
    ``tol``, which is what keeps a view the client already restricted from
    being resampled a second time.
    """
    src_ar, tile_ar = src_w / src_h, tile_w / tile_h
    if abs(tile_ar / src_ar - 1.0) <= tol:
        return 0, src_h, 0, src_w
    if src_ar > tile_ar:
        new_w = round(src_h * tile_ar)
        x0 = (src_w - new_w) // 2
        return 0, src_h, x0, x0 + new_w
    new_h = round(src_w / tile_ar)
    y0 = (src_h - new_h) // 2
    return y0, y0 + new_h, 0, src_w


def aspect_crop(
    frames: torch.Tensor, tile_h: int, tile_w: int, tol: float = ASPECT_TOL
) -> torch.Tensor:
    """Apply :func:`aspect_crop_box` to a ``(..., H, W)`` frame stack."""
    y0, y1, x0, x1 = aspect_crop_box(
        frames.shape[-2], frames.shape[-1], tile_h, tile_w, tol
    )
    if (y0, y1, x0, x1) == (0, frames.shape[-2], 0, frames.shape[-1]):
        return frames
    return frames[..., y0:y1, x0:x1]


def _resize(
    frames: torch.Tensor, size: tuple[int, int], antialias: bool
) -> torch.Tensor:
    """Bilinear resize of a uint8 stack in float32, left in float32."""
    return torch.nn.functional.interpolate(
        frames.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=antialias,
    )


def single_arm_fit_size(tile_w: int, native: tuple[int, int]) -> tuple[int, int]:
    """The fit resize size for ``tile_w``-wide tiles on the single-arm rig.

    The native view scaled down so its width is the tile's, height floored:
    ``(234, 416)`` for 416-wide tiles. It is also the size the client is asked
    to restrict each view to, which is why the resize is usually skipped.
    """
    nh, nw = native
    return min(nh, tile_w, nh * tile_w // nw), min(nw, tile_w)


def _fit_view_single_arm(
    frames: torch.Tensor,
    tile_h: int,
    tile_w: int,
    native: tuple[int, int],
    tol: float,
) -> torch.Tensor:
    """The fit-rule tile ``(T, 3, tile_h, tile_w)`` of one view, uint8."""
    rh, rw = single_arm_fit_size(tile_w, native)
    if rh < tile_h or rw < tile_w:
        raise ValueError(f"the fit size {rh}x{rw} is below the tile {tile_h}x{tile_w}")
    # The crop is to the *native* aspect here, not the tile's.
    cropped = aspect_crop(frames, *native, tol)
    if tuple(cropped.shape[-2:]) != (rh, rw):
        # Rounded per chunk, before the concatenation, as the reference does.
        cropped = torch.cat(
            [
                _resize(chunk, (rh, rw), antialias=True)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                for chunk in cropped.split(TILE_CHUNK_FRAMES, dim=0)
            ]
        )
    y0, x0 = (rh - tile_h) // 2, (rw - tile_w) // 2
    return cropped[..., y0 : y0 + tile_h, x0 : x0 + tile_w].contiguous()


def tile_view_single_arm(
    frames: torch.Tensor,
    tile_h: int,
    tile_w: int,
    native: tuple[int, int] = (720, 1280),
    antialias: bool = True,
    tol: float = ASPECT_TOL,
) -> torch.Tensor:
    """One view's uint8 tile ``(T, 3, tile_h, tile_w)``.

    ``frames`` is a uint8 ``(T, 3, H, W)`` stack. The two branches differ in
    the aspect they crop to and in where they round; see the module docstring.
    """
    if frames.dim() == 3:
        frames = frames[None]
    if frames.dim() != 4 or frames.shape[1] != 3:
        raise ValueError(f"frames must be (T, 3, H, W), got {tuple(frames.shape)}")
    if antialias:
        return _fit_view_single_arm(frames, tile_h, tile_w, native, tol)
    cropped = aspect_crop(frames, tile_h, tile_w, tol)
    # Rounded once, after the concatenation.
    out = torch.cat(
        [
            _resize(chunk, (tile_h, tile_w), antialias=False)
            for chunk in cropped.split(TILE_CHUNK_FRAMES, dim=0)
        ]
    )
    return out.round().clamp(0, 255).to(torch.uint8)


def bimanual_fit_size(
    h: int, w: int, max_side: int = RESTRICT_SIDE
) -> tuple[int, int]:
    """``(floor(h*s), floor(w*s))`` at ``s = min(1, max_side/h, max_side/w)``.

    The view scaled down to fit ``max_side`` with its aspect kept, never
    scaled up.
    """
    if h < 1 or w < 1 or max_side < 1:
        raise ValueError(f"bad fit: {h}x{w} into {max_side}")
    return min(h, max_side, (h * max_side) // w), min(w, max_side, (w * max_side) // h)


def bimanual_row_sizes(
    native: list[tuple[int, int]],
    max_side: int = RESTRICT_SIDE,
    patch: int = SPATIAL_PATCH,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The resize sizes and the trimmed sizes of one row of views.

    The views share one height, the smallest fitted height among them, and a
    taller view is resized to it with its aspect kept. Each resized view is
    then trimmed to the patch grid. This is the fit rule of the training
    canvas, and unlike the single-arm rule it never crops.
    """
    if not native:
        raise ValueError("the fit rule needs at least one view")
    fitted = [bimanual_fit_size(h, w, max_side) for h, w in native]
    common_h = min(rh for rh, _ in fitted)
    resize = [
        (common_h, (w * common_h) // h) if rh > common_h else (rh, rw)
        for (h, w), (rh, rw) in zip(native, fitted, strict=True)
    ]
    trimmed = []
    for rh, rw in resize:
        gh, gw = rh // patch * patch, rw // patch * patch
        if gh < patch or gw < patch:
            raise ValueError(
                f"a view fits to {rh}x{rw}: less than one {patch}-px patch"
            )
        trimmed.append((gh, gw))
    return resize, trimmed


def tile_views_bimanual(
    views: list[torch.Tensor], tile_h: int, tile_w: int
) -> list[torch.Tensor]:
    """The bimanual rig's tiles, one per view, uint8 ``(T, 3, tile_h, tile_w)``.

    Refuses a view whose fit does not land exactly on the tile: the rig serves
    16:10 views, and a view of another shape is a misconfigured camera rather
    than something to crop into place.
    """
    stacks = [v[None] if v.dim() == 3 else v for v in views]
    native = [(v.shape[-2], v.shape[-1]) for v in stacks]
    resize, trimmed = bimanual_row_sizes(native)
    out = []
    for stack, (rh, rw), (gh, gw) in zip(stacks, resize, trimmed, strict=True):
        if (gh, gw) != (tile_h, tile_w):
            raise ValueError(
                f"a {stack.shape[-2]}x{stack.shape[-1]} view fits to {gh}x{gw}, "
                f"not the tile {tile_h}x{tile_w}; praxis serves 16:10 views"
            )
        if tuple(stack.shape[-2:]) != (rh, rw):
            stack = torch.cat(
                [
                    _resize(chunk, (rh, rw), antialias=True)
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)
                    for chunk in stack.split(TILE_CHUNK_FRAMES, dim=0)
                ]
            )
        out.append(stack[..., :gh, :gw].contiguous())
    return out


def build_canvas(tiles: list[torch.Tensor]) -> torch.Tensor:
    """The tiles laid side by side as the VAE's input.

    Returns ``(1, 3, T, H, W_total)`` float32 holding integer values in
    ``[0, 255]``. The VAE takes it to ``[-1, 1]`` itself, so this does not
    normalize: doing it here would halve the range twice over.
    """
    if not tiles:
        raise ValueError("a canvas needs at least one view")
    stacks = [t[None] if t.dim() == 3 else t for t in tiles]
    canvas = torch.cat(stacks, dim=-1).float()
    return canvas.permute(1, 0, 2, 3).unsqueeze(0)
