"""Shared ``POST /v1/actions/generations`` (``application/msgpack``) handler.

Serves robot action policies (Pi0.5) and action-capable world models (Cosmos3)
over ``openpi_client.msgpack_numpy`` framing with exact parity against the Rust
axum frontend (``rust/server/src/actions.rs``).
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from mstar.api_server import media_io, openpi
from mstar.api_server.openai._util import error_type, rid
from mstar.api_server.rust_frontend import actions_facts
from mstar.utils.client_errors import exception_status_text

logger = logging.getLogger(__name__)

MAX_ACTION_BATCH_SIZE = 8
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_ROLLOUT_KEYS = frozenset({"sde_type", "noise_level", "sde_step", "return_state", "ignore_last", "video_solver"})
_RUNTIME_KEYS = frozenset({"response_format", "output_format"})
_IMAGE_INPUT_KEYS = ("images", "pixel_values", "camera_images")


@dataclass
class _RowSubmission:
    text: str | None
    file_paths: dict[str, list[str]] | None
    raw_inputs: dict[str, list[torch.Tensor]] | None
    input_modalities: list[str]
    output_modalities: list[str]
    model_kwargs: dict


def _error(status: int, message: str, type_: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_ or error_type(status), "code": status}},
    )


def _check_int64(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    if isinstance(value, np.integer):
        value = int(value)
    elif type(value) is not int:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise ValueError(f"{name} must fit in a signed 64-bit integer")
    return value


def _check_bool(value, name: str) -> bool:
    if isinstance(value, np.bool_):
        return bool(value)
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean, got {type(value).__name__}")
    return value


def _check_finite_number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a number, not bool")
    if isinstance(value, (np.integer, np.floating)):
        value = float(value)
    elif type(value) not in (int, float):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _to_json_value(value, path: str):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return _check_int64(value, path)
    if type(value) is int:
        return _check_int64(value, path)
    if isinstance(value, (float, np.floating)):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"{path} must be finite")
        return out
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError(f"{path} has unsupported dtype {value.dtype}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{path} contains non-finite float values")
        if value.dtype.kind == "u" and value.dtype.itemsize >= 8 and (value > np.uint64(_INT64_MAX)).any():
            raise ValueError(f"{path} must fit in a signed 64-bit integer")
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"{path} keys must be strings, got {type(k).__name__}")
            out[k] = _to_json_value(v, f"{path}.{k}")
        return out
    raise ValueError(f"{path} has unsupported type {type(value).__name__}")


def _validate_runtime(runtime) -> None:
    if runtime is None:
        return
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a map")
    for k in runtime:
        if not isinstance(k, str):
            raise ValueError("runtime keys must be strings")
    unknown = set(runtime) - _RUNTIME_KEYS
    if unknown:
        raise ValueError(f"Unknown runtime parameter(s): {sorted(unknown)}")
    resp_fmt = runtime.get("response_format")
    if resp_fmt is not None and resp_fmt != "envelope":
        raise ValueError(f"Unsupported runtime.response_format {resp_fmt!r}; expected 'envelope'")
    out_fmt = runtime.get("output_format")
    if out_fmt is not None and out_fmt != "numpy":
        raise ValueError(f"Unsupported runtime.output_format {out_fmt!r}; expected 'numpy'")


def _validate_rollout(raw_rollout, effective_steps: int | None) -> dict | None:
    if raw_rollout is None:
        return None
    if not isinstance(raw_rollout, dict):
        raise ValueError("parameters.rollout must be a map")
    for k in raw_rollout:
        if not isinstance(k, str):
            raise ValueError("parameters.rollout keys must be strings")
    unknown = set(raw_rollout) - _ROLLOUT_KEYS
    if unknown:
        raise ValueError(f"Unknown rollout parameter(s): {sorted(unknown)}")
    sde_type = raw_rollout.get("sde_type")
    if not isinstance(sde_type, str) or sde_type not in ("flow_sde", "flow_ode"):
        raise ValueError("rollout.sde_type must be 'flow_sde' or 'flow_ode'")
    raw_nl = raw_rollout.get("noise_level")
    noise_level = 0.5 if raw_nl is None else _check_finite_number(raw_nl, "rollout.noise_level")
    if noise_level < 0.0:
        raise ValueError("rollout.noise_level must be >= 0")
    raw_step = raw_rollout.get("sde_step")
    if raw_step is None or raw_step == "random":
        sde_step: str | int = "random"
    else:
        sde_step = _check_int64(raw_step, "rollout.sde_step")
        if sde_step < 0:
            raise ValueError("rollout.sde_step must be >= 0 or 'random'")
        if effective_steps is not None and sde_step >= effective_steps:
            raise ValueError(f"rollout.sde_step ({sde_step}) must be < num_steps ({effective_steps})")
    raw_ret = raw_rollout.get("return_state")
    return_state = True if raw_ret is None else _check_bool(raw_ret, "rollout.return_state")
    raw_ign = raw_rollout.get("ignore_last")
    ignore_last = False if raw_ign is None else _check_bool(raw_ign, "rollout.ignore_last")
    video_solver = raw_rollout.get("video_solver")
    if video_solver is not None and (
        not isinstance(video_solver, str) or video_solver not in ("hybrid_unipc", "euler")
    ):
        raise ValueError("rollout.video_solver must be 'hybrid_unipc' or 'euler'")
    out = {
        "sde_type": sde_type,
        "noise_level": noise_level,
        "sde_step": sde_step,
        "return_state": return_state,
        "ignore_last": ignore_last,
    }
    if video_solver is not None:
        out["video_solver"] = video_solver
    return out


def _parse_seeds(parameters: dict, batch_size: int) -> list[int | None]:
    raw_seed = parameters.get("seed")
    raw_seeds = parameters.get("seeds")
    if raw_seed is not None and raw_seeds is not None:
        raise ValueError("Provide either 'seed' or 'seeds', not both")
    if raw_seed is not None:
        return [_check_int64(raw_seed, "seed")] * batch_size
    if raw_seeds is not None:
        if isinstance(raw_seeds, np.ndarray):
            if raw_seeds.ndim != 1 or raw_seeds.dtype.kind not in "iu":
                raise ValueError("parameters.seeds must be a 1D integer list or array")
            if raw_seeds.dtype.kind == "u" and raw_seeds.dtype.itemsize >= 8 and (
                raw_seeds > np.uint64(_INT64_MAX)
            ).any():
                raise ValueError("seeds[] must fit in a signed 64-bit integer")
            items = [int(x) for x in raw_seeds.tolist()]
        elif isinstance(raw_seeds, list):
            items = [_check_int64(x, "seeds[]") for x in raw_seeds]
        else:
            raise ValueError("parameters.seeds must be a list of integers")
        if len(items) != batch_size:
            raise ValueError(f"parameters.seeds length ({len(items)}) must match batch size ({batch_size})")
        return items
    return [None] * batch_size


def _parse_robot_state_rows(raw_state, batch_size: int | None, name: str) -> list[list[float]]:
    if isinstance(raw_state, np.ndarray):
        if raw_state.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be numeric")
        if raw_state.dtype.kind == "f" and not np.isfinite(raw_state).all():
            raise ValueError(f"{name} must be finite")
        if raw_state.ndim == 1:
            if raw_state.shape[0] == 0:
                raise ValueError(f"{name} must not be empty")
            vec = [float(x) for x in raw_state.tolist()]
            if batch_size is not None and batch_size != 1:
                raise ValueError(f"{name} must be 2D [B, D] for batch size {batch_size}")
            return [vec]
        if raw_state.ndim == 2:
            if raw_state.shape[0] == 0 or raw_state.shape[1] == 0:
                raise ValueError(f"{name} must not be empty")
            if batch_size is not None and raw_state.shape[0] != batch_size:
                raise ValueError(f"{name} batch size ({raw_state.shape[0]}) must match ({batch_size})")
            return [[float(x) for x in row] for row in raw_state.tolist()]
        raise ValueError(f"{name} must be 1D or 2D, got {raw_state.ndim}D")
    if isinstance(raw_state, list):
        if not raw_state:
            raise ValueError(f"{name} must not be empty")
        if isinstance(raw_state[0], list):
            if batch_size is not None and len(raw_state) != batch_size:
                raise ValueError(f"{name} batch size ({len(raw_state)}) must match ({batch_size})")
            rows: list[list[float]] = []
            for row in raw_state:
                if not isinstance(row, list) or not row:
                    raise ValueError(f"{name} rows must be non-empty lists of numbers")
                rows.append([_check_finite_number(x, name) for x in row])
            return rows
        vec = [_check_finite_number(x, name) for x in raw_state]
        if batch_size is not None and batch_size != 1:
            raise ValueError(f"{name} must be 2D [B, D] for batch size {batch_size}")
        return [vec]
    raise ValueError(f"{name} must be a list or ndarray")


def _save_image_ref(ref: str, upload_dir: Path, saved_paths: list[str]) -> str:
    if not isinstance(ref, str) or not ref:
        raise ValueError("input_reference entries must be non-empty data URLs")
    modality, path = media_io.resolve_media_ref(ref, upload_dir)
    saved_paths.append(path)
    if modality != "image":
        raise ValueError(f"input_reference must be an image, got {modality}")
    return path


def _build_cosmos3_rows(
    inp: dict,
    parameters: dict,
    facts: dict,
    upload_dir: Path,
    saved_paths: list[str],
) -> tuple[list[_RowSubmission], bool]:
    raw_prompt = inp.get("prompt")
    if isinstance(raw_prompt, str):
        prompts = [raw_prompt]
    elif isinstance(raw_prompt, list) and all(isinstance(p, str) for p in raw_prompt):
        prompts = list(raw_prompt)
    else:
        raise ValueError("Cosmos3 input.prompt must be a string or list of strings")
    batch_size = len(prompts)
    if not 1 <= batch_size <= MAX_ACTION_BATCH_SIZE:
        raise ValueError(f"Batch size must be in [1, {MAX_ACTION_BATCH_SIZE}], got {batch_size}")

    raw_refs = inp.get("input_reference")
    ref_paths: list[str | None]
    if raw_refs is None:
        ref_paths = [None] * batch_size
    else:
        if isinstance(raw_refs, str) and batch_size == 1:
            refs = [raw_refs]
        elif isinstance(raw_refs, list):
            refs = raw_refs
        else:
            raise ValueError("input.input_reference must be a list of image data URLs")
        if len(refs) != batch_size:
            raise ValueError(
                f"input.input_reference length ({len(refs)}) must match input.prompt length ({batch_size})"
            )
        ref_paths = [_save_image_ref(ref, upload_dir, saved_paths) for ref in refs]

    raw_steps = parameters.get("num_inference_steps")
    effective_steps = _check_int64(raw_steps, "num_inference_steps") if raw_steps is not None else facts.get(
        "num_flow_steps"
    )
    rollout_cfg = _validate_rollout(parameters.get("rollout"), effective_steps)
    seeds = _parse_seeds(parameters, batch_size)

    in_rs = inp.get("robot_state") is not None or inp.get("robot_states") is not None
    pa_rs = parameters.get("robot_state") is not None or parameters.get("robot_states") is not None
    if in_rs and pa_rs:
        raise ValueError("Provide robot_state/robot_states in either input or parameters, not both")
    src_state = parameters if pa_rs else inp
    if src_state.get("robot_state") is not None and src_state.get("robot_states") is not None:
        raise ValueError("Provide either 'robot_state' or 'robot_states', not both")
    raw_state = src_state.get("robot_state") if src_state.get("robot_state") is not None else src_state.get(
        "robot_states"
    )
    state_rows = _parse_robot_state_rows(raw_state, batch_size, "robot_state") if raw_state is not None else None

    shared_mk: dict = {}
    skip_keys = {"seed", "seeds", "rollout", "robot_state", "robot_states"}
    for key, val in parameters.items():
        if not isinstance(key, str):
            raise ValueError("parameters keys must be strings")
        if key in skip_keys or val is None:
            continue
        shared_mk[key] = _to_json_value(val, f"parameters.{key}")
    if "action_mode" not in shared_mk:
        shared_mk["action_mode"] = "policy"
    elif not isinstance(shared_mk["action_mode"], str):
        raise ValueError("parameters.action_mode must be a string")
    for int_key in ("raw_action_dim", "num_frames", "action_chunk_size", "num_inference_steps"):
        raw_val = parameters.get(int_key)
        if raw_val is not None:
            iv = _check_int64(raw_val, f"parameters.{int_key}")
            if iv <= 0:
                raise ValueError(f"parameters.{int_key} must be >= 1")
            shared_mk[int_key] = iv
    if rollout_cfg is not None:
        shared_mk["rollout"] = rollout_cfg

    rows: list[_RowSubmission] = []
    for i in range(batch_size):
        row_mk = dict(shared_mk)
        if seeds[i] is not None:
            row_mk["seed"] = seeds[i]
        if state_rows is not None:
            row_mk["robot_state"] = state_rows[i]
        path = ref_paths[i]
        rows.append(_RowSubmission(
            text=prompts[i],
            file_paths={"image": [path]} if path is not None else None,
            raw_inputs=None,
            input_modalities=["image", "text"] if path is not None else ["text"],
            output_modalities=["action"],
            model_kwargs=row_mk,
        ))
    expect_rollout = rollout_cfg is not None and rollout_cfg["return_state"]
    return rows, expect_rollout


def _parse_tokenized_prompt_rows(raw_tp, raw_mask) -> tuple[list[list[int]], list[list[bool]] | None]:
    if isinstance(raw_tp, np.ndarray):
        if raw_tp.dtype.kind not in "iu":
            raise ValueError(f"tokenized_prompt must have integer dtype, not {raw_tp.dtype}")
        if raw_tp.ndim == 1:
            if raw_tp.shape[0] == 0:
                raise ValueError("tokenized_prompt must not be empty")
            if raw_tp.dtype.kind == "u" and raw_tp.dtype.itemsize >= 8 and (raw_tp > np.uint64(_INT64_MAX)).any():
                raise ValueError("tokenized_prompt[] must fit in a signed 64-bit integer")
            tp_rows = [[int(x) for x in raw_tp.tolist()]]
        elif raw_tp.ndim == 2:
            if raw_tp.shape[0] == 0 or raw_tp.shape[1] == 0:
                raise ValueError("tokenized_prompt must not be empty")
            if raw_tp.dtype.kind == "u" and raw_tp.dtype.itemsize >= 8 and (raw_tp > np.uint64(_INT64_MAX)).any():
                raise ValueError("tokenized_prompt[] must fit in a signed 64-bit integer")
            tp_rows = [[int(x) for x in row] for row in raw_tp.tolist()]
        else:
            raise ValueError(f"tokenized_prompt must be 1D or 2D, got {raw_tp.ndim}D")
    elif isinstance(raw_tp, list):
        if not raw_tp:
            raise ValueError("tokenized_prompt must not be empty")
        if isinstance(raw_tp[0], list):
            tp_rows = []
            for row in raw_tp:
                if not isinstance(row, list) or not row:
                    raise ValueError("tokenized_prompt rows must be non-empty integer lists")
                tp_rows.append([_check_int64(x, "tokenized_prompt[]") for x in row])
        else:
            tp_rows = [[_check_int64(x, "tokenized_prompt[]") for x in raw_tp]]
    else:
        raise ValueError("tokenized_prompt must be a 1D/2D integer array or list")

    for row in tp_rows:
        if any(tok < 0 for tok in row):
            raise ValueError("tokenized_prompt IDs must be non-negative")

    if raw_mask is None:
        return tp_rows, None

    if isinstance(raw_mask, np.ndarray):
        if raw_mask.dtype.kind not in "biu":
            raise ValueError(f"tokenized_prompt_mask must be bool or integer, not {raw_mask.dtype}")
        if raw_mask.ndim == 1:
            mask_rows_raw = [raw_mask.tolist()]
        elif raw_mask.ndim == 2:
            mask_rows_raw = raw_mask.tolist()
        else:
            raise ValueError(f"tokenized_prompt_mask must be 1D or 2D, got {raw_mask.ndim}D")
    elif isinstance(raw_mask, list):
        if not raw_mask:
            raise ValueError("tokenized_prompt_mask must not be empty")
        if isinstance(raw_mask[0], list):
            mask_rows_raw = raw_mask
        else:
            mask_rows_raw = [raw_mask]
    else:
        raise ValueError("tokenized_prompt_mask must be a 1D/2D bool/int array or list")

    if len(mask_rows_raw) != len(tp_rows):
        raise ValueError(
            f"tokenized_prompt_mask batch size ({len(mask_rows_raw)}) must match tokenized_prompt ({len(tp_rows)})"
        )
    mask_rows: list[list[bool]] = []
    for row_idx, (tp_row, m_row) in enumerate(zip(tp_rows, mask_rows_raw, strict=True)):
        if not isinstance(m_row, list) or len(m_row) != len(tp_row):
            raise ValueError(
                f"tokenized_prompt_mask row {row_idx} length must match tokenized_prompt ({len(tp_row)})"
            )
        parsed: list[bool] = []
        seen_false = False
        for v in m_row:
            if isinstance(v, (bool, np.bool_)):
                b = bool(v)
            elif isinstance(v, (int, np.integer)) and int(v) in (0, 1):
                b = bool(int(v))
            else:
                raise ValueError("tokenized_prompt_mask elements must be bool or 0/1")
            if seen_false and b:
                raise ValueError("tokenized_prompt_mask must be a contiguous prefix of True values")
            if not b:
                seen_false = True
            parsed.append(b)
        if not any(parsed):
            raise ValueError("tokenized_prompt_mask must have at least one True token")
        mask_rows.append(parsed)
    return tp_rows, mask_rows


def _parse_pi05_images(
    inp: dict,
    num_cameras: int,
    upload_dir: Path,
    saved_paths: list[str],
) -> tuple[list[list[torch.Tensor]] | None, list[list[str]] | None, list[str] | None]:
    present_img_keys = [k for k in _IMAGE_INPUT_KEYS if inp.get(k) is not None]
    if len(present_img_keys) > 1:
        raise ValueError(f"Provide only one of {present_img_keys}")
    raw_ref = inp.get("input_reference")
    if present_img_keys and raw_ref is not None:
        raise ValueError("Provide either image arrays or input_reference, not both")
    if not present_img_keys and raw_ref is None:
        raise ValueError("Pi0.5 input requires images (or input_reference)")

    if raw_ref is not None:
        if isinstance(raw_ref, str):
            ref_rows = [[raw_ref]]
        elif isinstance(raw_ref, list) and raw_ref:
            if all(isinstance(r, str) for r in raw_ref):
                ref_rows = [[r] for r in raw_ref]
            elif all(isinstance(r, list) and r and all(isinstance(x, str) for x in r) for r in raw_ref):
                ref_rows = raw_ref
            else:
                raise ValueError("input.input_reference must be a list of image data URLs")
        else:
            raise ValueError("input.input_reference must be a non-empty list of image data URLs")
        cam_count = len(ref_rows[0])
        if any(len(r) != cam_count for r in ref_rows):
            raise ValueError("All rows in input_reference must have the same number of cameras")
        if num_cameras > 0 and cam_count > num_cameras:
            raise ValueError(f"Received {cam_count} cameras; model accepts at most {num_cameras}")
        file_rows = [[_save_image_ref(uri, upload_dir, saved_paths) for uri in row] for row in ref_rows]
        return None, file_rows, None

    key = present_img_keys[0]
    val = inp[key]
    cam_order: list[str] | None = None
    if isinstance(val, np.ndarray):
        if val.ndim == 5:
            raw_rows = [[val[b, c] for c in range(val.shape[1])] for b in range(val.shape[0])]
        elif val.ndim == 4:
            raw_rows = [[val[c] for c in range(val.shape[0])]]
        else:
            raise ValueError(f"{key} ndarray must be 4D [C, ...] or 5D [B, C, ...], got {val.ndim}D")
    elif isinstance(val, dict):
        if not val:
            raise ValueError(f"{key} map must not be empty")
        cam_order = []
        per_cam_rows: list[list[np.ndarray]] = []
        expected_b: int | None = None
        for cam_name, cam_val in val.items():
            if not isinstance(cam_name, str):
                raise ValueError(f"{key} camera keys must be strings")
            cam_order.append(cam_name)
            if isinstance(cam_val, np.ndarray):
                if cam_val.ndim == 4:
                    cam_batch = [cam_val[b] for b in range(cam_val.shape[0])]
                elif cam_val.ndim == 3:
                    cam_batch = [cam_val]
                else:
                    raise ValueError(f"{key}.{cam_name} must be 3D or 4D, got {cam_val.ndim}D")
            elif isinstance(cam_val, list) and cam_val:
                cam_batch = cam_val
            else:
                raise ValueError(f"{key}.{cam_name} must be an image array or non-empty list of images")
            if expected_b is None:
                expected_b = len(cam_batch)
            elif len(cam_batch) != expected_b:
                raise ValueError(f"{key}.{cam_name} batch size ({len(cam_batch)}) does not match ({expected_b})")
            per_cam_rows.append(cam_batch)
        assert expected_b is not None
        raw_rows = [[per_cam_rows[c][b] for c in range(len(cam_order))] for b in range(expected_b)]
    elif isinstance(val, list):
        if not val:
            raise ValueError(f"{key} must not be empty")
        if all(isinstance(item, np.ndarray) and item.ndim == 4 for item in val):
            raw_rows = [[item[c] for c in range(item.shape[0])] for item in val]
        elif all(isinstance(item, list) and item for item in val):
            raw_rows = val
        elif all(isinstance(item, np.ndarray) and item.ndim == 3 for item in val):
            raw_rows = [val]
        else:
            raise ValueError(f"{key} list must contain 3D/4D image arrays or per-row camera lists")
    else:
        raise ValueError(f"{key} must be an ndarray, map, or list")

    if not raw_rows or not raw_rows[0]:
        raise ValueError(f"{key} must contain at least one row and one camera")
    cam_count = len(raw_rows[0])
    if any(len(r) != cam_count for r in raw_rows):
        raise ValueError("All rows must have the same number of cameras")
    if num_cameras > 0 and cam_count > num_cameras:
        raise ValueError(f"Received {cam_count} cameras; model accepts at most {num_cameras}")

    tensor_rows = [
        [openpi._image(cam, f"{key}[{c}]") for c, cam in enumerate(row)]
        for row in raw_rows
    ]
    return tensor_rows, None, cam_order


def _parse_camera_mask(
    raw_mask, batch_size: int, cam_count: int, cam_order: list[str] | None,
) -> list[list[bool]]:
    if isinstance(raw_mask, dict):
        if cam_order is None:
            raise ValueError("camera_mask can only be a mapping when images is a mapping")
        if set(raw_mask.keys()) != set(cam_order):
            raise ValueError("camera_mask keys must match images keys")
        per_cam: list[list[bool]] = []
        for cam_name in cam_order:
            v = raw_mask[cam_name]
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    vals = [v.item()]
                elif v.ndim == 1:
                    vals = v.tolist()
                else:
                    raise ValueError(f"camera_mask.{cam_name} must be scalar or 1D")
            elif isinstance(v, list):
                vals = v
            else:
                vals = [v]
            if len(vals) != batch_size:
                raise ValueError(f"camera_mask.{cam_name} length ({len(vals)}) must match batch size ({batch_size})")
            per_cam.append([_bool_01(x, f"camera_mask.{cam_name}") for x in vals])
        rows = [[per_cam[c][b] for c in range(cam_count)] for b in range(batch_size)]
    elif isinstance(raw_mask, np.ndarray):
        if raw_mask.dtype.kind not in "biu":
            raise ValueError(f"camera_mask must be bool or 0/1 integer, not {raw_mask.dtype}")
        if raw_mask.ndim == 1:
            if batch_size != 1:
                raise ValueError(f"camera_mask must be 2D [B, C] for batch size {batch_size}")
            rows = [[_bool_01(x, "camera_mask") for x in raw_mask.tolist()]]
        elif raw_mask.ndim == 2:
            if raw_mask.shape[0] != batch_size:
                raise ValueError(f"camera_mask batch size ({raw_mask.shape[0]}) must match ({batch_size})")
            rows = [[_bool_01(x, "camera_mask") for x in row] for row in raw_mask.tolist()]
        else:
            raise ValueError(f"camera_mask must be 1D or 2D, got {raw_mask.ndim}D")
    elif isinstance(raw_mask, list):
        if not raw_mask:
            raise ValueError("camera_mask must not be empty")
        if isinstance(raw_mask[0], list):
            if len(raw_mask) != batch_size:
                raise ValueError(f"camera_mask batch size ({len(raw_mask)}) must match ({batch_size})")
            rows = [[_bool_01(x, "camera_mask") for x in row] for row in raw_mask]
        else:
            if batch_size != 1:
                raise ValueError(f"camera_mask must be 2D [B, C] for batch size {batch_size}")
            rows = [[_bool_01(x, "camera_mask") for x in raw_mask]]
    else:
        raise ValueError("camera_mask must be an array, list, or mapping")

    for row in rows:
        if len(row) != cam_count:
            raise ValueError(f"camera_mask length ({len(row)}) must match camera count ({cam_count})")
        if not any(row):
            raise ValueError("camera_mask must keep at least one camera")
    return rows


def _bool_01(val, name: str) -> bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)) and int(val) in (0, 1):
        return bool(int(val))
    raise ValueError(f"{name} elements must be bool or 0/1")


def _build_pi05_rows(
    inp: dict,
    parameters: dict,
    facts: dict,
    upload_dir: Path,
    saved_paths: list[str],
) -> tuple[list[_RowSubmission], bool]:
    tensor_rows, file_rows, cam_order = _parse_pi05_images(
        inp, int(facts.get("num_cameras", 3)), upload_dir, saved_paths,
    )
    img_batch = len(tensor_rows) if tensor_rows is not None else len(file_rows)  # type: ignore[arg-type]
    cam_count = len(tensor_rows[0]) if tensor_rows is not None else len(file_rows[0])  # type: ignore[index]

    raw_tp = inp.get("tokenized_prompt")
    raw_tp_mask = inp.get("tokenized_prompt_mask")
    if raw_tp is None and raw_tp_mask is not None:
        raise ValueError("tokenized_prompt_mask requires tokenized_prompt")
    tp_rows = mask_rows = None
    if raw_tp is not None:
        tp_rows, mask_rows = _parse_tokenized_prompt_rows(raw_tp, raw_tp_mask)
        if len(tp_rows) != img_batch:
            raise ValueError(
                f"tokenized_prompt batch size ({len(tp_rows)}) must match images ({img_batch})"
            )

    raw_prompt = inp.get("prompt")
    prompts: list[str | None]
    if raw_prompt is None:
        if tp_rows is None:
            raise ValueError("Pi0.5 input requires either 'prompt' or 'tokenized_prompt'")
        prompts = [None] * img_batch
    elif isinstance(raw_prompt, str):
        prompts = [raw_prompt] * img_batch
    elif isinstance(raw_prompt, list) and all(isinstance(p, str) for p in raw_prompt):
        if len(raw_prompt) != img_batch:
            raise ValueError(f"input.prompt length ({len(raw_prompt)}) must match images ({img_batch})")
        prompts = list(raw_prompt)
    else:
        raise ValueError("input.prompt must be a string or list of strings")

    batch_size = img_batch
    if not 1 <= batch_size <= MAX_ACTION_BATCH_SIZE:
        raise ValueError(f"Batch size must be in [1, {MAX_ACTION_BATCH_SIZE}], got {batch_size}")

    # camera_mask / image_mask (in input or parameters, not both)
    in_cm = inp.get("camera_mask") is not None or inp.get("image_mask") is not None
    pa_cm = parameters.get("camera_mask") is not None or parameters.get("image_mask") is not None
    if in_cm and pa_cm:
        raise ValueError("Provide camera_mask/image_mask in either input or parameters, not both")
    cm_src = inp if in_cm else parameters
    if cm_src.get("camera_mask") is not None and cm_src.get("image_mask") is not None:
        raise ValueError("Provide either 'camera_mask' or 'image_mask', not both")
    raw_cm = cm_src.get("camera_mask") if cm_src.get("camera_mask") is not None else cm_src.get("image_mask")
    cam_mask_rows = _parse_camera_mask(raw_cm, batch_size, cam_count, cam_order) if raw_cm is not None else None

    # robot_state / state (in input or parameters, not both)
    in_rs = inp.get("robot_state") is not None or inp.get("state") is not None
    pa_rs = parameters.get("robot_state") is not None or parameters.get("state") is not None
    if in_rs and pa_rs:
        raise ValueError("Provide robot_state/state in either input or parameters, not both")
    rs_src = inp if in_rs else parameters
    if rs_src.get("robot_state") is not None and rs_src.get("state") is not None:
        raise ValueError("Provide either 'robot_state' or 'state', not both")
    raw_rs = rs_src.get("robot_state") if rs_src.get("robot_state") is not None else rs_src.get("state")
    state_rows = _parse_robot_state_rows(raw_rs, batch_size, "robot_state") if raw_rs is not None else None

    rollout_cfg = _validate_rollout(parameters.get("rollout"), facts.get("num_flow_steps"))
    seeds = _parse_seeds(parameters, batch_size)

    shared_mk: dict = {}
    skip_keys = {"seed", "seeds", "rollout", "robot_state", "state", "camera_mask", "image_mask"}
    for key, val in parameters.items():
        if not isinstance(key, str):
            raise ValueError("parameters keys must be strings")
        if key in skip_keys or val is None:
            continue
        shared_mk[key] = _to_json_value(val, f"parameters.{key}")
    if rollout_cfg is not None:
        shared_mk["rollout"] = rollout_cfg

    state_kwarg = facts.get("state_kwarg", "robot_state")
    rows: list[_RowSubmission] = []
    for i in range(batch_size):
        row_mk = dict(shared_mk)
        if seeds[i] is not None:
            row_mk["seed"] = seeds[i]
        if tp_rows is not None:
            row_mk["tokenized_prompt"] = tp_rows[i]
        if mask_rows is not None:
            row_mk["tokenized_prompt_mask"] = mask_rows[i]
        if cam_mask_rows is not None:
            row_mk["camera_mask"] = cam_mask_rows[i]
        if state_rows is not None:
            row_mk[state_kwarg] = state_rows[i]
        rows.append(_RowSubmission(
            text=prompts[i],
            file_paths={"image": file_rows[i]} if file_rows is not None else None,
            raw_inputs={"image": tensor_rows[i]} if tensor_rows is not None else None,
            input_modalities=["image", "text"],
            output_modalities=["action"],
            model_kwargs=row_mk,
        ))
    expect_rollout = rollout_cfg is not None and rollout_cfg["return_state"]
    return rows, expect_rollout


def _release_submitted(api, keys: list[tuple[str, str | None]]) -> None:
    for request_id, key in keys:
        try:
            if key is not None and hasattr(api, "release_submission"):
                api.release_submission(key)
            elif key is not None and hasattr(api, "abort_submission"):
                api.abort_submission(key)
            elif hasattr(api, "release_request"):
                api.release_request(request_id)
            elif hasattr(api, "abort_request"):
                api.abort_request(request_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to release request %s: %r", request_id, exc)


def _reshape_action(data: bytes, metadata: dict, facts: dict) -> np.ndarray:
    if not data or len(data) % 4 != 0:
        raise HTTPException(status_code=500, detail=f"Invalid action byte length: {len(data)}")
    shape = metadata.get("shape")
    n_floats = len(data) // 4
    if (
        isinstance(shape, (list, tuple))
        and len(shape) == 2
        and all(type(x) is int and x > 0 for x in shape)
        and shape[0] * shape[1] == n_floats
    ):
        horizon, action_dim = shape[0], shape[1]
    else:
        action_dim = int(facts["action_dim"])
        if action_dim <= 0 or n_floats % action_dim != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Action output has {n_floats} floats, incompatible with action_dim={action_dim}",
            )
        horizon = n_floats // action_dim
    return np.frombuffer(data, dtype="<f4").reshape(horizon, action_dim).copy()


def _extract_rollout(data: bytes, weight_version: str | None) -> dict:
    try:
        unpacked = openpi.unpack(data)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt rollout tensor chunk: {exc}") from None
    if not isinstance(unpacked, dict):
        raise HTTPException(status_code=500, detail="Rollout tensor chunk must unpack to a map")
    required = ("chain", "denoise_ind", "logprobs", "logprobs_mask", "seed")
    missing = [k for k in required if k not in unpacked]
    if missing:
        raise HTTPException(status_code=500, detail=f"Rollout tensor chunk missing keys: {missing}")
    chain = unpacked["chain"]
    logprobs = unpacked["logprobs"]
    logprobs_mask = unpacked["logprobs_mask"]
    if not isinstance(chain, np.ndarray) or not isinstance(logprobs, np.ndarray) or not isinstance(
        logprobs_mask, np.ndarray
    ):
        raise HTTPException(status_code=500, detail="Rollout tensor chunk arrays must be ndarrays")
    try:
        denoise_ind = _check_int64(unpacked["denoise_ind"], "denoise_ind")
        seed = _check_int64(unpacked["seed"], "seed")
    except ValueError:
        raise HTTPException(status_code=500, detail="Rollout denoise_ind and seed must be integers") from None
    return {
        "chain": np.ascontiguousarray(chain, dtype="<f4"),
        "denoise_ind": denoise_ind,
        "logprobs": np.ascontiguousarray(logprobs, dtype="<f4"),
        "logprobs_mask": np.ascontiguousarray(logprobs_mask, dtype="|b1"),
        "seed": seed,
        "weight_version": weight_version,
    }


def _pending_weight_version(api, key: str | None) -> str | None:
    if key is None:
        return None
    pending = getattr(api, "pending_requests", None)
    if not isinstance(pending, dict):
        return None
    req = pending.get(key)
    if req is None:
        return None
    wv = getattr(req, "weight_version", None)
    if wv is not None:
        return wv
    return getattr(req, "chunk_weight_version", None)


async def _collect_row(
    api,
    index: int,
    request_id: str,
    key: str | None,
    facts: dict,
    expect_rollout: bool,
) -> dict:
    action_data: bytes | None = None
    action_meta: dict = {}
    tensor_data: bytes | None = None
    tensor_is_rollout = False
    fallback_wv = _pending_weight_version(api, key)
    chunk_wv: str | None = None

    chunks = api.iter_result_chunks(request_id)
    try:
        async for chunk in chunks:
            meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            if fallback_wv is None:
                fallback_wv = _pending_weight_version(api, key)
            if chunk_wv is None and isinstance(meta.get("weight_version"), str):
                chunk_wv = meta["weight_version"]

            if chunk.modality == "error":
                status = int(meta.get("status", 500))
                msg = bytes(chunk.data).decode("utf-8", "replace")
                raise HTTPException(status_code=status, detail=msg)
            if chunk.modality == "action":
                action_data = bytes(chunk.data)
                action_meta = meta
            elif chunk.modality == "tensor":
                if meta.get("rollout_state") is True or not tensor_is_rollout:
                    tensor_data = bytes(chunk.data)
                    tensor_is_rollout = bool(meta.get("rollout_state") is True)
    finally:
        aclose = getattr(chunks, "aclose", None)
        if aclose is not None:
            await aclose()

    weight_version = chunk_wv if chunk_wv is not None else fallback_wv

    if action_data is None:
        raise HTTPException(status_code=500, detail="Request finished without action output")

    entry: dict = {
        "input_index": index,
        "action": {"values": _reshape_action(action_data, action_meta, facts)},
    }
    if expect_rollout:
        if tensor_data is None:
            raise HTTPException(status_code=500, detail="Request finished without rollout state output")
        entry["rollout"] = _extract_rollout(tensor_data, weight_version)
    return entry


async def create_actions(api, request: Request) -> Response:
    """Handle ``POST /v1/actions/generations`` (``application/msgpack``)."""
    if api is None:
        return _error(503, "Server not ready")
    if getattr(api, "fatal_error", None) is not None:
        return _error(503, str(api.fatal_error))
    if getattr(api, "paused", False):
        return _error(503, "Generation is paused")

    facts = actions_facts(api)
    if facts is None:
        model_name = getattr(api, "model_name", "unknown")
        return _error(404, f"Model {model_name!r} does not support /v1/actions/generations")

    ctype = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if ctype not in ("application/msgpack", "application/x-msgpack"):
        return _error(415, "Content-Type must be application/msgpack")

    raw_body = await request.body()
    if not raw_body:
        return _error(400, "Request body must not be empty")
    try:
        payload = openpi.unpack(raw_body)
    except ValueError as exc:
        return _error(400, str(exc))

    if not isinstance(payload, dict):
        return _error(400, "Request body must be a msgpack map")
    inp = payload.get("input")
    if not isinstance(inp, dict):
        return _error(400, "Request body requires an 'input' map")
    parameters = payload.get("parameters")
    if parameters is None:
        parameters = {}
    elif not isinstance(parameters, dict):
        return _error(400, "'parameters' must be a map")

    upload_dir = Path(getattr(api, "upload_dir", "/tmp/mstar_uploads"))
    saved_paths: list[str] = []
    try:
        def _prepare():
            _validate_runtime(payload.get("runtime"))
            if facts["family"] == "cosmos3":
                return _build_cosmos3_rows(inp, parameters, facts, upload_dir, saved_paths)
            return _build_pi05_rows(inp, parameters, facts, upload_dir, saved_paths)

        rows, expect_rollout = await asyncio.to_thread(_prepare)
    except ValueError as exc:
        media_io.remove_uploads({"image": saved_paths}, upload_dir)
        return _error(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        media_io.remove_uploads({"image": saved_paths}, upload_dir)
        status, msg = exception_status_text(exc, "/v1/actions/generations", log=logger)
        return _error(status, msg)

    submitted: list[tuple[str, str | None]] = []
    try:
        for row in rows:
            req_id = rid("act")
            returned_id = api.submit_request(
                text=row.text,
                file_paths=row.file_paths,
                raw_inputs=row.raw_inputs,
                input_modalities=row.input_modalities,
                output_modalities=row.output_modalities,
                model_kwargs=row.model_kwargs,
                streaming=False,
                request_id=req_id,
            )
            if isinstance(returned_id, str):
                req_id = returned_id
            key = api.submission_key(req_id) if hasattr(api, "submission_key") else None
            submitted.append((req_id, key))
    except BaseException as exc:
        _release_submitted(api, submitted)
        if saved_paths:
            if submitted:
                media_io.schedule_upload_cleanup({"image": saved_paths}, upload_dir)
            else:
                media_io.remove_uploads({"image": saved_paths}, upload_dir)
        if isinstance(exc, Exception):
            status, msg = exception_status_text(exc, "/v1/actions/generations", log=logger)
            return _error(status, msg)
        raise

    tasks: list[asyncio.Task[dict]] = []
    failed = True
    try:
        tasks = [
            asyncio.create_task(_collect_row(api, idx, req_id, key, facts, expect_rollout))
            for idx, (req_id, key) in enumerate(submitted)
        ]
        pending: set[asyncio.Task[dict]] = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
            for task in tasks:
                if task in done:
                    exc = task.exception()
                    if exc is not None:
                        raise exc
        results = [task.result() for task in tasks]
        failed = False
        results.sort(key=lambda item: item["input_index"])
        return Response(
            content=openpi.pack({"data": results}),
            media_type="application/msgpack",
            status_code=200,
        )
    except HTTPException as exc:
        return _error(exc.status_code, str(exc.detail))
    except Exception as exc:  # noqa: BLE001
        status, msg = exception_status_text(exc, "/v1/actions/generations", log=logger)
        return _error(status, msg)
    finally:
        try:
            if failed:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                try:
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                finally:
                    _release_submitted(api, submitted)
        finally:
            if saved_paths:
                media_io.schedule_upload_cleanup({"image": saved_paths}, upload_dir)
