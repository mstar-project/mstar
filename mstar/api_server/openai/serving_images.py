"""/v1/images/generations (text-to-image) and /v1/images/edits (image-to-image) handlers."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from mstar.api_server import media_io
from mstar.api_server.openai._util import now, release_unread, rid
from mstar.utils.generation_kwargs import offset_seed


async def create_images(api, model_name, adapter, req, raw_request=None):  # noqa: ARG001
    args = adapter.image_to_request(req, api.upload_dir)
    # OpenAI ``n``: submit n engine requests up front and let cross-request
    # batching serve them together (the reference pipelines run n sequential
    # diffusions instead). Seeded requests follow the reference seed contract:
    # image i uses seed + i, so image 0 is bit-identical to the same request
    # with n=1; unseeded requests draw independent per-request seeds.
    n = image_count(req)
    seed = args.model_kwargs.get("seed")
    request_ids = []
    for i in range(n):
        model_kwargs = dict(args.model_kwargs)
        if seed is not None and i > 0:
            model_kwargs["seed"] = offset_seed(int(seed), i)
        request_id = rid("img")
        # Listed before submitting: submit_request registers the request
        # before handing it on, so a failed submit still holds an entry.
        request_ids.append(request_id)
        try:
            api.submit_request(
                text=args.text,
                file_paths=args.file_paths,
                input_modalities=args.input_modalities,
                output_modalities=["image"],
                model_kwargs=model_kwargs,
                streaming=False,
                request_id=request_id,
                model=req.model,
            )
        except BaseException:
            release_unread(api, request_ids, 0)
            raise

    data = []
    read = 0  # images collect_results has released; the rest are released on early exit
    try:
        for request_id in request_ids:
            try:
                chunks = await api.collect_results(request_id, raw_request)
            except HTTPException:
                read += 1  # collect_results released it before raising
                raise
            read += 1
            data.extend(
                {"b64_json": base64.b64encode(c.data).decode("ascii"), "url": None}
                for c in chunks
                if c.modality == "image"
            )
    finally:
        # One failed, timed-out or cancelled image fails the whole request, so
        # nothing will read the others: abort the ones still running and drop
        # the finished ones, which would otherwise hold their bytes forever.
        release_unread(api, request_ids, read)
    return {"created": now(), "data": data}


async def create_image_edit(
    api,
    model_name,
    adapter,
    *,
    prompt,
    image_bytes,
    image_filename,
    model_kwargs,
    requested_model=None,
    raw_request=None,
):  # noqa: ARG001
    # Persist the uploaded image so the model's loader can read it by path
    # (same contract as multipart uploads), then run the image-to-image edit.
    upload_dir = Path(api.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(image_filename or "")[1] or ".png"
    image_path = upload_dir / f"{uuid4().hex}{ext}"
    image_path.write_bytes(image_bytes)

    try:
        args = adapter.image_edit_to_request(prompt, str(image_path), model_kwargs)
    except Exception as exc:
        image_path.unlink(missing_ok=True)
        if isinstance(exc, NotImplementedError):
            # a model with images but no editing: the client's error, a 400
            raise ValueError(str(exc)) from exc
        raise
    request_id = rid("img")

    try:
        api.submit_request(
            text=args.text,
            file_paths=args.file_paths,
            input_modalities=args.input_modalities,
            output_modalities=["image"],
            model_kwargs=args.model_kwargs,
            streaming=False,
            request_id=request_id,
            model=requested_model,
        )
        chunks = await api.collect_results(request_id, raw_request)
    finally:
        media_io.schedule_upload_cleanup({"image": [str(image_path)]}, upload_dir)
    data = [
        {"b64_json": base64.b64encode(c.data).decode("ascii"), "url": None}
        for c in chunks
        if c.modality == "image"
    ]
    return {"created": now(), "data": data}


DEFAULT_MAX_IMAGES_PER_REQUEST = 10


def max_images_per_request() -> int:
    """Largest ``n`` one image request may ask for: ``MSTAR_MAX_IMAGES_PER_REQUEST``,
    default 10 (OpenAI's range for the endpoint), at least 1. The Rust frontend
    parses it the same way: an ASCII integer, surrounding whitespace allowed,
    anything else the default."""
    raw = os.environ.get("MSTAR_MAX_IMAGES_PER_REQUEST", "").strip()
    if not re.fullmatch(r"[+-]?[0-9]+", raw):
        return DEFAULT_MAX_IMAGES_PER_REQUEST
    return max(1, int(raw))


def image_count(req) -> int:
    """The request's ``n``, 1 when omitted. Outside ``1..max_images_per_request()``
    it raises ``ValueError`` (a 400), before anything is submitted."""
    n = getattr(req, "n", None)
    if n is None:
        return 1
    limit = max_images_per_request()
    if not 1 <= n <= limit:
        raise ValueError(f"n must be between 1 and {limit}, got {n}")
    return n
