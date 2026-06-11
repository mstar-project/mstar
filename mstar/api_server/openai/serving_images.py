"""/v1/images/generations (text-to-image) and /v1/images/edits (image-to-image) handlers."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from uuid import uuid4

from starlette.concurrency import run_in_threadpool

from mstar.api_server.openai._util import now, rid


async def create_images(api, model_name, adapter, req):  # noqa: ARG001
    args = adapter.image_to_request(req, api.upload_dir)
    request_id = rid("img")

    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=["image"],
        model_kwargs=args.model_kwargs,
        streaming=False,
        request_id=request_id,
    )

    chunks = await run_in_threadpool(api.collect_results, request_id)
    data = [
        {"b64_json": base64.b64encode(c.data).decode("ascii"), "url": None}
        for c in chunks
        if c.modality == "image"
    ]
    return {"created": now(), "data": data}


async def create_image_edit(api, model_name, adapter, *, prompt, image_bytes, image_filename, model_kwargs):  # noqa: ARG001
    # Persist the uploaded image so the model's loader can read it by path
    # (same contract as multipart uploads), then run the image-to-image edit.
    upload_dir = Path(api.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(image_filename or "")[1] or ".png"
    image_path = upload_dir / f"{uuid4().hex}{ext}"
    image_path.write_bytes(image_bytes)

    args = adapter.image_edit_to_request(prompt, str(image_path), model_kwargs)
    request_id = rid("img")

    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=["image"],
        model_kwargs=args.model_kwargs,
        streaming=False,
        request_id=request_id,
    )

    chunks = await run_in_threadpool(api.collect_results, request_id)
    data = [
        {"b64_json": base64.b64encode(c.data).decode("ascii"), "url": None}
        for c in chunks
        if c.modality == "image"
    ]
    return {"created": now(), "data": data}
