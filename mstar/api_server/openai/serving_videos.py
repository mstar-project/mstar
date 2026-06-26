"""/v1/videos/generations (text-to-video and image-to-video) handler."""

from __future__ import annotations

import base64

from starlette.concurrency import run_in_threadpool

from mstar.api_server.openai._util import now, rid


async def create_videos(api, model_name, adapter, req):  # noqa: ARG001
    args = adapter.video_to_request(req, api.upload_dir)
    request_id = rid("vid")

    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=["video"],
        model_kwargs=args.model_kwargs,
        streaming=False,
        request_id=request_id,
    )

    chunks = await run_in_threadpool(api.collect_results, request_id)
    # Each video chunk is an mp4 (H.264); return it base64-encoded, mirroring the
    # image endpoint's b64_json shape.
    data = [
        {"b64_json": base64.b64encode(c.data).decode("ascii"), "url": None}
        for c in chunks
        if c.modality == "video"
    ]
    return {"created": now(), "data": data}
