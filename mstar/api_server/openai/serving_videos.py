"""/v1/videos/generations (text-to-video and image-to-video) handler."""

from __future__ import annotations

import base64
import logging

from mstar.api_server import media_io
from mstar.api_server.openai._util import now, rid

logger = logging.getLogger(__name__)


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

    chunks = await api.collect_results(request_id)
    # Each video chunk is an mp4 (H.264); return it base64-encoded, mirroring the
    # image endpoint's b64_json shape. A sound request additionally emits a raw
    # 16-bit PCM audio chunk, which is muxed into the mp4 as an AAC track (one
    # playable file with sound); if the mux fails the plain video is returned.
    # The mux also rescales the video timestamps to the request frame rate: the
    # video-only container always plays at the model's default fps, which would
    # drift out of sync with the true-time audio track for other request rates.
    request_fps = args.model_kwargs.get("fps")
    videos = [c.data for c in chunks if c.modality == "video"]
    audios = [c for c in chunks if c.modality == "audio"]
    data = []
    for i, video in enumerate(videos):
        if i < len(audios):
            audio = audios[i]
            try:
                video = media_io.mux_mp4_with_pcm16(
                    video,
                    audio.data,
                    sample_rate=int((audio.metadata or {}).get("sample_rate", 48000)),
                    num_channels=int((audio.metadata or {}).get("num_channels", 2)),
                    video_fps=float(request_fps) if request_fps else None,
                )
            except Exception:  # noqa: BLE001 — degrade to video-only, keep serving
                logger.exception("Muxing generated audio into the mp4 failed; returning video only")
        data.append({"b64_json": base64.b64encode(video).decode("ascii"), "url": None})
    return {"created": now(), "data": data}
