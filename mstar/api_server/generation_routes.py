"""Optional generation-control routes, backed by #845's conductor protocol."""

from __future__ import annotations

import json
import sys

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()
_BODY_LIMIT = 4096


def _admin():
    for name in ("mstar.api_server.entrypoint", "__main__"):
        module = sys.modules.get(name)
        api = getattr(module, "api_server", None) if module is not None else None
        if api is not None:
            return api.generation_admin
    raise HTTPException(status_code=503, detail="Server not ready")


async def _body(request: Request) -> dict:
    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > _BODY_LIMIT:
            raise HTTPException(status_code=413, detail="generation control body too large")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        return {}
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="body must be a JSON object") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return value


async def _call(op: str, body: dict | None = None) -> JSONResponse:
    try:
        result = await _admin().call(op, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(status_code=result["status"], content=result)


@router.post("/pause_generation")
async def pause_generation(request: Request):
    return await _call("pause_generation", await _body(request))


@router.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request):
    return await _call("update_weights_from_disk", await _body(request))


@router.post("/continue_generation")
async def continue_generation(request: Request):
    return await _call("continue_generation", await _body(request))


@router.get("/is_paused")
async def is_paused():
    return await _call("is_paused")


@router.post("/weight_checker")
async def weight_checker(request: Request):
    return await _call("weight_checker", await _body(request))
