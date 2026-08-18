"""Embed an M* server and serve it on a Dynamo runtime endpoint.

Launch plumbing mirrors the standalone entrypoint (lightweight model →
APIServer → conductor process → SETUP_DONE gate); instead of binding
uvicorn, the server registers with the Dynamo runtime and serves the
bridge handlers. Registration derives from the model's OpenAI adapter
capabilities: chat, speech, and images share one endpoint (their bodies
are distinguishable), video generation gets its own (a minimal video
body looks like an image body). The realtime surface still needs its
wire format mapped.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
from typing import TYPE_CHECKING

import yaml

from mstar.api_server.openai.adapters import get_adapter
from mstar.integrations.dynamo.bridges import RequestBridge

if TYPE_CHECKING:
    from mstar.api_server.entrypoint import APIServer

logger = logging.getLogger(__name__)


def build_server(args) -> tuple[APIServer, mp.process.BaseProcess, str]:
    """Start conductor + GPU workers and the API-server core (no HTTP)."""
    # Engine imports live here so importing this module (registration
    # mapping, --help) never loads the model stacks.
    from mstar.api_server.entrypoint import APIServer, _conductor_process_target
    from mstar.communication.communicator import CommProtocol
    from mstar.model.registry import HF_MODELS, get_model_class

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_name = config.get("model", "dummy")
    yaml_model_kwargs = config.get("model_kwargs", {}) or {}

    model = get_model_class(model_name)(
        model_path_hf=HF_MODELS.get(model_name, {}).get("model_path_hf", ""),
        cache_dir=args.cache_dir,
        **yaml_model_kwargs,
    )

    server = APIServer(
        socket_path_prefix=args.socket_path_prefix,
        upload_dir=args.upload_dir,
        timeout_seconds=args.timeout,
        tensor_comm_protocol=CommProtocol(args.tensor_comm_protocol),
        model=model,
        model_name=model_name,
    )

    ctx = mp.get_context("spawn")
    conductor_proc = ctx.Process(
        target=_conductor_process_target,
        args=(
            model_name,
            args.config,
            args.socket_path_prefix,
            False,  # enable_nvtx
            False,  # log_stats
            args.log_level,
            args.cache_dir,
            CommProtocol(args.tensor_comm_protocol),
            "",     # tcp_transfer_device
        ),
    )
    conductor_proc.start()
    logger.info("Conductor process started (pid=%d, model=%s)", conductor_proc.pid, model_name)
    return server, conductor_proc, model_name


def _model_type(adapter):
    """Map adapter capabilities onto the Dynamo registration vocabulary."""
    from dynamo.llm import ModelType

    model_type = None
    if adapter.supports_chat:
        model_type = ModelType.Chat
    if adapter.supports_speech:
        model_type = ModelType.Audios if model_type is None else model_type | ModelType.Audios
    if adapter.supports_images:
        model_type = ModelType.Images if model_type is None else model_type | ModelType.Images
    return model_type


def serve(server: APIServer, model_name: str, args) -> None:
    """Register with the Dynamo runtime and serve until shutdown."""
    import uvloop
    from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
    from dynamo.runtime import DistributedRuntime, dynamo_worker

    adapter = get_adapter(model_name)
    model_type = _model_type(adapter) if adapter is not None else None
    supports_videos = adapter is not None and getattr(adapter, "supports_videos", False)
    if model_type is None and not supports_videos:
        raise SystemExit(
            f"model {model_name!r} has no OpenAI adapter surface to register; "
            "models without one are served via the native /generate only"
        )

    served = args.served_model_name or model_name
    bridge = RequestBridge(server, adapter, served)

    @dynamo_worker()
    async def _run(runtime: DistributedRuntime):
        surfaces = []
        if model_type is not None:
            endpoint = runtime.endpoint(f"{args.namespace}.{args.component}.{args.endpoint}")
            await register_model(
                ModelInput.Text,
                model_type,
                endpoint,
                args.model_path,
                served,
                worker_type=WorkerType.Aggregated,
                needs=[],
            )
            surfaces.append((f"{args.endpoint}={model_type}",
                             endpoint.serve_endpoint(bridge.generate, graceful_shutdown=True)))
        if supports_videos:
            # Registered under the same served name: the frontend keeps one
            # worker set per (model, type), so the videos surface routes here
            # while chat/images keep the endpoint above.
            video_ep = runtime.endpoint(
                f"{args.namespace}.{args.component}.{args.endpoint}_videos"
            )
            await register_model(
                ModelInput.Text,
                ModelType.Videos,
                video_ep,
                args.model_path,
                served,
                worker_type=WorkerType.Aggregated,
                needs=[],
            )
            surfaces.append((f"{args.endpoint}_videos={ModelType.Videos}",
                             video_ep.serve_endpoint(bridge.videos, graceful_shutdown=True)))
        logger.info(
            "MSTAR_DYNAMO_READY model=%s component=%s.%s surfaces=%s",
            served, args.namespace, args.component,
            ", ".join(name for name, _ in surfaces),
        )
        await asyncio.gather(*(serve_coro for _, serve_coro in surfaces))

    uvloop.install()
    asyncio.run(_run())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="mstar — serve a deployment as a Dynamo backend worker"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--model-path", required=True,
                        help="HF snapshot dir backing the frontend-side model card")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--socket-path-prefix", default="/tmp/mstar")
    parser.add_argument("--upload-dir", default="/tmp/mstar_uploads")
    parser.add_argument("--tensor-comm-protocol", default="RDMA",
                        help="Tensor transfer protocol: RDMA, TCP, or SHM (shared memory)")
    parser.add_argument("--namespace", default="mstar")
    parser.add_argument("--component", default="backend")
    parser.add_argument("--endpoint", default="generate")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    from mstar.api_server.entrypoint import _shutdown_conductor_process
    from mstar.utils.logging_config import quiet_noisy_loggers

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [dynamo_worker] %(name)s: %(message)s",
    )
    quiet_noisy_loggers()

    server, conductor_proc, model_name = build_server(args)
    try:
        # Same gate the native server binds behind: every worker has loaded
        # weights, warmed up, and captured CUDA graphs.
        server.finalize_setup()
        serve(server, model_name, args)
    except KeyboardInterrupt:
        pass
    finally:
        server.cleanup()
        _shutdown_conductor_process(conductor_proc)
