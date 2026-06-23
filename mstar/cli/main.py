"""The ``mstar`` command — a one-command quickstart wrapper.

    mstar serve bagel                 # defaults, port 8000
    mstar serve qwen3_omni --gpus 0,1,2
    mstar serve orpheus --port 9000

``serve <model>`` resolves a sensible default config, fills in the
plumbing that the low-level ``mstar-serve`` requires (socket/upload dirs, a
single-node-safe tensor protocol, HF cache), and then delegates to the same
server entry point. Power users can still call ``mstar-serve --config ...``
directly, or pass ``--config`` here to override the default.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

import mstar

# Model name -> default config (relative to the repo's configs/).
DEFAULT_CONFIGS: dict[str, str] = {
    "bagel": "bagel_single_gpu.yaml",
    "bagel_cfg_parallel": "bagel_cfg_parallel.yaml",
    "orpheus": "orpheus_colocated.yaml",
    "qwen3_omni": "qwen3omni_2gpu.yaml",
    "pi05": "pi05.yaml",
    "vjepa2": "vjepa2.yaml",
    "vjepa2_ac": "vjepa2_ac.yaml",
}


def _repo_root() -> Path:
    # mstar package lives at <repo>/mstar/ ; configs/ sits at <repo>/configs/.
    return Path(mstar.__file__).resolve().parent.parent


def _resolve_config(model: str, override: str | None) -> str:
    if override:
        p = Path(override)
        if not p.exists():
            sys.exit(f"error: --config path not found: {override}")
        return str(p)
    if model not in DEFAULT_CONFIGS:
        avail = ", ".join(sorted(DEFAULT_CONFIGS))
        sys.exit(f"error: unknown model {model!r}. Known models: {avail}\n"
                 f"       (or pass --config <path.yaml> for a custom deployment)")
    candidate = _repo_root() / "configs" / DEFAULT_CONFIGS[model]
    if candidate.exists():
        return str(candidate)
    # Fall back to a CWD-relative configs/ (e.g. running from a checkout).
    cwd_candidate = Path("configs") / DEFAULT_CONFIGS[model]
    if cwd_candidate.exists():
        return str(cwd_candidate)
    sys.exit(f"error: default config for {model!r} not found at {candidate}")


def _client_host(host: str) -> str:
    # 0.0.0.0 binds all interfaces but isn't a connectable address for clients.
    return "localhost" if host in ("0.0.0.0", "") else host


def _next_steps(model: str, host: str, port: int) -> str:
    base = f"http://{_client_host(host)}:{port}"
    lines = [
        "",
        f"  ✓ mstar serving '{model}' on {base}",
        "",
        "  Python SDK:",
        "    from mstar import MStarClient",
        f"    client = MStarClient(\"{base}\")",
    ]
    if model in ("bagel", "bagel_cfg_parallel", "qwen3_omni"):
        lines.append("    print(client.chat(\"Hello!\").text)")
    if model in ("bagel", "bagel_cfg_parallel"):
        lines.append("    open(\"out.png\",\"wb\").write(client.generate_image(\"a cat in a hat\"))")
    if model == "qwen3_omni":
        lines.append("    client.chat(\"Say hi\", output_modalities=(\"text\",\"audio\")).save_audio(\"out.wav\")")
    if model in ("orpheus", "qwen3_omni"):
        voice = "tara" if model == "orpheus" else "Ethan"
        lines.append(f"    client.tts(\"Hello there\", voice=\"{voice}\").to_wav(\"out.wav\")")
    if model in ("pi05", "vjepa2", "vjepa2_ac"):
        lines.append("    res = client.generate(text=\"...\", output_modalities=(\"" +
                     ("action" if model == "pi05" else "video") + "\",))")

    # OpenAI-compatible snippet for the models that map to OpenAI semantics.
    if model in ("bagel", "qwen3_omni", "orpheus"):
        lines += ["", "  OpenAI-compatible:",
                  "    from openai import OpenAI",
                  f"    oai = OpenAI(base_url=\"{base}/v1\", api_key=\"none\")"]
        if model in ("bagel", "qwen3_omni"):
            lines.append(f"    oai.chat.completions.create(model=\"{model}\", "
                         "messages=[{\"role\":\"user\",\"content\":\"hi\"}])")
        if model in ("orpheus", "qwen3_omni"):
            voice = "tara" if model == "orpheus" else "Ethan"
            lines.append(f"    oai.audio.speech.create(model=\"{model}\", input=\"hi\", voice=\"{voice}\")")
        if model == "bagel":
            lines.append("    oai.images.generate(model=\"bagel\", prompt=\"a cat\")")
    lines.append("")
    return "\n".join(lines)


def _serve(args: argparse.Namespace) -> None:
    config = _resolve_config(args.model, args.config)
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    who = getpass.getuser() if hasattr(getpass, "getuser") else "mstar"
    socket_prefix = args.socket_path_prefix or f"/tmp/mstar_{who}/"
    upload_dir = args.upload_dir or f"/tmp/mstar_uploads_{who}/"

    argv = [
        "--config", config,
        "--host", args.host,
        "--port", str(args.port),
        "--socket-path-prefix", socket_prefix,
        "--upload-dir", upload_dir,
        "--tensor-comm-protocol", args.tensor_comm_protocol,
        "--log-level", args.log_level,
    ]
    if args.cache_dir:
        argv += ["--cache-dir", args.cache_dir]
    if args.log_stats:
        argv += ["--log-stats"]
    if args.log_stats_file:
        argv += ["--log-stats-file", args.log_stats_file]

    print(_next_steps(args.model, args.host, args.port), file=sys.stderr)

    from mstar.api_server.entrypoint import main as serve_main

    serve_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mstar", description="mstar multimodal inference CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="launch a server for a model (one command)")
    serve.add_argument("model", help=f"model name ({', '.join(sorted(DEFAULT_CONFIGS))}) or use --config")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1,2'")
    serve.add_argument("--config", default=None, help="override the default config (path to YAML)")
    serve.add_argument("--cache-dir", default=None, help="HuggingFace weight cache directory")
    serve.add_argument("--socket-path-prefix", default=None, help="ZMQ IPC socket prefix")
    serve.add_argument("--upload-dir", default=None, help="temp dir for uploaded media")
    serve.add_argument(
        "--tensor-comm-protocol", default="SHM",
        help="tensor transfer protocol (SHM is the safe single-node default; also TCP/RDMA)",
    )
    serve.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    serve.add_argument(
        "--log-stats", action="store_true",
        help="print per-request profiling stats when each request finishes",
    )
    serve.add_argument(
        "--log-stats-file", default=None,
        help="append per-request profiling stats to this file (implies --log-stats)",
    )
    serve.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
