"""Standalone Qwen3-TTS codec harness: per-batch timing, kernel mix and streaming parity.

Builds the serving ``CodecSubmodule`` (same loader, Snake fusion and compiled
incremental decode as the server) over a ``RecurrentStatePool`` built from the
model's own ``codec_state`` declaration, its rows admitted and planned as the
engine does, and times ``CodecSubmodule._decode`` (gather, decode, scatter), so
codec changes can be measured in seconds instead of a server restart:

    python -m benchmark.qwen3tts_codec.codec_bench timing  --out DIR
    python -m benchmark.qwen3tts_codec.codec_bench kernels --out DIR
    python -m benchmark.qwen3tts_codec.codec_bench parity  --codes codes.pt --out DIR

``parity`` streams real code sequences through the served decode in the
model's chunk ramp and reports SNR against one whole-sequence float32 decode
(TF32 GEMMs and convs, matching a server whose codec runs in float32).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def load_codec(dtype: str = "float32"):
    from mstar.engine.engine import compile_node_regions
    from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel

    model = Qwen3TTSModel(MODEL, codec_dtype=dtype)
    model._ensure_full_snapshot()
    codec = model._create_codec_submodule("cuda")
    # as the engine does (the codec declares no fp32 precision and opts out of autocast)
    compile_node_regions("Codec", codec, None, None, "cuda")
    return model, codec


class ServedStep:
    """``batch`` streams on the codec's declared state pool, each step declared,
    admitted, planned and committed as the engine does it; ``decode`` is the
    served step (gather, compiled decode, scatter)."""

    def __init__(self, model, codec, batch: int):
        from mstar.engine.resources.base import EngineResourceInfo
        from mstar.engine.resources.recurrent.pool import RecurrentStatePool
        from mstar.engine.resources.runner import StepRunner
        from mstar.model.qwen3_tts.config import CODEC_STATE
        from mstar.model.submodule_base import ARNodeInputs

        self.codec, self.key = codec, CODEC_STATE
        spec = next(s for s in model.get_node_resources() if s.resource_key == CODEC_STATE)
        spec.apply_yaml_overrides(max_slots=batch + 1)
        self.pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cuda")))
        self.runner = StepRunner({CODEC_STATE: self.pool})
        self.rids = [f"r{i}" for i in range(batch)]
        frame = torch.zeros(1, codec.config.codec.num_quantizers, dtype=torch.long)
        self.rows = [ARNodeInputs(tensor_inputs={"codec_tokens": frame}) for _ in self.rids]
        for rid in self.rids:
            self.pool.ingest_request(rid)

    def plan(self) -> None:
        """Declare, admit and plan the next step of every stream."""
        from mstar.engine.resources import StepContext
        from mstar.model.submodule_base import ModelInputsFromEngine

        self.step = self.codec.declare_step("codec_chunk", self.rids, self.rows)
        self.step.set_ctx(StepContext(request_ids=tuple(self.rids), graph_walk="codec_chunk", slot=0, capture=False))
        assert self.runner.admit(self.step).ok
        self.runner.plan(self.step)
        self.engine_inputs = ModelInputsFromEngine(
            request_ids=self.rids, per_request_info={}, resources={self.key: self.pool}, step=self.step,
        )

    def decode(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``codes [B, Q, width]`` -> PCM16 ``[B, width * upsample]`` for the planned step."""
        return self.codec._decode(self.engine_inputs, codes, lengths)

    def commit(self) -> None:
        self.runner.commit(self.step)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _graph(fn, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def _time(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def step_fn(served: ServedStep, batch: int, width: int):
    """One planned step replayed: every replay reads the state the plan
    addresses, so the timing is a steady-state step's."""
    q = served.codec.config.codec.num_quantizers
    codes = torch.randint(0, 2048, (batch, q, width), device="cuda")
    lengths = torch.full((batch,), width, dtype=torch.long, device="cuda")
    return lambda: served.decode(codes, lengths)


def cmd_timing(args):
    torch.backends.cudnn.benchmark = True
    model, codec = load_codec(args.dtype)
    rows = []
    for batch in args.batches:
        served = ServedStep(model, codec, batch)
        served.plan()
        for width in sorted({*codec.windows, *args.extra_widths}):
            rows.append(dict(B=batch, width=width, ms=_time(_graph(step_fn(served, batch, width)), args.iters)))
            print(json.dumps(rows[-1]), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"timing_{args.dtype}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


# ---------------------------------------------------------------------------
# Kernel mix
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("layout", ("nchwToNhwc", "nhwcToNchw", "transpose", "Transpose")),
    ("conv", ("conv", "Conv", "implicit_gemm", "xmma", "cudnn", "sm90_", "dgrad", "wgrad")),
    ("gemm", ("gemm", "Gemm", "cutlass", "cublas", "matmul")),
    ("attention", ("flash", "fmha", "attention", "sdpa")),
    ("copy", ("copy", "Copy", "cat", "index", "gather", "scatter")),
    ("triton", ("triton_",)),
]


def _category(name: str) -> str:
    for cat, keys in CATEGORIES:
        if any(k in name for k in keys):
            return cat
    return "elementwise/other"


def cmd_kernels(args):
    from torch.profiler import ProfilerActivity, profile

    torch.backends.cudnn.benchmark = True
    model, codec = load_codec(args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    served = ServedStep(model, codec, args.batch)
    served.plan()
    cases = {f"served_w{width}": step_fn(served, args.batch, width) for width in codec.windows}
    report = {}
    for label, fn in cases.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        by_cat, by_name, count = defaultdict(float), defaultdict(float), 0
        for evt in prof.events():
            if evt.device_type.name != "CUDA":
                continue
            us = evt.device_time_total if hasattr(evt, "device_time_total") else evt.cuda_time_total
            by_cat[_category(evt.name)] += us / args.iters
            by_name[evt.name[:120]] += us / args.iters
            count += 1
        total = sum(by_cat.values())
        report[label] = {
            "kernels_per_step": count / args.iters,
            "gpu_us_per_step": total,
            "by_category_pct": {k: round(100 * v / total, 1) for k, v in sorted(by_cat.items(), key=lambda x: -x[1])},
            "top": sorted(((round(v, 1), k) for k, v in by_name.items()), reverse=True)[:25],
        }
        keys = ("kernels_per_step", "gpu_us_per_step", "by_category_pct")
        print(label, json.dumps({k: report[label][k] for k in keys}))
        prof.export_chrome_trace(str(out / f"trace_{label}_{args.dtype}_b{args.batch}.json"))
    (out / f"kernels_{args.dtype}_b{args.batch}.json").write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def stream_chunks(total: int, schedule, chunk: int) -> list[int]:
    """New frames per chunk of the model's ramp over a stream of ``total``
    frames, the producer done at the end (a short final flush)."""
    out, delivered, i = [], 0, 0
    while delivered < total:
        size = min(schedule[i] if i < len(schedule) else chunk, total - delivered)
        out.append(size)
        delivered += size
        i += 1
    return out


def snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    n = min(ref.numel(), test.numel())
    ref, test = ref[:n].double(), test[:n].double()
    noise = (ref - test).pow(2).sum().clamp_min(1e-20)
    return float(10 * torch.log10(ref.pow(2).sum() / noise))


@torch.no_grad()
def cmd_parity(args):
    """SNR of the served streamed decode against one whole-sequence float32
    decode (TF32 GEMMs and convs, matching a server whose codec runs in float32), both
    int16-quantized like the server's, and of the whole decode in ``--dtype``
    against it."""
    ref_model, ref_codec = load_codec("float32")
    model, codec = load_codec(args.dtype) if args.dtype != "float32" else (ref_model, ref_codec)
    cfg = codec.config.codec
    up = codec.total_upsample
    q = lambda x: (x.float().clamp(-1, 1) * 32767).round().clamp(-32768, 32767) / 32767  # noqa: E731
    seqs = torch.load(args.codes)   # list of [T, Q] long tensors (EOS stripped)
    results = []
    for n, codes in enumerate(seqs[: args.limit]):
        codes = codes.cuda().long()
        frames = codes.shape[0]
        full = q(ref_codec.decoder(codes.t()[None]).reshape(-1))
        full_dtype = q(codec.decoder(codes.t()[None]).reshape(-1))
        served, streamed, off = ServedStep(model, codec, 1), [], 0
        for new in stream_chunks(frames, cfg.chunk_schedule, cfg.chunk_frames):
            width = codec._bucket(new)
            chunk = torch.nn.functional.pad(codes[off: off + new].t()[None], (0, width - new))
            served.plan()
            pcm = served.decode(chunk, torch.tensor([new], device="cuda"))
            served.commit()
            streamed.append(pcm[0, : new * up].float() / 32767)
            off += new
        wav = torch.cat(streamed)
        assert wav.numel() == frames * up, (wav.numel(), frames * up)
        row = dict(
            item=n, frames=frames, whole_dtype_vs_full=snr_db(full, full_dtype), served_vs_full=snr_db(full, wav),
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    keys = [k for k in results[0] if k not in ("item", "frames")]
    summary = {k: dict(mean=sum(r[k] for r in results) / len(results), min=min(r[k] for r in results)) for k in keys}
    print("summary", json.dumps(summary))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"parity_{args.dtype}.json").write_text(json.dumps(dict(rows=results, summary=summary), indent=1))


@torch.no_grad()
def cmd_encode(args):
    """WAVs (24 kHz) -> list of ``[T, 16]`` code tensors via the speech tokenizer's encoder."""
    import soundfile
    from transformers import MimiConfig

    from mstar.model.loader import load_hf_weights
    from mstar.model.loader.iterators import iter_safetensors_shards
    from mstar.model.qwen3_tts.qwen3_tts_model import _load_buffers, _load_qwen3_tts_codec_classes

    model, _ = load_codec("float32")
    _, _, encoder_cls = _load_qwen3_tts_codec_classes()
    enc = encoder_cls(MimiConfig(**model.config.codec.encoder_config)).to("cuda")
    codec_dir = Path(model.local_dir) / "speech_tokenizer"

    def weights():
        for name, tensor in iter_safetensors_shards(codec_dir, device="cuda", prefix="encoder."):
            yield name.removeprefix("encoder."), tensor

    load_hf_weights(enc, weights())
    _load_buffers(enc, weights())
    enc.eval()
    seqs = []
    for path in sorted(Path(args.wav_dir).glob("*.wav"))[: args.limit]:
        wav, sr = soundfile.read(path, dtype="float32")
        if sr != 24000:
            raise ValueError(f"{path}: {sr} Hz, expected 24000")
        x = torch.from_numpy(wav).cuda().view(1, 1, -1)
        codes = enc.encode(input_values=x, return_dict=True).audio_codes[0, :16].t().contiguous().cpu()
        seqs.append(codes)
        print(path.name, tuple(codes.shape), flush=True)
    torch.save(seqs, args.out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("timing", "kernels", "parity"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--dtype", default="float32")
        s.add_argument("--iters", type=int, default=30)
    sub.choices["timing"].add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    sub.choices["timing"].add_argument("--extra-widths", type=int, nargs="+", default=[10, 12])
    sub.choices["kernels"].add_argument("--batch", type=int, default=16)
    sub.choices["parity"].add_argument("--codes", required=True)
    sub.choices["parity"].add_argument("--limit", type=int, default=20)
    e = sub.add_parser("encode")
    e.add_argument("--wav-dir", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=40)
    args = p.parse_args()
    with torch.inference_mode():
        {"timing": cmd_timing, "kernels": cmd_kernels, "parity": cmd_parity, "encode": cmd_encode}[args.cmd](args)


if __name__ == "__main__":
    main()
