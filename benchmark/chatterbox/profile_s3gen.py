"""Micro-benchmarks of the S3Gen stages on one GPU, without a server.

Answers, for the standard (10-step CFG flow) and Turbo (2-step mean flow)
decoders: how long one estimator call and one whole flow solve take as a
function of batch rows and mel frames; how much of that is kernel launch
overhead (kernel count and GPU busy time from ``torch.profiler``, and the
same call replayed from a CUDA graph); what TF32, bfloat16 and float16 buy
and what they cost in mel/waveform deviation; what the flow encoder, the HiFT
vocoder and the Perth watermark (CPU resampling vs. a device-only path) add
per chunk. Writes a markdown report.

Usage::

    python benchmark/chatterbox/profile_s3gen.py --variant standard --out results/<date>/micro_standard.md
    python benchmark/chatterbox/profile_s3gen.py --variant turbo --quick
"""

from __future__ import annotations

import argparse
import copy
import functools
import math
import os
import statistics
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mstar.model.chatterbox.components.s3gen import FlowRow, ReferenceConditioning, S3Gen  # noqa: E402
from mstar.model.chatterbox.config import S3GenConfig  # noqa: E402
from mstar.model.chatterbox.loader import iter_weights, resolve_snapshot  # noqa: E402

VARIANTS = {
    "standard": ("ResembleAI/chatterbox", "s3gen.safetensors", False, 10),
    "turbo": ("ResembleAI/chatterbox-turbo", "s3gen_meanflow.safetensors", True, 2),
}
SAMPLE_RATE = 24_000
DTYPES = {"fp32": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------

CUDA = torch.cuda.is_available()


def _sync() -> None:
    if CUDA:
        torch.cuda.synchronize()


def time_ms(fn, iters: int = 10, warmup: int = 3) -> tuple[float, float]:
    """Median GPU-event time and median CPU wall time (ms) of ``fn()``
    (on the CPU both are wall time)."""
    for _ in range(warmup):
        fn()
    _sync()
    gpu, wall = [], []
    for _ in range(iters):
        if CUDA:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        t0 = time.perf_counter()
        fn()
        if CUDA:
            end.record()
        _sync()
        wall.append((time.perf_counter() - t0) * 1000)
        gpu.append(start.elapsed_time(end) if CUDA else wall[-1])
    return statistics.median(gpu), statistics.median(wall)


def capture(fn, warmup: int = 2):
    """Warm ``fn`` on a side stream, then capture it; returns the graph
    (None on the CPU, where the caller times ``fn`` again instead)."""
    if not CUDA:
        return None, fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def kernel_profile(fn, label: str) -> list[str]:
    """Kernel count, GPU busy time and the top kernels of one ``fn()`` call."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    _sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        fn()
        _sync()
        wall = (time.perf_counter() - t0) * 1000
    kernels = [
        e for e in prof.key_averages()
        if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type)
    ]
    count = sum(e.count for e in kernels)
    busy_us = sum(getattr(e, "self_device_time_total", 0.0) for e in kernels)
    lines = [f"**{label}**: {count} kernel launches, GPU busy {busy_us / 1000:.1f} ms of {wall:.1f} ms wall "
             f"({100 * busy_us / 1000 / max(wall, 1e-9):.0f} % busy)", "",
             "kernel | launches | GPU ms | share", "--- | --- | --- | ---"]
    top = sorted(kernels, key=lambda e: -getattr(e, "self_device_time_total", 0.0))[:12]
    for e in top:
        us = getattr(e, "self_device_time_total", 0.0)
        name = e.key if len(e.key) <= 90 else e.key[:87] + "..."
        lines.append(f"`{name}` | {e.count} | {us / 1000:.2f} | {100 * us / max(busy_us, 1e-9):.0f} %")
    return lines


def snr_db(reference: torch.Tensor, other: torch.Tensor) -> float:
    reference, other = reference.float().flatten(), other.float().flatten()
    n = min(reference.numel(), other.numel())
    noise = (reference[:n] - other[:n]).pow(2).mean()
    if noise == 0:
        return math.inf
    return float(10 * torch.log10(reference[:n].pow(2).mean() / noise))


def log_mel_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load(variant: str, device: str) -> tuple[S3Gen, ReferenceConditioning, int]:
    repo, filename, meanflow, n_steps = VARIANTS[variant]
    snap = Path(resolve_snapshot(repo))
    config = S3GenConfig.meanflow_distilled() if meanflow else S3GenConfig.standard()
    s3gen = S3Gen(config)
    s3gen.load_weights(iter_weights(snap / filename))
    s3gen = s3gen.to(device).eval()
    conds = torch.load(snap / "conds.pt", map_location="cpu", weights_only=True)["gen"]
    ref = ReferenceConditioning(
        prompt_tokens=conds["prompt_token"], prompt_feat=conds["prompt_feat"], embedding=conds["embedding"],
    ).to(device)
    return s3gen, ref, n_steps


class Estimators:
    """The estimator in every precision under test; ``tf32`` shares fp32 weights."""

    def __init__(self, s3gen: S3Gen):
        base = s3gen.decoder.estimator
        self.modules = {
            "fp32": base, "tf32": base,
            "bf16": copy.deepcopy(base).to(torch.bfloat16),
            "fp16": copy.deepcopy(base).to(torch.float16),
        }

    def __getitem__(self, name: str):
        return self.modules[name]


def set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def solve(
    s3gen: S3Gen, estimator, dtype: torch.dtype, mu, mask, spks, cond, noise, n_steps: int,
) -> torch.Tensor:
    """The decoder's Euler solve with the estimator run in ``dtype`` and the
    state kept in float32 (what a mixed-precision production path would do)."""
    cfm = s3gen.decoder
    meanflow = s3gen.config.meanflow
    x = noise.float()
    b = mu.shape[0]
    t_span = cfm.time_grid(n_steps, mu.device, torch.float32, meanflow=meanflow)
    mu_d, spks_d, cond_d, mask_d = (v.to(dtype) for v in (mu, spks, cond, mask))
    if not meanflow:
        rate = cfm.config.inference_cfg_rate
        mu_in = torch.cat([mu_d, torch.zeros_like(mu_d)], dim=0)
        spks_in = torch.cat([spks_d, torch.zeros_like(spks_d)], dim=0)
        cond_in = torch.cat([cond_d, torch.zeros_like(cond_d)], dim=0)
        mask_in = torch.cat([mask_d, mask_d], dim=0)
    for t, r in zip(t_span[:-1], t_span[1:], strict=True):
        if meanflow:
            dxdt = estimator(x.to(dtype), mask_d, mu_d, t.to(dtype).expand(b), spks_d, cond_d, r=r.to(dtype).expand(b))
        else:
            x_in = x.to(dtype)
            dxdt = estimator(
                torch.cat([x_in, x_in], dim=0), mask_in, mu_in, t.to(dtype).expand(2 * b), spks_in, cond_in,
            )
            dxdt, cfg_dxdt = dxdt.float().split([b, b], dim=0)
            dxdt = (1.0 + rate) * dxdt - rate * cfg_dxdt
        x = x + (r - t) * dxdt.float()
    return x


def estimator_inputs(s3gen: S3Gen, batch: int, frames: int, dtype: torch.dtype, device: str) -> dict:
    """Random inputs of the shape one estimator call sees (``batch`` rows, all valid)."""
    out_ch = s3gen.config.output_size
    g = torch.Generator(device=device).manual_seed(0)
    return {
        "x": torch.randn(batch, out_ch, frames, device=device, generator=g).to(dtype),
        "mask": torch.ones(batch, 1, frames, device=device, dtype=dtype),
        "mu": torch.randn(batch, out_ch, frames, device=device, generator=g).to(dtype),
        "t": torch.full((batch,), 0.3, device=device, dtype=dtype),
        "spks": torch.randn(batch, out_ch, device=device, generator=g).to(dtype),
        "cond": torch.randn(batch, out_ch, frames, device=device, generator=g).to(dtype),
    }


def call_estimator(estimator, inputs: dict, meanflow: bool):
    kwargs = {"r": inputs["t"] + 0.1} if meanflow else {}
    return estimator(inputs["x"], inputs["mask"], inputs["mu"], inputs["t"], inputs["spks"], inputs["cond"], **kwargs)


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def section_estimator(s3gen: S3Gen, ests: Estimators, device: str, quick: bool) -> list[str]:
    meanflow = s3gen.config.meanflow
    batches = [2, 4, 16] if quick else [1, 2, 4, 8, 16]
    frames = [416, 736] if quick else [352, 416, 512, 736, 1024]
    if quick == "smoke":
        batches, frames = [2], [352]
    lines = ["## One estimator call", "",
             "Rows are estimator rows (the standard model's CFG doubles the request rows; Turbo does not). "
             "`eager` is GPU-event time of a plain call, `graph` the same call replayed from a CUDA graph; "
             "`wall` is the CPU time of the eager call, so wall ≫ graph means launch-bound.", "",
             "precision | rows | frames | eager ms | wall ms | graph ms | eager / graph",
             "--- | --- | --- | --- | --- | --- | ---"]
    for name in ("fp32", "tf32", "bf16", "fp16"):
        set_tf32(name == "tf32")
        est = ests[name]
        for b in batches:
            for f in frames:
                inputs = estimator_inputs(s3gen, b, f, DTYPES[name], device)
                fn = functools.partial(call_estimator, est, inputs, meanflow)
                eager, wall = time_ms(fn)
                graph, _ = capture(fn)
                replay, _ = time_ms(graph.replay if graph is not None else fn)
                del graph
                lines.append(f"{name} | {b} | {f} | {eager:.2f} | {wall:.2f} | {replay:.2f} | {eager / replay:.1f}×")
    set_tf32(False)
    return lines


def section_kernels(s3gen: S3Gen, ests: Estimators, device: str) -> list[str]:
    meanflow = s3gen.config.meanflow
    lines = ["## Kernel profile of one estimator call (2 rows, 416 frames)", ""]
    for name in ("fp32", "bf16"):
        set_tf32(False)
        inputs = estimator_inputs(s3gen, 2, 416, DTYPES[name], device)
        lines += kernel_profile(functools.partial(call_estimator, ests[name], inputs, meanflow), f"estimator {name}")
        lines.append("")
    return lines


def _rows_for(ref: ReferenceConditioning, n_rows: int, chunk_tokens: int, finalize: bool, device: str, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    vocab = 6561
    return [
        FlowRow(
            tokens=torch.randint(0, vocab, (chunk_tokens,), device=device, generator=g),
            ref=ref, finalize=finalize, generator=torch.Generator(device=device).manual_seed(seed + i),
        )
        for i in range(n_rows)
    ]


def section_solve(s3gen: S3Gen, ests: Estimators, ref: ReferenceConditioning, n_steps: int, device: str,
                  quick: bool) -> list[str]:
    """Whole flow solves on realistic chunk plans through ``tokens_to_mel_rows``
    (flow encoder included), eager fp32 as today's path, then the estimator in
    each precision, then the solve loop captured as one CUDA graph."""
    prompt_frames = ref.prompt_feat.shape[1]
    plans = [(1, 15, False), (1, 50, False), (8, 50, False), (8, 200, True)] if quick else [
        (1, 15, False), (1, 50, False), (1, 200, True), (4, 50, False), (8, 50, False), (8, 200, False), (8, 200, True),
    ]
    if quick == "smoke":
        plans = [(2, 15, False)]
    lines = [f"## Whole flow solve per chunk ({n_steps} steps, prompt {prompt_frames} frames)", "",
             "`today` is `S3Gen.tokens_to_mel_rows` as served (fp32 eager, flow encoder + solve). "
             "The precision columns run the same solve with the estimator in that precision and the state in fp32 "
             "(encoder excluded); `graph` captures the whole fp32-state / bf16-estimator solve as one CUDA graph.", "",
             "request rows | chunk tokens | final | frames | today ms | fp32 ms | tf32 ms | bf16 ms | fp16 ms | "
             "bf16 graph ms | Δmel tf32 | Δmel bf16 | Δmel fp16 | mel corr bf16 | wav SNR bf16 dB",
             "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---"]
    for n_rows, chunk, final in plans:
        rows = _rows_for(ref, n_rows, chunk, final, device, seed=7)
        set_tf32(False)
        today, _ = time_ms(functools.partial(s3gen.tokens_to_mel_rows, rows, n_timesteps=n_steps), iters=5, warmup=2)
        # the solve's inputs, built once the way tokens_to_mel_rows builds them
        mu, mask, spks, cond, noise = _solve_inputs(s3gen, rows)
        frames = mu.shape[-1]
        timings, mels = {}, {}
        for name in ("fp32", "tf32", "bf16", "fp16"):
            set_tf32(name == "tf32")
            fn = functools.partial(solve, s3gen, ests[name], DTYPES[name], mu, mask, spks, cond, noise, n_steps)
            timings[name], _ = time_ms(fn, iters=5, warmup=2)
            mels[name] = fn()
        set_tf32(False)
        fn = functools.partial(solve, s3gen, ests["bf16"], torch.bfloat16, mu, mask, spks, cond, noise, n_steps)
        graph, _ = capture(fn)
        graph_ms, _ = time_ms(graph.replay if graph is not None else fn, iters=10)
        del graph
        gen = slice(prompt_frames, None)
        base = mels["fp32"][:, :, gen]
        deltas = {k: float((mels[k][:, :, gen] - base).abs().max()) for k in ("tf32", "bf16", "fp16")}
        corr = log_mel_corr(base, mels["bf16"][:, :, gen])
        wav_ref = s3gen.mel_to_wav(base[:1], generator=torch.Generator(device=device).manual_seed(1))
        wav_bf16 = s3gen.mel_to_wav(mels["bf16"][:1, :, gen], generator=torch.Generator(device=device).manual_seed(1))
        lines.append(
            f"{n_rows} | {chunk} | {'yes' if final else 'no'} | {frames} | {today:.0f} | {timings['fp32']:.0f} | "
            f"{timings['tf32']:.0f} | {timings['bf16']:.0f} | {timings['fp16']:.0f} | {graph_ms:.0f} | "
            f"{deltas['tf32']:.3f} | {deltas['bf16']:.3f} | {deltas['fp16']:.3f} | {corr:.4f} | "
            f"{snr_db(wav_ref, wav_bf16):.1f}"
        )
    return lines


def _solve_inputs(s3gen: S3Gen, rows: list[FlowRow]):
    """Mirror of the tensor set-up in ``S3Gen.tokens_to_mel_rows`` up to the solve."""
    from mstar.model.chatterbox.components.s3gen_flow import lengths_to_mask

    ratio = s3gen.config.token_mel_ratio
    lookahead = s3gen.config.encoder.pre_lookahead_len * ratio
    refs = [row.ref.to(s3gen.device, s3gen.dtype) for row in rows]
    full = [
        torch.cat([ref.prompt_tokens[0], row.tokens.to(s3gen.device, torch.long)])
        for ref, row in zip(refs, rows, strict=True)
    ]
    full_lens = torch.tensor([f.numel() for f in full], dtype=torch.long, device=s3gen.device)
    tokens = torch.nn.utils.rnn.pad_sequence(full, batch_first=True)
    spk = s3gen.flow_encoder.project_speaker(torch.cat([ref.embedding for ref in refs], dim=0))
    mu, h_masks = s3gen.flow_encoder(tokens, full_lens)
    cuts = torch.tensor([0 if row.finalize else lookahead for row in rows], device=s3gen.device)
    h_lens = (h_masks.sum(dim=-1).squeeze(-1) - cuts).clamp_min(0)
    total = mu.shape[-1]
    cond = torch.zeros(len(rows), s3gen.config.output_size, total, device=s3gen.device, dtype=mu.dtype)
    noise = torch.zeros_like(cond)
    for i, (ref, row) in enumerate(zip(refs, rows, strict=True)):
        mel_prompt = ref.num_prompt_tokens * ratio
        valid = int(h_lens[i])
        cond[i, :, :mel_prompt] = ref.prompt_feat[0].transpose(0, 1)
        noise[i, :, :valid] = s3gen._draw_noise(1, mel_prompt, valid - mel_prompt, row.generator)[0]
    mask = lengths_to_mask(h_lens, total).unsqueeze(1).to(mu.dtype)
    return mu, mask, spk, cond, noise


def section_encoder_vocoder(s3gen: S3Gen, ref: ReferenceConditioning, device: str, smoke: bool = False) -> list[str]:
    lines = ["## Flow encoder and HiFT vocoder", "",
             "stage | rows | tokens or mel frames | eager ms | wall ms", "--- | --- | --- | --- | ---"]
    prompt = ref.prompt_tokens.shape[1]
    for n_rows in ((1,) if smoke else (1, 8)):
        for chunk in ((15,) if smoke else (15, 50, 200)):
            tokens = torch.randint(0, 6561, (n_rows, prompt + chunk), device=device)
            lens = torch.full((n_rows,), prompt + chunk, device=device)
            eager, wall = time_ms(functools.partial(s3gen.flow_encoder, tokens, lens))
            lines.append(f"flow encoder | {n_rows} | {prompt + chunk} | {eager:.2f} | {wall:.2f}")
    cache = 8
    sweep = (2 * 15 + cache, 2 * 50 + cache, 2 * 100 + cache, 2 * 200 + cache, 800)
    for frames in ((2 * 15 + cache,) if smoke else sweep):
        mel = torch.randn(1, s3gen.config.output_size, frames, device=device) * 2 - 6
        g = torch.Generator(device=device).manual_seed(0)
        eager, wall = time_ms(functools.partial(s3gen.vocode, mel, generator=g))
        lines.append(f"HiFT vocode | 1 | {frames} | {eager:.2f} | {wall:.2f}")
    lines += ["", "HiFT draws its excitation noise inside `vocode`, so a graph capture would have to take the noise as "
              "an input; the flow encoder's length varies per chunk (bucketing needed)."]
    return lines


def section_watermark(device: str, wav: torch.Tensor) -> list[str]:
    """Perth as served today (network on the device, librosa resampling and a
    numpy round trip on the CPU) against a device-only path."""
    try:
        import perth
    except ImportError:
        return ["## Watermark", "", "resemble-perth is not installed"]
    import numpy as np
    import torchaudio.functional as taf

    from mstar.model.chatterbox.components.watermark import PerthWatermarker

    seconds = wav.numel() / SAMPLE_RATE
    lines = [f"## Perth watermark on {seconds:.1f} s of 24 kHz audio", "",
             "path | ms per utterance | detected | SNR vs unmarked dB | SNR vs today dB", "--- | --- | --- | --- | ---"]
    served = PerthWatermarker.build(device)
    impl = served._impl
    net = impl.perth_net
    perth_sr = net.hp.sample_rate

    def today() -> torch.Tensor:
        return served.apply(wav, SAMPLE_RATE)

    def device_only() -> torch.Tensor:
        x = taf.resample(wav.float(), SAMPLE_RATE, perth_sr)
        mag, phase = net.ap.signal_to_magphase(x)
        wm_mag, _ = net.encoder(mag[None])
        y = net.ap.magphase_to_signal(wm_mag[0], phase)
        y = taf.resample(y, perth_sr, SAMPLE_RATE)[: wav.numel()]
        return y

    detector = perth.PerthImplicitWatermarker(device="cpu")

    def detected(x: torch.Tensor) -> str:
        score = detector.get_watermark(x.detach().cpu().float().numpy(), sample_rate=SAMPLE_RATE)
        return f"{float(np.asarray(score).mean()):.2f}"

    marked_today = today()
    ms_today = statistics.median([_wall(today) for _ in range(5)])
    marked_dev = device_only()
    ms_dev = statistics.median([_wall(device_only) for _ in range(5)])
    lines.append(f"today (numpy + librosa on CPU, network on {device}) | {ms_today:.1f} | {detected(marked_today)} | "
                 f"{snr_db(wav, marked_today):.1f} | -")
    lines.append(f"device only (torchaudio resample) | {ms_dev:.1f} | {detected(marked_dev)} | "
                 f"{snr_db(wav, marked_dev):.1f} | {snr_db(marked_today, marked_dev):.1f}")
    lines.append(f"unmarked | - | {detected(wav)} | - | -")
    return lines


def _wall(fn) -> float:
    _sync()
    t0 = time.perf_counter()
    fn()
    _sync()
    return (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=list(VARIANTS), default="standard")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", help="markdown report path (default: stdout)")
    parser.add_argument("--quick", action="store_true", help="smaller sweeps")
    parser.add_argument("--smoke", action="store_true", help="tiny sweeps, runs on the CPU too (logic check)")
    parser.add_argument("--sections", default="kernels,estimator,solve,encoder,watermark")
    args = parser.parse_args()
    if not CUDA and not args.smoke:
        raise SystemExit("needs a CUDA device (or --smoke for a CPU logic check)")
    if not CUDA:
        args.device = "cpu"
    quick = "smoke" if args.smoke else args.quick
    torch.manual_seed(0)
    s3gen, ref, n_steps = load(args.variant, args.device)
    ests = Estimators(s3gen)
    name = torch.cuda.get_device_properties(args.device).name if CUDA else "CPU"
    report = [f"# S3Gen micro-benchmarks: {args.variant} ({name}, torch {torch.__version__})", ""]
    wanted = set(args.sections.split(","))
    with torch.no_grad():
        if "kernels" in wanted and CUDA:
            report += section_kernels(s3gen, ests, args.device) + [""]
        if "estimator" in wanted:
            report += section_estimator(s3gen, ests, args.device, quick) + [""]
        if "solve" in wanted:
            report += section_solve(s3gen, ests, ref, n_steps, args.device, quick) + [""]
        if "encoder" in wanted:
            report += section_encoder_vocoder(s3gen, ref, args.device, smoke=args.smoke) + [""]
        if "watermark" in wanted:
            rows = _rows_for(ref, 1, 15 if args.smoke else 100, True, args.device, seed=3)
            mel = s3gen.tokens_to_mel_rows(rows, n_timesteps=n_steps)[0]
            wav = s3gen.mel_to_wav(mel, generator=torch.Generator(device=args.device).manual_seed(0))[0]
            report += section_watermark(args.device, wav) + [""]
    text = "\n".join(report) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
