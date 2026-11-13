"""The in-process backend: one process holds a model's text encoder, DiT submodule and VAE decoder, and drives each
request's denoise loop through the scaffold's engine contract (``prepare_inputs``, ``preprocess``, ``forward``), the
way the reference-equivalence tests do. The step cache policy is swapped between runs, so a sweep over many settings
pays one model load.

There is no worker and no engine resource in the loop: attention runs SDPA, nothing is speculated, and one request
runs at a time. The math of a policy is the served one, so quality rankings carry over; the seconds are this loop's
own, and the points that matter are to be confirmed through the ``server`` backend.

``tiny`` builds Wan2.2's DiT at toy size with random weights on the CPU and decodes latents with a fixed random
projection: enough to run the whole harness without a GPU or a checkpoint (its tests, a dry run of a sweep), and
nothing more.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from typing import Any, NamedTuple
from unittest import mock

import torch

from benchmark.step_cache.backends.base import Generated
from benchmark.step_cache.methods import Method, plan_cost
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.diffusion.step_cache import StepCachePolicy, step_cache_policy
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.wan22.config import MAG_RATIOS, WAN22_VARIANT_T2V_A14B, WAN22_VARIANT_TI2V_5B, Wan22Config
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, Wan22DitSubmodule
from mstar.model.wan22.wan22_model import Wan22Model

_REQUEST = "bench-r0"
_WAN_REPOS = {
    "wan22_ti2v_5b": (WAN22_VARIANT_TI2V_5B, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    "wan22_t2v_a14b": (WAN22_VARIANT_T2V_A14B, "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
}
# Wan2.2's DiT at toy size, as test/modular/test_step_cache.py builds it
_TINY = dict(
    num_attention_heads=2, attention_head_dim=12, num_layers=4, ffn_dim=32,
    in_channels=4, out_channels=4, text_dim=8, freq_dim=16, rope_max_seq_len=64, compile_dit=False,
)


class LoopResult(NamedTuple):
    """One request's denoise loop."""

    latents: torch.Tensor
    seconds: float
    # the policy's action per step; None for a dense run
    plan: tuple | None
    info: CurrentForwardPassInfo


# Called after each step with (step index, the request's state, the step's outputs by edge name).
Observer = Callable[[int, Any, dict[str, list[torch.Tensor]]], None]


def _drive(submodule, walk: str, info: CurrentForwardPassInfo, inputs: dict) -> dict[str, list[torch.Tensor]]:
    """One engine-shaped pass over one request."""
    node_inputs = submodule.prepare_inputs(graph_walk=walk, fwd_info=info, inputs=inputs)
    engine_inputs = ModelInputsFromEngine(request_ids=[info.request_id], per_request_info={info.request_id: info})
    kwargs = submodule.preprocess(graph_walk=walk, engine_inputs=engine_inputs, inputs=[node_inputs])
    return submodule.forward(graph_walk=walk, engine_inputs=engine_inputs, **kwargs)


class _Wan:
    """Wan2.2 from its checkpoint: the DiT submodule, with the text encoder and the VAE decoder beside it."""

    def __init__(self, model_name: str, model_path: str | None, device: torch.device, compile_dit: bool,
                 request_extra: dict[str, Any] | None = None):
        variant, repo = _WAN_REPOS[model_name]
        # request knobs every run sends (a flow shift, a guidance scale), as a client's body would
        self.request_extra = dict(request_extra or {})
        self.model = Wan22Model(model_path_hf=model_path or repo, variant=variant)
        self.model.config.compile_dit = compile_dit
        self.device = device
        self.dit: Wan22DitSubmodule = self.model.get_submodule("dit", device=str(device)).to(device)
        self.text = self.model.get_submodule("text_encoder").to(device)
        self.decoder = self.model.get_submodule("vae_decoder").to(device)
        self.tables = MAG_RATIOS[variant]
        self._embeds: dict[str, dict[str, list[torch.Tensor]]] = {}

    def metadata(self, size: str, frames: int, steps: int) -> dict[str, Any]:
        """The request's step metadata, as ``Wan22Model.walk_plan`` builds it."""
        width, height = (int(v) for v in size.lower().split("x"))
        request = {
            **self.request_extra, "height": height, "width": width, "num_frames": frames, "num_inference_steps": steps,
        }
        self.model._validate_generation_size(request)
        knobs = self.model._resolve_denoise_knobs({**request, "negative_prompt": ""})
        return {**request, **{k: v for k, v in knobs.items() if k != "negative_prompt"}}

    def embed(self, prompt: str, info: CurrentForwardPassInfo) -> dict[str, list[torch.Tensor]]:
        if prompt not in self._embeds:
            tokens = self.model.process_prompt(
                prompt=prompt, input_modalities=["text"], output_modalities=["video"], negative_prompt="",
            )
            out = _drive(self.text, Wan22Model.ENCODE_TEXT_WALK, info, {"text_inputs": tokens["text_inputs"]})
            self._embeds[prompt] = {name: out[name] for name in ("text_embeds_pos", "text_embeds_neg")}
        return self._embeds[prompt]

    def decode(self, latents: torch.Tensor, info: CurrentForwardPassInfo) -> torch.Tensor:
        if self.device.type == "cuda":
            out = _drive(self.decoder, Wan22Model.VIDEO_GEN_WALK, info, {"latents": [latents]})
        else:
            # the decoder sizes its tiling from free VRAM; on the CPU (a smoke run) there is none to ask about
            with mock.patch.object(torch.cuda, "mem_get_info", lambda device=None: (1 << 40, 1 << 40)):
                out = _drive(self.decoder, Wan22Model.VIDEO_GEN_WALK, info, {"latents": [latents]})
        return out["video_output"][0].cpu()


class _TinyWan:
    """Wan2.2's DiT at toy size with random weights, on the CPU. Text embeddings are noise seeded by the prompt and
    the decode is a fixed projection of the latents to pixels."""

    def __init__(self):
        from mstar.model.wan22.components.dit import Wan22DiT

        self.device = torch.device("cpu")
        config = Wan22Config(**_TINY)
        torch.manual_seed(0)
        dit = Wan22DiT(config)
        for param in dit.parameters():
            torch.nn.init.normal_(param, std=0.2)
        self.dit = Wan22DitSubmodule(transformer=dit.eval(), config=config)
        self.dit.get_device = lambda: self.device
        self.tables = {"default": (1.0, *([0.98] * 49))}
        self._pixels = torch.randn(3, config.in_channels, generator=torch.Generator().manual_seed(1))

    def metadata(self, size: str, frames: int, steps: int) -> dict[str, Any]:
        width, height = (int(v) for v in size.lower().split("x"))
        return {
            "height": height, "width": width, "num_frames": frames, "num_inference_steps": steps,
            "guidance_scale": 5.0,
        }

    def embed(self, prompt: str, info: CurrentForwardPassInfo) -> dict[str, list[torch.Tensor]]:
        seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "little")
        gen = torch.Generator().manual_seed(seed)
        return {name: [torch.randn(1, 5, _TINY["text_dim"], generator=gen)]
                for name in ("text_embeds_pos", "text_embeds_neg")}

    def decode(self, latents: torch.Tensor, info: CurrentForwardPassInfo) -> torch.Tensor:
        # [1, C, T, H, W] -> uint8 [T, 8H, 8W, 3]
        pixels = torch.einsum("pc,cthw->thwp", self._pixels, latents[0].float())
        pixels = pixels.repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)
        return (pixels.tanh() * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)


class InprocBackend:
    name = "inproc"

    def __init__(self, model: str = "wan22_ti2v_5b", *, model_path: str | None = None, device: str | None = None,
                 compile_dit: bool = False, request_extra: dict[str, Any] | None = None):
        self.model_name = model
        if model == "tiny":
            self.adapter = _TinyWan()
        elif model in _WAN_REPOS:
            device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.adapter = _Wan(model, model_path, device, compile_dit, request_extra)
        else:
            raise ValueError(f"the inproc backend runs {sorted(_WAN_REPOS)} and 'tiny'; got {model!r}")
        self.compile_dit = compile_dit
        self.policy: StepCachePolicy | None = None
        # clip shapes that have run once: a shape's first forward and decode pay one-time costs no later one does
        self._warm: set[tuple[str, int]] = set()

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model-path", default=None, help="inproc: a local checkpoint dir instead of the hub id")
        parser.add_argument("--device", default=None, help="inproc: the device (default: cuda when there is one)")
        parser.add_argument("--compile-dit", action="store_true", help="inproc: torch.compile the DiT, as served")
        parser.add_argument("--request-extra", type=json.loads, default=None,
                            help='inproc: request knobs for every run, as JSON (e.g. \'{"flow_shift": 12}\')')

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> InprocBackend:
        return cls(
            args.model, model_path=args.model_path, device=args.device, compile_dit=args.compile_dit,
            request_extra=args.request_extra,
        )

    def describe(self) -> dict[str, Any]:
        device = self.adapter.device
        return {
            "backend": self.name, "model": self.model_name, "torch": torch.__version__,
            "cuda": torch.version.cuda if device.type == "cuda" else None,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "attention": "sdpa (no engine resources in process)", "compile_dit": self.compile_dit,
        }

    # ------------------------------------------------------------------ policies
    def policy_of(self, spec: dict[str, Any] | None) -> StepCachePolicy | None:
        """The policy a server would build from ``spec`` for this model."""
        return step_cache_policy(spec, self.adapter.tables)

    def use(self, policy: StepCachePolicy | None) -> None:
        """Run the next loops under ``policy`` (None: dense), refused as the model's constructor would refuse it."""
        dit = self.adapter.dit
        dit.check_step_cache(policy)
        dit.step_cache = self.policy = policy

    def configure(self, method: Method) -> None:
        self.use(self.policy_of(method.step_cache))

    # ------------------------------------------------------------------ one request
    def request_info(self, seed: int, *, size: str, frames: int, steps: int) -> CurrentForwardPassInfo:
        return CurrentForwardPassInfo(
            request_id=_REQUEST, graph_walk=Wan22Model.VIDEO_GEN_WALK, fwd_index=0, random_seed=seed, max_tokens=0,
            step_metadata=self.adapter.metadata(size, frames, steps), dynamic_loop_iter_counts={DENOISE_LOOP_NAME: 0},
        )

    def restarts(self, seed: int = 0, **request) -> tuple[int, ...]:
        """The steps where the model changes weights for this request (A14B's expert switch)."""
        dit, info = self.adapter.dit, self.request_info(seed, **request)
        return dit.step_cache_segments(info, dit.bucket_key_for(info))

    def step_times(self, forecast, seed: int = 0, **request) -> list[float]:
        """What ``forecast`` spaces this request's steps by: their index, or the schedule's sigma."""
        if forecast.time == "step":
            return [float(k) for k in range(request["steps"])]
        dit, info = self.adapter.dit, self.request_info(seed, **request)
        return list(dit.schedule_of(info, dit.bucket_key_for(info)).sigmas.tolist())[: request["steps"]]

    def run_loop(self, prompt: str, seed: int, *, size: str, frames: int, steps: int,
                 observer: Observer | None = None) -> LoopResult:
        """The request's denoise loop under the current policy; ``observer`` sees each step's state and outputs."""
        dit, device = self.adapter.dit, self.adapter.device
        info = self.request_info(seed, size=size, frames=frames, steps=steps)
        with torch.no_grad():
            cond = self.adapter.embed(prompt, info)
            dit.request_states.pop(_REQUEST, None)
            carried: dict[str, list[torch.Tensor]] = {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for k in range(steps):
                info.dynamic_loop_iter_counts[DENOISE_LOOP_NAME] = k
                out = _drive(dit, Wan22Model.VIDEO_GEN_WALK, info, {**cond, **carried})
                carried = {name: out[name] for name in dit.loop_back_names}
                if observer is not None:
                    observer(k, dit.request_state(_REQUEST), out)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
        plan = None
        if self.policy is not None:
            bucket = dit.bucket_key_for(info)
            plan = self.policy.plan(steps, dit.step_cache_segments(info, bucket))
        dit.request_states.pop(_REQUEST, None)
        return LoopResult(carried[dit.LATENTS][0], seconds, plan, info)

    def generate(self, prompt: str, seed: int, *, size: str, frames: int, steps: int) -> Generated:
        device = self.adapter.device
        if device.type == "cuda" and (size, frames) not in self._warm:
            self._warm.add((size, frames))
            warm = self.run_loop(prompt, seed, size=size, frames=frames, steps=min(steps, 2))
            with torch.no_grad():
                self.adapter.decode(warm.latents, warm.info)
        loop = self.run_loop(prompt, seed, size=size, frames=frames, steps=steps)
        start = time.perf_counter()
        with torch.no_grad():
            video = self.adapter.decode(loop.latents, loop.info)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter() - start
        dit = self.adapter.dit
        guided = float(loop.info.step_metadata["guidance_scale"]) > 1.0
        num_blocks = len(dit.transformer.blocks)
        if loop.plan is None:
            computed, forwards = steps, float(steps * (2 if guided else 1))
        else:
            computed, forwards = plan_cost(loop.plan, num_blocks, guided)
        return Generated(
            frames=video, seconds=loop.seconds + decode_seconds, dit_seconds=loop.seconds, computed_steps=computed,
            branch_forwards=forwards,
            extra={
                "decode_seconds": decode_seconds,
                "plan": None if loop.plan is None else "".join("-" if a.reuse else ("C" if a.uncond else "c")
                                                               for a in loop.plan),
            },
        )

    def close(self) -> None:
        self.adapter = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
