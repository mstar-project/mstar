#!/usr/bin/env python3
"""Parity of the M* Qwen3-TTS port against the ``qwen-tts`` reference (GPU).

Three checks, all on real weights, all against the reference package that the
checkpoint ships with (``qwen_tts`` 0.1.1, bf16 Talker, fp32 codec):

1. **Teacher-forced Talker.** The reference generates ``--frames`` greedy codec
   frames and returns the Talker hidden state that produced each one. M* is
   driven through its served path (process_prompt -> prefill -> paged
   FlashInfer decode, one step per frame) while being fed the reference's
   frames, and its group-0 logits at every frame are compared with the
   reference's. The two sides build their prefill independently, so the first
   frame validates the prompt construction as well as the backbone. A second
   reference pass over M*'s own prefill (``backbone_only``) isolates the
   backbone from the prompt if the first comparison ever fails.
2. **Teacher-forced CodePredictor.** For every frame, M*'s depth loop receives
   the reference Talker hidden state and the reference codes and its 15 group
   logits are compared with the reference ``forward_finetune`` logits.
3. **Greedy end to end.** M* generates the same number of frames on its own
   (temperature 0 on both samplers, the checkpoint's repetition penalty) and
   the codes are compared frame by frame; both code sequences are decoded to
   audio by the codec on each side and the waveform max-abs-diff is reported.

Run inside the GPU allocation (weights must already be in the HF cache)::

    python test/qwen3-tts/parity_qwen3_tts.py --repo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --frames 64 --voice vivian --language English --json results/parity_1p7b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.resources import StepContext, StepRunner, apply_yaml_overrides, resolve_spec_dependencies
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
from mstar.model.submodule_base import ModelInputsFromEngine

GREEDY_KWARGS = {"do_sample": False, "subtalker_dosample": False}


# ---------------------------------------------------------------------------
# Checkpoint + reference
# ---------------------------------------------------------------------------


def default_deployment(config) -> str | None:
    """The deployment YAML ``mstar serve`` would use for this checkpoint variant."""
    root = Path(__file__).resolve().parents[2] / "configs"
    if config.is_base:
        name = "qwen3tts_base.yaml"
    elif config.is_voice_design:
        name = "qwen3tts_voicedesign.yaml"
    else:
        name = "qwen3tts.yaml" if config.tts_model_size == "0b6" else "qwen3tts_1p7b.yaml"
    path = root / name
    return str(path) if path.is_file() else None


def resolve_snapshot(repo: str) -> str:
    if Path(repo).is_dir():
        return repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True)


def load_reference(snapshot: str, device: str):
    """The reference stack: ``Qwen3TTSForConditionalGeneration`` + processor + codec."""
    from qwen_tts import Qwen3TTSModel as ReferenceModel

    return ReferenceModel.from_pretrained(
        snapshot, device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa",
    )


def reference_clone_prompt(ref, args):
    """The reference's voice-clone prompt (x-vector + codes) for ``--ref-audio``, or None."""
    if not args.ref_audio:
        return None
    items = ref.create_voice_clone_prompt(
        ref_audio=args.ref_audio, ref_text=args.ref_text, x_vector_only_mode=args.x_vector_only,
    )
    return ref._prompt_items_to_voice_clone_prompt(items), items[0]


def reference_generate(ref, args, clone=None) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Greedy reference codes ``[frames, groups]`` and the hidden states behind them.

    Uses the low-level ``generate`` of the reference so the prompt layout
    (speaker, language, instruct, reference clip, streaming vs non-streaming
    text) is exactly the reference's own, independent of M*'s
    ``process_prompt``.
    """
    model = ref.model
    device = model.device
    input_ids = ref._tokenize_texts([ref._build_assistant_text(args.text)])
    instruct_ids = None
    if args.instruct:
        instruct_ids = ref._tokenize_texts([ref._build_instruct_text(args.instruct)])
    speakers = [args.voice] if args.voice else None
    non_streaming = model.tts_model_type in ("custom_voice", "voice_design")
    if args.non_streaming_mode is not None:
        non_streaming = args.non_streaming_mode
    extra = {}
    if clone is not None:
        prompt_dict, item = clone
        extra = {
            "voice_clone_prompt": prompt_dict,
            "ref_ids": [ref._tokenize_texts([ref._build_ref_text(item.ref_text)])[0]] if item.ref_text else None,
        }
    codes_list, hidden_list = model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[args.language or "auto"],
        speakers=speakers,
        non_streaming_mode=non_streaming,
        max_new_tokens=args.frames,
        do_sample=False,
        subtalker_dosample=False,
        repetition_penalty=args.repetition_penalty,
        **extra,
    )
    codes = codes_list[0].to(device)
    if codes.shape[0] < args.frames:
        print(f"reference stopped at EOS after {codes.shape[0]} frames", file=sys.stderr)
    return codes, hidden_list[0]


def reference_frame_embeds(ref, codes: torch.Tensor) -> torch.Tensor:
    """Sum of the 16 codec embeddings per frame, in the Talker width."""
    talker = ref.model.talker
    embeds = talker.get_input_embeddings()(codes[:, 0])
    residual_tables = talker.code_predictor.get_input_embeddings()
    for group in range(1, codes.shape[1]):
        embeds = embeds + residual_tables[group - 1](codes[:, group])
    return embeds


def reference_teacher_forced(ref, prefill: torch.Tensor, frame_embeds: torch.Tensor, trailing: torch.Tensor,
                             tts_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference group-0 logits and normed hidden per frame over one full sequence."""
    talker = ref.model.talker
    steps = frame_embeds.shape[0]
    text_cond = torch.stack([
        trailing[t] if t < trailing.shape[0] else tts_pad for t in range(steps)
    ])
    # Frame t's logits come from the position holding frame t-1's input; the
    # prefill's last position predicts frame 0, so the last frame input is
    # not needed as an input at all.
    inputs = torch.cat([prefill, (frame_embeds + text_cond)[:-1]], dim=0).unsqueeze(0)
    out = talker.model(inputs_embeds=inputs, use_cache=False)
    hidden = out.last_hidden_state[0, prefill.shape[0] - 1:]
    return talker.codec_head(hidden), hidden


def reference_code_predictor_logits(ref, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Reference residual-group logits ``[frames, groups-1, vocab]`` (teacher forced)."""
    talker = ref.model.talker
    cp = talker.code_predictor
    first = talker.get_input_embeddings()(codes[:, :1])
    residual = [cp.get_input_embeddings()[g - 1](codes[:, g:g + 1]) for g in range(1, codes.shape[1] - 1)]
    inputs = torch.cat([hidden.unsqueeze(1), first, *residual], dim=1)
    return cp.forward_finetune(inputs_embeds=inputs).logits


# ---------------------------------------------------------------------------
# M* side: the served path without the worker around it
# ---------------------------------------------------------------------------


class MStarTalkerDriver:
    """Drive ``TalkerSubmodule`` through declare -> admit -> plan -> forward -> commit.

    Mirrors ``Engine._drive_step`` for one eager request, on resources built
    from the model's own ``get_node_resources`` declaration.
    """

    def __init__(self, model: Qwen3TTSModel, talker, device: str, deployment: str | None = None,
                 max_num_pages: int = 64):
        self.model = model
        self.talker = talker
        self.device = torch.device(device)
        specs = model.get_node_resources()
        by_key = resolve_spec_dependencies(specs)
        if deployment is not None:
            # The served deployment's resource overrides (e.g. the FA2 pin), so
            # the harness runs the same kernels as ``mstar serve``.
            import yaml

            apply_yaml_overrides(specs, yaml.safe_load(Path(deployment).read_text(encoding="utf-8")))
        for spec in specs:
            if hasattr(spec, "apply_yaml_overrides") and hasattr(spec.config, "max_num_pages"):
                spec.apply_yaml_overrides(max_num_pages=max_num_pages)
        groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        transfer = TransferEngineInfo("h", "h", LocalTransferEngine("h"))
        self.resources = {
            spec.resource_key: build_resource(
                spec,
                EngineResourceInfo(
                    device=self.device,
                    joint_comm_group=groups,
                    transfer_engine_info=transfer,
                    kv_dtype=torch.bfloat16,
                    dependencies={key: by_key[key] for key in spec.depends_on()},
                ),
            )
            for spec in specs
        }
        talker.bind_node_resources(self.resources)
        self.runner = StepRunner(self.resources)
        self.fwd_index = 0

    def open_request(self, rid: str, model_kwargs: dict[str, Any]) -> CurrentForwardPassInfo:
        configs = self.model.get_request_resource_configs({}, model_kwargs)
        for config in configs.values():
            config.apply_conductor_config(seed=1234)
        self.runner.ingest_request(rid, configs)
        return CurrentForwardPassInfo(
            request_id=rid, graph_walk="talker_prefill", fwd_index=0, random_seed=1234,
            max_tokens=8192, resource_configs=configs,
            step_metadata={"talker_max_tokens": 8192, "is_prefill": True},
        )

    def close_request(self, rid: str) -> None:
        self.runner.remove_request(rid)
        self.talker.cleanup_request(rid)

    def step(self, walk: str, fwd: CurrentForwardPassInfo, inputs: dict, forward):
        """One step; ``forward(engine_inputs, **preprocessed)`` runs the compute."""
        rid = fwd.request_id
        if walk == "talker_prefill" and ("speaker_embed" in inputs or "ref_codes" in inputs):
            walk = "talker_prefill_clone"
        fwd.graph_walk = walk
        prepared = self.talker.prepare_inputs(walk, fwd, inputs)
        step = self.talker.declare_step(walk, [rid], [prepared])
        step.set_ctx(StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False))
        outcome = self.runner.admit(step)
        assert outcome.ok, f"admit failed: {outcome.reason}"
        self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=[rid], per_request_info={rid: fwd}, resources=dict(self.resources),
            per_request_states={rid: self.talker.request_state(rid)}, step=step,
        )
        preprocessed = self.talker.preprocess(walk, engine_inputs, [prepared])
        out = forward(engine_inputs, **preprocessed)
        self.runner.commit(step)
        return out


@torch.no_grad()
def mstar_teacher_forced(driver: MStarTalkerDriver, tensors: dict, frame_embeds: torch.Tensor,
                         greedy_kwargs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """M* group-0 logits and normed hidden per frame, fed the reference frames."""
    talker = driver.talker
    rid = "teacher-forced"
    fwd = driver.open_request(rid, greedy_kwargs)
    logits, hiddens = [], []

    def backbone(engine_inputs, input_embeds, last_token_indices, suppress_eos):
        del engine_inputs, suppress_eos
        hidden = talker.model(input_embeds, label="main")
        last = hidden.index_select(0, last_token_indices)
        hiddens.append(last[0])
        logits.append(talker.model.codec_head(last)[0])

    driver.step("talker_prefill", fwd, tensors, backbone)
    for t in range(frame_embeds.shape[0] - 1):
        # prepare_inputs adds the text condition for this step itself.
        driver.step("talker_decode", fwd, {"talker_input_embeds": [frame_embeds[t:t + 1]]}, backbone)
    driver.close_request(rid)
    return torch.stack(logits), torch.stack(hiddens)


@torch.no_grad()
def mstar_code_predictor_logits(talker, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """M* residual-group logits per frame, fed the reference hidden and codes."""
    out = []
    for t in range(codes.shape[0]):
        collected = []
        target = iter(codes[t, 1:].tolist())

        def teacher(cp_logits, collected=collected, target=target):
            collected.append(cp_logits[0])
            return cp_logits.new_tensor([next(target)], dtype=torch.long)

        talker._depth_loop(hidden[t:t + 1], codes[t:t + 1, 0], teacher)
        out.append(torch.stack(collected))
    return torch.stack(out)


@torch.no_grad()
def mstar_greedy(driver: MStarTalkerDriver, tensors: dict, frames: int, greedy_kwargs: dict) -> torch.Tensor:
    """M*'s own greedy generation through ``TalkerSubmodule.forward``."""
    talker = driver.talker
    rid = "greedy"
    fwd = driver.open_request(rid, greedy_kwargs)
    codes = []

    def forward(engine_inputs, **kw):
        return talker.forward(fwd.graph_walk, engine_inputs, **kw)

    out = driver.step("talker_prefill", fwd, tensors, forward)
    codes.append(out["codec_tokens"][0][0])
    talker.postprocess(rid, fwd, out)
    eos = talker.talker_config.codec_eos_token_id
    while len(codes) < frames and int(codes[-1][0]) != eos:
        out = driver.step("talker_decode", fwd, {"talker_input_embeds": out["talker_input_embeds"]}, forward)
        codes.append(out["codec_tokens"][0][0])
        talker.postprocess(rid, fwd, out)
    driver.close_request(rid)
    codes = torch.stack(codes)
    return codes[codes[:, 0] != eos]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_logits(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> dict[str, Any]:
    ours = ours.float()
    theirs = theirs.float()
    diff = (ours - theirs).abs()
    agree = (ours.argmax(-1) == theirs.argmax(-1)).float()
    top2 = theirs.topk(2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1])
    return {
        "name": name,
        "positions": int(agree.numel()),
        "argmax_agreement": float(agree.mean()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "ref_logit_scale": float(theirs.abs().mean()),
        # disagreements should sit on near-ties: report the reference top-2
        # margin where the argmax differs
        "disagreement_margin_median": (
            float(margin[agree == 0].median()) if (agree == 0).any() else None
        ),
    }


def compare_codes(ours: torch.Tensor, theirs: torch.Tensor) -> dict[str, Any]:
    n = min(ours.shape[0], theirs.shape[0])
    equal_frames = (ours[:n] == theirs[:n]).all(dim=1)
    first_div = int(equal_frames.logical_not().nonzero()[0]) if not equal_frames.all() else n
    return {
        "frames_mstar": int(ours.shape[0]),
        "frames_reference": int(theirs.shape[0]),
        "frames_compared": n,
        "identical_frames_before_divergence": first_div,
        "group0_agreement": float((ours[:n, 0] == theirs[:n, 0]).float().mean()),
        "all_groups_agreement": float(equal_frames.float().mean()),
    }


@torch.no_grad()
def decode_audio(codec, codes: torch.Tensor) -> torch.Tensor:
    """M* codec: ``[frames, groups]`` -> float waveform in [-1, 1]."""
    wav = codec.decoder(codes.t().unsqueeze(0).contiguous())
    return wav.squeeze().float()


@torch.no_grad()
def reference_decode_audio(snapshot: str, device: str, codes: torch.Tensor) -> torch.Tensor:
    """Reference codec in float32 (M* runs its codec in float32; the reference
    wrapper would otherwise inherit the Talker's bf16)."""
    from qwen_tts import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(Path(snapshot) / "speech_tokenizer"), device_map=device, dtype=torch.float32,
    )
    wavs, _ = tokenizer.decode([{"audio_codes": codes}])
    return torch.as_tensor(wavs[0]).float()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument(
        "--text",
        default="The quick brown fox jumps over the lazy dog while the sun sets behind the hills.",
    )
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--non-streaming-mode", type=lambda s: s.lower() == "true", default=None)
    parser.add_argument("--ref-audio", default=None, help="Base: reference clip (voice clone)")
    parser.add_argument("--ref-text", default=None, help="Base: transcript of the reference clip")
    parser.add_argument("--x-vector-only", action="store_true", help="Base: skip in-context frames")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default=None,
                        help="deployment YAML for resource overrides (default: the variant's)")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if args.voice == "":
        args.voice = None

    torch.manual_seed(0)
    snapshot = resolve_snapshot(args.repo)
    t0 = time.perf_counter()
    ref = load_reference(snapshot, args.device)
    print(f"reference loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    model = Qwen3TTSModel(model_path_hf=snapshot)
    talker = model.get_submodule("Talker", device=args.device, autocast_dtype=torch.bfloat16)
    codec = model.get_submodule("Codec", device=args.device)
    deployment = args.config or default_deployment(model.config)
    print(f"resource overrides from {deployment}", file=sys.stderr)
    driver = MStarTalkerDriver(model, talker, args.device, deployment=deployment)
    print(f"M* loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    request_kwargs = {"language": args.language, **GREEDY_KWARGS,
                      "repetition_penalty": args.repetition_penalty}
    if args.voice:
        request_kwargs["voice"] = args.voice
    if args.instruct:
        request_kwargs["instruct"] = args.instruct
    if args.non_streaming_mode is not None:
        request_kwargs["non_streaming_mode"] = args.non_streaming_mode
    clone = reference_clone_prompt(ref, args)
    clone_report = None
    if clone is None:
        tensors = model.process_prompt(args.text, ["text"], ["audio"], **request_kwargs)
    else:
        # Voice clone: M*'s load_audio -> process_prompt -> RefEncoder, compared
        # with the reference's own x-vector and codec frames for the same clip.
        clip = model.load_audio(args.ref_audio, args.device)
        request_kwargs.update({"ref_text": args.ref_text, "x_vector_only_mode": args.x_vector_only})
        tensors = model.process_prompt(
            args.text, ["audio", "text"], ["audio"], tensors={"audio_inputs": [clip.data]}, **request_kwargs,
        )
        ref_encoder = model.get_submodule("RefEncoder", device=args.device, autocast_dtype=torch.bfloat16)
        prepared = ref_encoder.prepare_inputs(
            "talker_prefill_clone", CurrentForwardPassInfo(
                request_id="clone", graph_walk="talker_prefill_clone", fwd_index=0, random_seed=0, max_tokens=0,
            ), {"audio_inputs": [clip.data], "prompt_layout": tensors["prompt_layout"]},
        )
        with torch.no_grad():
            encoded = ref_encoder.forward(
                "talker_prefill_clone", ModelInputsFromEngine(request_ids=["clone"], per_request_info={}),
                **ref_encoder.preprocess("talker_prefill_clone", None, [prepared]),
            )
        tensors["speaker_embed"] = encoded["speaker_embed"]
        tensors["ref_codes"] = encoded["ref_codes"]
        _, item = clone
        their_xvec = item.ref_spk_embedding.to(args.device).float()
        our_xvec = encoded["speaker_embed"][0].float()
        clone_report = {
            "xvector_cosine": float(torch.nn.functional.cosine_similarity(our_xvec, their_xvec, dim=0)),
            "xvector_max_abs_diff": float((our_xvec - their_xvec).abs().max()),
            "xvector_scale": float(their_xvec.abs().mean()),
        }
        if item.ref_code is not None:
            their_codes = item.ref_code.to(args.device)
            our_codes = encoded["ref_codes"][0]
            n = min(their_codes.shape[0], our_codes.shape[0])
            clone_report.update({
                "ref_frames_mstar": int(our_codes.shape[0]),
                "ref_frames_reference": int(their_codes.shape[0]),
                "ref_code_agreement": float((our_codes[:n] == their_codes[:n]).float().mean()),
            })
        del ref_encoder
        torch.cuda.empty_cache()

    # 1 + 2: teacher forced against the reference's greedy frames. ``ref_hidden``
    # is the hidden state the reference's own generation used for each frame.
    ref_codes, ref_hidden = reference_generate(ref, args, clone)
    ref_hidden = ref_hidden.to(torch.bfloat16)
    frames = ref_codes.shape[0]
    theirs_logits = ref.model.talker.codec_head(ref_hidden)
    frame_embeds = reference_frame_embeds(ref, ref_codes)
    ours_logits, ours_hidden = mstar_teacher_forced(driver, tensors, frame_embeds, request_kwargs)
    talker_report = compare_logits("talker_group0_logits", ours_logits, theirs_logits)
    hidden_report = {
        "name": "talker_hidden",
        "max_abs_diff": float((ours_hidden.float() - ref_hidden.float()).abs().max()),
        "rel_diff": float((ours_hidden.float() - ref_hidden.float()).norm() / ref_hidden.float().norm()),
    }
    # Diagnostic: the reference backbone over M*'s own prefill embeddings.
    prefill = talker._build_prefill(
        "layout-probe", tensors["text_inputs"][0], tensors["prompt_layout"][0],
        int(tensors["speaker_id"][0]), int(tensors["language_id"][0]),
        speaker_embed=tensors.get("speaker_embed", [None])[0],
        ref_codes=tensors.get("ref_codes", [None])[0],
    )
    probe_state = talker.request_state("layout-probe")
    backbone_logits, _ = reference_teacher_forced(
        ref, prefill, frame_embeds, probe_state["trailing_text_hidden"], probe_state["tts_pad_embed"],
    )
    talker.cleanup_request("layout-probe")
    backbone_report = compare_logits("backbone_only_on_mstar_prefill", ours_logits, backbone_logits)
    cp_ours = mstar_code_predictor_logits(talker, ref_hidden, ref_codes)
    cp_theirs = reference_code_predictor_logits(ref, ref_hidden, ref_codes)
    cp_report = compare_logits("code_predictor_logits", cp_ours, cp_theirs)

    # 3: greedy end to end + audio.
    ours_codes = mstar_greedy(driver, tensors, frames, request_kwargs)
    codes_report = compare_codes(ours_codes, ref_codes)
    n = codes_report["frames_compared"]
    audio_ref_codes_mstar = decode_audio(codec, ref_codes[:n])
    audio_ref_codes_ref = reference_decode_audio(snapshot, args.device, ref_codes[:n]).to(audio_ref_codes_mstar.device)
    m = min(audio_ref_codes_mstar.numel(), audio_ref_codes_ref.numel())
    codec_report = {
        "name": "codec_same_codes",
        "samples": m,
        "max_abs_diff": float((audio_ref_codes_mstar[:m] - audio_ref_codes_ref[:m]).abs().max()),
        "length_mismatch": int(audio_ref_codes_mstar.numel() - audio_ref_codes_ref.numel()),
    }
    audio_ours = decode_audio(codec, ours_codes[:n])
    k = min(audio_ours.numel(), audio_ref_codes_ref.numel())
    e2e_audio = {
        "name": "audio_greedy_e2e",
        "samples": k,
        "max_abs_diff": float((audio_ours[:k] - audio_ref_codes_ref[:k]).abs().max()),
        "snr_db": float(10 * torch.log10(
            audio_ref_codes_ref[:k].pow(2).mean() / ((audio_ours[:k] - audio_ref_codes_ref[:k]).pow(2).mean() + 1e-12)
        )),
    }

    report = {
        "repo": args.repo, "snapshot": snapshot, "text": args.text, "voice": args.voice,
        "language": args.language, "instruct": args.instruct, "frames": int(ref_codes.shape[0]),
        "clone": clone_report,
        "talker": talker_report, "talker_hidden": hidden_report, "backbone_only": backbone_report,
        "code_predictor": cp_report,
        "greedy_codes": codes_report, "codec": codec_report, "audio": e2e_audio,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = talker_report["argmax_agreement"] >= 0.99 and cp_report["argmax_agreement"] >= 0.99 \
        and codec_report["max_abs_diff"] < 1e-3
    print("PARITY OK" if ok else "PARITY FAILED", file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
