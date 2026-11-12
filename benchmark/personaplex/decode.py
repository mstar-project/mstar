"""The PersonaPlex decoder node driven outside a server, chunk by chunk, as the codec partition drives it:
admit, plan, forward and commit over a real ``mimi_state`` pool, then postprocess."""

from types import SimpleNamespace

import torch

from mstar.model.personaplex import config as cfg


def schedule_sizes(total: int) -> list[int]:
    """The chunk sizes the stream releases for ``total`` frames (the last one short), plus an empty final
    chunk when ``total`` ends on a chunk boundary."""
    sizes, done, full = [], 0, 0
    ramp = list(cfg.STREAM_CHUNK_SCHEDULE)
    while done < total:
        full = ramp.pop(0) if ramp else cfg.STREAM_CHUNK_FRAMES
        sizes.append(min(full, total - done))
        done += sizes[-1]
    if not sizes or sizes[-1] == full:  # a full last chunk leaves before the stream ends
        sizes.append(0)
    return sizes


class DecoderDriver:
    """One decoder node and its ``mimi_state`` pool, sized from the decoder's own state blocks."""

    def __init__(self, decoder, max_slots: int = 16):
        from mstar.engine.resources.base import EngineResourceInfo
        from mstar.engine.resources.recurrent import RecurrentBlockConfig, RecurrentStateConfig, RecurrentStateSpec
        from mstar.engine.resources.recurrent.pool import RecurrentStatePool
        from mstar.engine.resources.runner import StepRunner

        blocks = {name: RecurrentBlockConfig(shape, dtype)
                  for name, (shape, dtype) in decoder.incremental.state_blocks().items()}
        spec = RecurrentStateSpec(resource_key=cfg.MIMI_STATE, nodes={cfg.MIMI_DECODER_NODE},
                                  config=RecurrentStateConfig(num_layers=1, blocks=blocks, max_slots=max_slots))
        self.decoder = decoder
        self.pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=decoder.get_device()))
        self.runner = StepRunner({cfg.MIMI_STATE: self.pool})

    def add(self, rid):
        self.pool.ingest_request(rid)

    def remove(self, rid):
        self.pool.remove_request(rid)
        self.decoder.cleanup_request(rid)

    def prepare(self, rid, num_samples, frames, final=False):
        """``frames`` (each ``[1, 8, 1]``) as the stream hands them over; None when there is nothing to run."""
        info = SimpleNamespace(step_metadata={"num_samples": num_samples})
        inputs = {"agent_frame": [torch.stack(frames) if len(frames) > 1 else frames[0]]} if frames else {}
        return self.decoder.prepare_inputs(cfg.CODEC_WALK, info, inputs, is_final_stream_chunk=final)

    def step(self, rids, node_inputs, batched=None) -> dict[str, torch.Tensor]:
        """One step over ``rids`` (each already prepared), in the engine's order; each row's emitted audio."""
        from mstar.engine.resources.step import StepContext
        from mstar.model.submodule_base import ModelInputsFromEngine

        step = self.decoder.declare_step(cfg.CODEC_WALK, list(rids), list(node_inputs))
        step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk=cfg.CODEC_WALK, slot=0, capture=False))
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        engine = ModelInputsFromEngine(request_ids=list(rids), per_request_info={}, step=step,
                                       resources={cfg.MIMI_STATE: self.pool})
        staged = self.decoder.preprocess(cfg.CODEC_WALK, engine, list(node_inputs))
        if batched if batched is not None else len(rids) > 1:
            rows = self.decoder.forward_batched(cfg.CODEC_WALK, engine, **staged).row_outputs["audio_output"]
            outputs = {rid: {"audio_output": [rows[i : i + 1]]} for i, rid in enumerate(rids)}
        else:
            outputs = {rids[0]: self.decoder.forward(cfg.CODEC_WALK, engine, **staged)}
        self.runner.commit(step)
        for rid, node in zip(rids, node_inputs, strict=True):
            self.decoder.postprocess(rid, None, outputs[rid], inputs=node)
        return {rid: outputs[rid]["audio_output"][0] for rid in rids}


def stream_decode(decoder, frames, info, rid="decode", sizes=None, driver=None):
    """Feed ``frames`` (each ``[1, 8, 1]``) through the decoder node in chunks of ``sizes`` (default: the
    stream's schedule); the last chunk is the final one. Returns the joined PCM."""
    sizes = schedule_sizes(len(frames)) if sizes is None else sizes
    assert sum(sizes) == len(frames), (sizes, len(frames))
    driver = driver or DecoderDriver(decoder)
    driver.add(rid)
    pieces, start = [], 0
    for i, size in enumerate(sizes):
        group = frames[start : start + size]
        start += size
        node = driver.prepare(rid, int(info.step_metadata["num_samples"]), group, final=i == len(sizes) - 1)
        if node is not None:
            pieces.append(driver.step([rid], [node])[rid])
    driver.remove(rid)
    return torch.cat(pieces, -1)
