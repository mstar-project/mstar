"""CPU parity harness for MSTAR_FAST_CHECKSTOP_TALKER (Item A, speech-floor).

The batched talker stop check in ``worker._postprocess_batch`` replaces the
per-request ``TalkerSubmodule.check_stop`` (one ``layer0_codes.item()`` + attr
chain per rid) with one ``flat.tolist()`` over a pinned buffer + pure-int
compares. This test does NOT need a GPU: it drives both the reference formula
(copied verbatim from ``TalkerSubmodule.check_stop``) and the fast batched
formula (copied from the worker fast path) over the same synthetic batch and
asserts the resulting stop sets are identical for every request.

Run:  python -m pytest test/modular/test_talker_fast_checkstop_parity.py -q
  or:  python test/modular/test_talker_fast_checkstop_parity.py
"""

import torch

CODEC_EOS = 2150  # config.py talker.codec_eos_token_id default


class _Info:
    """Minimal stand-in for CurrentForwardPassInfo (only the fields the two
    stop formulas read)."""

    def __init__(self, iters, max_tokens, talker_max_tokens=None):
        self.dynamic_loop_iter_counts = {"talker_decode_loop": iters}
        self.max_tokens = max_tokens
        self.step_metadata = (
            {} if talker_max_tokens is None
            else {"talker_max_tokens": talker_max_tokens}
        )


def _reference_stop(info, token, eos_id):
    """Verbatim TalkerSubmodule.check_stop condition (submodules.py:2353)."""
    max_tokens = info.step_metadata.get("talker_max_tokens", info.max_tokens)
    if (eos_id is not None and eos_id == token) or (
        info.dynamic_loop_iter_counts.get("talker_decode_loop", 0) + 1
        >= max_tokens
    ):
        return {"talker_decode_loop"}
    return set()


def _fast_batched_stops(infos, code_tensors, eos_id):
    """Verbatim worker N1-Talker fast path: one cat + tolist + int compares."""
    flat_gpu = torch.cat([t.reshape(1) for t in code_tensors])
    flat_cpu = flat_gpu.to("cpu")  # stands in for the pinned D->H copy
    tokens = flat_cpu.tolist()
    new_stops = {}
    for i, info in enumerate(infos):
        max_tokens = info.step_metadata.get("talker_max_tokens", info.max_tokens)
        if (
            (eos_id is not None and int(tokens[i]) == eos_id)
            or info.dynamic_loop_iter_counts.get("talker_decode_loop", 0) + 1
            >= max_tokens
        ):
            new_stops[i] = {"talker_decode_loop"}
    return new_stops


def test_parity_matrix():
    # Batch mixes: an eos code, a normal code, at-max iters, under-max iters,
    # and a per-request talker_max_tokens override that fires before max_tokens.
    cases = [
        # (iters, max_tokens, talker_max_tokens, token)
        (0, 100, None, CODEC_EOS),      # eos -> stop
        (0, 100, None, 42),             # normal, plenty of budget -> continue
        (99, 100, None, 42),            # iter+1 == max_tokens -> stop
        (98, 100, None, 42),            # iter+1 < max_tokens -> continue
        (4, 100, 5, 42),                # talker_max_tokens override fires -> stop
        (3, 100, 5, 42),                # under override -> continue
        (50, 100, None, CODEC_EOS),     # eos AND under budget -> stop (eos wins)
        (0, 1, None, 7),                # max_tokens==1, iter 0 -> stop
    ]
    infos = [_Info(it, mx, tmt) for (it, mx, tmt, _tok) in cases]
    codes = [torch.tensor([tok], dtype=torch.long) for (*_r, tok) in cases]

    fast = _fast_batched_stops(infos, codes, CODEC_EOS)
    for i, (info, tok) in enumerate(zip(infos, [c.item() for c in codes], strict=False)):
        ref = _reference_stop(info, tok, CODEC_EOS)
        got = fast.get(i, set())
        assert ref == got, (
            f"case {i} token={tok}: reference={ref} fast={got}"
        )

    # Also exercise eos_id=None (degenerate config): only max_tokens can stop.
    infos_n = [_Info(0, 100), _Info(99, 100)]
    codes_n = [torch.tensor([CODEC_EOS]), torch.tensor([CODEC_EOS])]
    fast_n = _fast_batched_stops(infos_n, codes_n, None)
    assert fast_n.get(0, set()) == _reference_stop(infos_n[0], CODEC_EOS, None)
    assert fast_n.get(1, set()) == _reference_stop(infos_n[1], CODEC_EOS, None)


if __name__ == "__main__":
    test_parity_matrix()
    print("talker fast-checkstop parity: OK")
