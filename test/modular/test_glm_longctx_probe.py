"""benchmark/glm/longctx_probe.py: a length's reps are different prompts, each key at its depth (CPU)."""
import json
import sys
from types import SimpleNamespace

from benchmark.glm import longctx_probe as probe


class _WordTokenizer:
    def encode(self, text):
        return text.split()


def test_each_rep_draws_its_own_key_and_puts_it_at_the_next_depth(monkeypatch, tmp_path):
    tokenizer, sent = _WordTokenizer(), []

    def stream_one(url, model, text, *args):
        sent.append(text)
        return {"status": 200, "ttft_s": 0.1, "decode_ms_per_token": 1.0, "chunk_gap_p50_ms": 1.0,
                "completion_tokens": 1, "prompt_tokens": None,
                "text": probe.rep_key(0, 2000, 0) if len(sent) == 1 else "no idea"}

    monkeypatch.setattr(probe, "stream_one", stream_one)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _name: tokenizer)))
    out = tmp_path / "probe.jsonl"
    monkeypatch.setattr(sys, "argv", ["longctx_probe", "--model", "m", "--tokenizer", "t",
                                      "--lengths", "2000", "--reps", "4", "--output", str(out)])
    probe.main()

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["depth"] for r in rows] == [0.1, 0.5, 0.9, 0.1]
    assert len({r["key"] for r in rows}) == 4 and len(set(sent)) == 4
    assert [r["passkey_found"] for r in rows] == [True, False, False, False]
    for row, text in zip(rows, sent, strict=True):
        assert abs(row["prompt_tokens_counted"] / 2000 - 1) < 0.02
        filler = text[:text.index("\nWhat is")]
        at = filler.index(f"The pass key is {row['key']}.")
        # within one filler line of the requested depth
        assert abs(at / len(filler) - row["depth"]) < 2 * len(probe.FILLER) / len(filler)
