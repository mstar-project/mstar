"""Grammar-constrained sampling: a request's constraint on which token ids it
may draw next, advanced by the tokens it drew (JSON schema, JSON object, regex,
choice list, EBNF; xgrammar underneath).

The sampler resource owns it, beside its other per-request draw state. A step
that holds a constrained row stages each row's bitmask before it launches, from
the tokens the rows drew in their last step: the sampler copies those to the
host at commit, and the next step's plan waits for that copy alone (step N's
draw, not the worker's postprocess), accepts them and fills the masks. The draw
then applies the mask before either sampling path, so a captured step bakes the
apply over the slot's static mask buffer.
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mstar.model.utils import TextOutput

logger = logging.getLogger(__name__)

# The request field: vLLM's ``structured_outputs`` shape (one of these keys), plus
# ``after``, the text that opens the constrained part (a reasoning model's think-end tag).
STRUCTURED_OUTPUTS_KWARG = "structured_outputs"
GRAMMAR_KINDS = ("json", "json_object", "regex", "choice", "grammar")


def _xgr():
    try:
        import xgrammar
    except ImportError as e:  # pragma: no cover - a dependency
        raise ValueError("structured output needs the xgrammar package") from e
    return xgrammar


@dataclass(frozen=True)
class GrammarRequest:
    """A request's constraint: ``kind`` (one of ``GRAMMAR_KINDS``) and its text
    (a JSON schema as JSON, a regex, an EBNF grammar, a choice list as JSON).
    With ``after``, the request draws freely until its output holds that text,
    and the grammar holds from there: a structural tag with that one trigger."""
    kind: str
    value: str
    after: str | None = None

    def _regex(self) -> str:
        return "|".join(_regex_escape(c) for c in json.loads(self.value))

    def _content(self):
        xgr = _xgr()
        if self.kind == "json":
            return xgr.Grammar.from_json_schema(self.value)
        if self.kind == "json_object":
            return xgr.Grammar.builtin_json_grammar()
        if self.kind in ("regex", "choice"):
            return xgr.Grammar.from_regex(self.value if self.kind == "regex" else self._regex())
        return xgr.Grammar.from_ebnf(self.value)

    def _structural_tag(self, content) -> str:
        """Free text without ``after``, then, if the reply goes on, ``after`` and
        ``content``: a reply that ends before the trigger is not held to it."""
        constrained = {"type": "sequence", "elements": [
            {"type": "const_string", "value": self.after}, {"type": "grammar", "grammar": str(content)},
        ]}
        return json.dumps({"type": "structural_tag", "format": {"type": "sequence", "elements": [
            {"type": "any_text", "excludes": [self.after]}, {"type": "optional", "content": constrained},
        ]}})

    def grammar(self):
        """The xgrammar ``Grammar``; tokenizer-free, so it checks the request
        anywhere. ``ValueError`` names what is unsupported."""
        try:
            content = self._content()
            if self.after is None:
                return content
            return _xgr().Grammar.from_structural_tag(self._structural_tag(content))
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"structured_outputs {self.kind}: {e}") from e

    def compile(self, compiler):
        """Compiled against ``compiler``'s vocabulary, through the calls its cache keys by text."""
        if self.after is not None:
            return compiler.compile_structural_tag(self._structural_tag(self._content()))
        if self.kind == "json":
            return compiler.compile_json_schema(self.value)
        if self.kind == "json_object":
            return compiler.compile_builtin_json_grammar()
        if self.kind in ("regex", "choice"):
            return compiler.compile_regex(self.value if self.kind == "regex" else self._regex())
        return compiler.compile_grammar(self.value)


def _regex_escape(text: str) -> str:
    return "".join("\\" + c if c in r"\^$.|?*+()[]{}/" else c for c in text)


def grammar_request(value: Any) -> GrammarRequest | None:
    """The request's ``structured_outputs`` as a ``GrammarRequest``, checked; None
    when unset. ``ValueError`` (the client's 400) for a bad shape or grammar."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"structured_outputs must be an object; got {value!r}")
    unknown = set(value) - {*GRAMMAR_KINDS, "after"}
    if unknown:
        raise ValueError(f"structured_outputs takes one of {GRAMMAR_KINDS} and 'after'; got {sorted(unknown)}")
    kinds = [k for k in GRAMMAR_KINDS if value.get(k) not in (None, False)]
    if len(kinds) != 1:
        raise ValueError(f"structured_outputs takes exactly one of {GRAMMAR_KINDS}; got {kinds or 'none'}")
    kind, raw = kinds[0], value[kinds[0]]
    if kind == "json":
        if isinstance(raw, (dict, bool)):
            raw = json.dumps(raw)
        elif not isinstance(raw, str):
            raise ValueError("structured_outputs json must be a JSON schema")
    elif kind == "json_object":
        raw = ""
    elif kind == "choice":
        if not isinstance(raw, list) or not raw or not all(isinstance(c, str) for c in raw):
            raise ValueError("structured_outputs choice must be a non-empty list of strings")
        raw = json.dumps(raw)
    elif not isinstance(raw, str):
        raise ValueError(f"structured_outputs {kind} must be a string")
    after = value.get("after")
    if after is not None and (not isinstance(after, str) or not after):
        raise ValueError("structured_outputs after must be a non-empty string")
    request = GrammarRequest(kind, raw, after)
    request.grammar()
    return request


def response_format_request(response_format: Any) -> dict | None:
    """OpenAI ``response_format`` as a ``structured_outputs`` value; None for
    text. ``ValueError`` for another type."""
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise ValueError(f"response_format must be an object; got {response_format!r}")
    kind = response_format.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return {"json_object": True}
    if kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict) or spec.get("schema") is None:
            raise ValueError("response_format json_schema needs json_schema.schema")
        return {"json": spec["schema"]}
    raise ValueError(f"response_format type must be text, json_object or json_schema; got {kind!r}")


class _Row:
    """One request's matcher, compiled in the background: the first step that
    needs it waits for the compile (usually done by then: the prefill runs first)."""
    __slots__ = ("_compiled", "_matcher", "active")

    def __init__(self, compiled: Future):
        self._compiled = compiled
        self._matcher = None
        self.active = True

    @property
    def matcher(self):
        if self._matcher is None:
            try:
                self._matcher = _xgr().GrammarMatcher(self._compiled.result())
            except Exception:
                # checked tokenizer-free at the conductor already; a compile that
                # still fails leaves the row free rather than failing its batch
                logger.exception("structured_outputs grammar failed to compile; the row draws freely")
                self.active = False
                raise
        return self._matcher

    @property
    def constrained(self) -> bool:
        if not self.active:
            return False
        try:
            return not self.matcher.is_terminated()
        except Exception:
            return False

    def accept(self, token: int) -> None:
        if not self.constrained:
            return
        if not self.matcher.accept_token(token):
            # the mask allowed it, so this is a row the mask did not cover; its
            # output is not checked
            logger.warning("grammar refused token %d; the row's constraint is off from here", token)
            self.active = False


class GrammarRows:
    """The sampler's constrained rows: one matcher per request, the drawn tokens
    still on their way to the host, and the bitmask each step applies."""

    def __init__(self, text_output: TextOutput, vocab_size: int | None, device: torch.device):
        xgr = _xgr()
        self._xgr = xgr
        detok = text_output.detokenizer
        stop = sorted(int(t) for t in text_output.stop_token_ids)
        tokenizer = getattr(detok, "tokenizer", None)
        if tokenizer is not None:
            size = vocab_size or len(tokenizer)
            self.info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=size, stop_token_ids=stop or None)
        else:
            if vocab_size is None:
                raise ValueError("enable_grammar needs the SamplerSpec's vocab_size for this model's detokenizer")
            size = vocab_size
            vocab = [detok.token_bytes(i) for i in range(size)]
            self.info = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, vocab_size=size, stop_token_ids=stop)
        self.vocab_size = size
        self.words = math.ceil(size / 32)
        self._compiler = xgr.GrammarCompiler(self.info, max_threads=8)
        # compiles run off the worker's threads (xgrammar releases the GIL)
        self._compile_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="grammar-compile")
        self._batch = xgr.BatchGrammarMatcher()
        self._device = device
        self._rows: dict[Any, _Row] = {}
        # (rows, host tokens, event) per committed step not yet accepted, oldest first.
        # Rows, not handles: a request removed meanwhile leaves its row here, and its
        # last draw never reaches the next request given its handle.
        self._pending: list[tuple[list[_Row], torch.Tensor, Any]] = []
        self._shifts = torch.arange(32, dtype=torch.int32, device=device)

    def __contains__(self, rid) -> bool:
        return rid in self._rows

    def any_of(self, rids) -> bool:
        return bool(self._rows) and any(r in self._rows for r in rids)

    def add(self, rid, request: GrammarRequest) -> None:
        self._rows[rid] = _Row(self._compile_pool.submit(request.compile, self._compiler))

    def remove(self, rid) -> None:
        self._rows.pop(rid, None)

    def release(self, rids, why: str) -> None:
        """Stop constraining ``rids``: their draws can no longer be tracked."""
        for rid in rids:
            row = self._rows.pop(rid, None)
            if row is not None:
                logger.error("structured_outputs dropped for request %r: %s", rid, why)

    def note_draws(self, rids, tokens: torch.Tensor, kept) -> None:
        """Queue the copy of this step's drawn tokens for its constrained rows
        (``kept`` None keeps every row); the next ``stage`` accepts them."""
        index = [i for i, rid in enumerate(rids) if rid in self._rows and (kept is None or rid in kept)]
        if not index:
            return
        rows = [self._rows[rids[i]] for i in index]
        picked = tokens.reshape(-1)[torch.tensor(index, device=tokens.device)]
        if picked.device.type == "cpu":
            self._pending.append((rows, picked.clone(), None))
            return
        host = torch.empty(picked.shape, dtype=picked.dtype, pin_memory=True)
        host.copy_(picked, non_blocking=True)
        event = torch.Event(picked.device)
        event.record()
        self._pending.append((rows, host, event))

    def _accept_pending(self) -> None:
        for rows, host, event in self._pending:
            if event is not None:
                event.synchronize()
            for row, token in zip(rows, host.tolist(), strict=True):
                row.accept(token)
        self._pending.clear()

    def fill(self, rids) -> torch.Tensor:
        """The host bitmask for a step's rows, after accepting every draw so far:
        all ones for a free row."""
        self._accept_pending()
        if len(set(rids)) != len(rids):
            raise ValueError("a constrained request has two rows in one step (speculative decoding is not served)")
        mask = torch.full((len(rids), self.words), -1, dtype=torch.int32)
        matchers, index = [], []
        for i, rid in enumerate(rids):
            row = self._rows.get(rid)
            if row is not None and row.constrained:
                matchers.append(row.matcher)
                index.append(i)
        if len(matchers) == 1:
            matchers[0].fill_next_token_bitmask(mask, index[0])
        elif matchers:
            self._batch.batch_fill_next_token_bitmask(matchers, mask, index)
        return mask

    def apply(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        """Set every disallowed logit to -inf, in place; capturable (no host
        reads). Columns past the vocabulary the mask covers are disallowed."""
        rows, width = logits.shape[0], logits.shape[-1]
        covered = min(width, mask.shape[-1] * 32)
        if logits.is_cuda:
            from xgrammar.kernels.apply_token_bitmask_inplace_triton import apply_token_bitmask_inplace_triton

            # one pass, no host reads: capturable
            apply_token_bitmask_inplace_triton(logits, mask[:rows], vocab_size=covered)
        else:
            allowed = ((mask[:rows].unsqueeze(-1) >> self._shifts) & 1).reshape(rows, -1)
            logits[:, :covered].masked_fill_(allowed[:, :covered] == 0, float("-inf"))
        if width > covered:
            logits[:, covered:] = float("-inf")
