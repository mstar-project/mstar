"""Grammar-constrained sampling: a request's constraint on which token ids it
may draw next, advanced by the tokens it drew (JSON schema, JSON object, regex,
choice list, EBNF; xgrammar underneath).

The sampler resource owns it, beside its other per-request draw state. A step
that holds a constrained row fills each row's bitmask as its draw begins, from
the tokens the rows drew in their last step: the sampler copies those to the
host at commit, and the next step waits for that copy alone (step N's draw, not
the worker's postprocess), accepts them and fills the masks. A captured step
splits there (`split_capture`), so its forward is launched first and the masks
are filled while it runs. The draw then applies the mask before either sampling
path, so a captured step bakes the apply over the slot's static mask buffer.
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mstar.utils.cpus import plan as cpu_plan
from mstar.utils.profiler import phase_end, phase_start
from mstar.utils.threads import ThreadPool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mstar.model.utils import TextOutput

# The request field: vLLM's ``structured_outputs`` shape (one of these keys), plus
# ``after``, the text that opens the constrained part (a reasoning model's think-end tag).
STRUCTURED_OUTPUTS_KWARG = "structured_outputs"
GRAMMAR_KINDS = ("json", "json_object", "regex", "choice", "grammar")
# a grammar compiles while its request waits for admission, so its size is bounded
MAX_GRAMMAR_CHARS = 1 << 17
# xgrammar's batch fill starts every one of its threads on each call, however few the rows:
# measured on the dev machine, about 25 us of the caller's time per thread, and a 6-row fill
# took 0.29 ms serially against 0.23 ms with 2 threads. A fill takes a thread per this many
# rows, up to 8 threads; the 16-row cut is provisional until the H200 profile sets it
FILL_ROWS_PER_THREAD = 8


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
        import xgrammar as xgr

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
            import xgrammar as xgr

            return xgr.Grammar.from_structural_tag(self._structural_tag(content))
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


# xgrammar is an optional dependency (the `grammar` extra), imported only where a grammar is built
XGRAMMAR_INSTALL = "pip install 'mstar-ai[grammar]'"


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
    size = len(raw) + len(after or "")
    if size > MAX_GRAMMAR_CHARS:
        raise ValueError(f"structured_outputs is {size} characters; at most {MAX_GRAMMAR_CHARS} are served")
    request = GrammarRequest(kind, raw, after)
    try:
        request.grammar()
    except ImportError as e:
        raise ValueError(f"structured_outputs needs xgrammar on the server ({XGRAMMAR_INSTALL})") from e
    return request


def request_grammar(model_kwargs: dict | None) -> GrammarRequest | None:
    """The grammar a request's kwargs ask for (``structured_outputs``), checked, for a model to
    attach to its `SampleSettings`; None when unset. ``ValueError`` (the client's 400) for a bad one."""
    return grammar_request((model_kwargs or {}).get(STRUCTURED_OUTPUTS_KWARG))


def after_reasoning(structured: Any, text_output: TextOutput | None, prompt_ids: torch.Tensor) -> Any:
    """``structured`` with ``after`` the model's ``reasoning_end`` when the prompt
    leaves the reasoning open, so a thinking turn reasons freely and then answers
    in its grammar; as it came on a turn whose prompt ends past the reasoning.
    The check reads the last 8 ids and the whitespace before the end does not
    count. While they do not end in ``reasoning_end``, it drops trailing special
    tokens one at a time, reading again after each, so a template may add special tokens after the end
    (MiniCPM-o's ``<|tts_bos|>``) and an end made of special tokens still matches."""
    end = None if text_output is None else text_output.reasoning_end
    if end is None or not isinstance(structured, dict) or "after" in structured:
        return structured
    detokenizer = text_output.detokenizer
    ids = prompt_ids[-32:].tolist()
    while True:
        tail = b"".join(map(detokenizer.token_bytes, ids[-8:]))
        if tail.decode("utf-8", errors="replace").rstrip().endswith(end):
            return structured
        if not ids or ids[-1] not in detokenizer.kept_special_ids:
            return {**structured, "after": end}
        ids.pop()


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
    """One request's matcher, compiled in the background while its request is
    not ready (`GrammarSamplerResource.admit_retrieve`), and ``error``: why the
    row can no longer be served, which fails its request at its next admit."""
    __slots__ = ("compiled", "_matcher", "error")

    def __init__(self, compiled: Future):
        self.compiled = compiled
        self._matcher = None
        self.error: str | None = None

    @property
    def matcher(self):
        """None once the row failed; the first use waits for the compile."""
        if self._matcher is None and self.error is None:
            failed = self.compiled.exception()
            if failed is not None:
                self.error = f"structured_outputs grammar failed to compile: {failed}"
            else:
                import xgrammar as xgr

                try:
                    self._matcher = xgr.GrammarMatcher(self.compiled.result())
                except Exception as exc:  # noqa: BLE001 - the request's error, not the step's
                    self.error = f"structured_outputs grammar failed to start: {exc}"
        return None if self.error is not None else self._matcher

    @property
    def constrained(self) -> bool:
        matcher = self.matcher
        return matcher is not None and not matcher.is_terminated()

    def accept(self, token: int) -> None:
        if self.constrained and not self._matcher.accept_token(token):
            # the mask allowed it, so the draw was not the one the mask covered
            self.error = f"structured_outputs grammar refused drawn token {token}"


class GrammarRows:
    """The sampler's constrained rows: one matcher per request, the drawn tokens
    still on their way to the host, and the bitmask each step applies. ``decides``: this
    rank serves alone, so a row whose fill fails fails its request (`fill`)."""

    def __init__(
        self, text_output: TextOutput, vocab_size: int | None, device: torch.device, cache_bytes: int, decides: bool,
    ):
        import xgrammar as xgr

        detok = text_output.detokenizer
        stop = sorted(int(t) for t in text_output.stop_token_ids)
        tokenizer = getattr(detok, "tokenizer", None)
        if tokenizer is not None:
            if vocab_size is not None and vocab_size < len(tokenizer):
                raise ValueError(
                    f"enable_grammar belongs on the text sampler; this one draws from {vocab_size} ids, "
                    f"fewer than the {len(tokenizer)} of the text vocabulary"
                )
            size = vocab_size or len(tokenizer)
            self.info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=size, stop_token_ids=stop or None)
        else:
            if vocab_size is None:
                raise ValueError("enable_grammar needs the SamplerSpec's vocab_size for this model's detokenizer")
            size = vocab_size
            vocab = [detok.token_bytes(i) for i in range(size)]
            self.info = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, vocab_size=size, stop_token_ids=stop)
        self.vocab_size = size
        self._decides = decides
        self.words = math.ceil(size / 32)
        # the ids xgrammar resolved as stops (the model's, else its tokenizer's EOS), which a
        # failed row is kept from (`fill`)
        stop = sorted(self.info.stop_token_ids)
        bits = torch.zeros(self.words * 32, dtype=torch.int64)
        bits[stop] = 1
        words = (bits.view(-1, 32) << torch.arange(32)).sum(-1)
        self._stops = torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32) if stop else None
        # sized by the CPUs the process's plan leaves unpinned (not the step threads'),
        # which xgrammar's own "auto" doesn't read
        cpus = cpu_plan().left
        self._compiler = xgr.GrammarCompiler(self.info, max_threads=min(8, cpus), cache_limit_bytes=cache_bytes)
        # compiles run off the worker's threads (xgrammar releases the GIL)
        self._compile_pool = ThreadPool(max_workers=min(2, cpus), thread_name_prefix="grammar-compile")
        self._fill_threads = max(1, min(8, cpus // 2))
        self._batches: dict[int, Any] = {}  # a batch matcher by its thread count
        self._device = device
        self._rows: dict[Any, _Row] = {}
        # (rows, their indices in the step, the step's whole row of draws on the host, the
        # event its copy waits on) per committed step not yet accepted, oldest first. Rows,
        # not handles: a request removed meanwhile leaves its row here, and its last draw
        # never reaches the next request given its handle.
        self._pending: list[tuple[list[_Row], list[int], torch.Tensor, Any]] = []
        self._shifts = torch.arange(32, dtype=torch.int32, device=device)

    def __contains__(self, rid) -> bool:
        return rid in self._rows

    def any_of(self, rids) -> bool:
        return bool(self._rows) and any(r in self._rows for r in rids)

    def add(self, rid, request: GrammarRequest) -> None:
        self._rows[rid] = _Row(self._compile_pool.submit(request.compile, self._compiler))

    def remove(self, rid) -> None:
        self._rows.pop(rid, None)

    def compiling(self, rid) -> bool:
        row = self._rows.get(rid)
        return row is not None and not row.compiled.done()

    def uncompiled(self, rid) -> bool:
        """Whether ``rid``'s grammar failed to compile: the request's own fault."""
        row = self._rows.get(rid)
        return row is not None and row.compiled.done() and row.compiled.exception() is not None

    def failures(self, rids) -> dict:
        """Why each of ``rids`` that can no longer be served can't, by rid."""
        rows = self._rows
        return {rid: row.error for rid in rids if (row := rows.get(rid)) is not None and row.matcher is None}

    def note_draws(self, rids, tokens: torch.Tensor, kept) -> None:
        """Queue the copy of this step's drawn tokens for its constrained rows
        (``kept`` None keeps every row); the next `accept_pending` accepts them."""
        index = [i for i, rid in enumerate(rids) if rid in self._rows and (kept is None or rid in kept)]
        if not index:
            return
        rows = [self._rows[rids[i]] for i in index]
        drawn = tokens.reshape(-1)[:len(rids)]
        if drawn.device.type == "cpu":
            self._pending.append((rows, index, drawn.clone(), None))
            return
        # the whole row of draws, picked on the host: a device index built from a host
        # list is a pageable copy, which waits for the step just launched to finish
        host = torch.empty(drawn.shape, dtype=drawn.dtype, pin_memory=True)
        host.copy_(drawn, non_blocking=True)
        event = torch.Event(drawn.device)
        event.record()
        self._pending.append((rows, index, host, event))

    def accept_pending(self) -> None:
        """Advance each row by the draws `note_draws` queued, once they reach the
        host. A row whose accept raises fails (`failures`), and the rest go on: the
        same on every rank, which accept the same draws."""
        pending = self._pending
        # one sample per step, however many steps' draws are pending
        waited = phase_start() if pending else None
        for *_, event in pending:
            if event is not None:
                event.synchronize()
        phase_end("res.grammar.draw_wait", waited)
        while pending:
            rows, index, host, _ = pending[0]
            drawn = host.tolist()
            # taken before its accepts, so no draw is accepted twice, and none is lost
            # to a raise above, which leaves this step's draws and every later one queued
            pending.pop(0)
            for row, i in zip(rows, index, strict=True):
                try:
                    row.accept(drawn[i])
                except Exception as exc:  # noqa: BLE001 - that request's error, not the step's
                    logger.exception("structured_outputs: accepting a drawn token failed")
                    row.error = f"structured_outputs grammar failed to accept drawn token {drawn[i]}: {exc}"

    def fill(self, rids) -> torch.Tensor:
        """The host bitmask for a step's rows: all ones for a free row. A row whose grammar
        failed (in this step's accepts, alike on every rank) draws anything but a stop, so its
        request reaches the next admit and fails there with the error instead of ending as a
        200. A row whose own fill raises does the same when this rank serves alone; under
        several ranks it may only stop, since only rank 0 fills and an error set there would
        split the admits. A model that resolves no stop gets all ones. If the batched call
        raises, its rows fill one by one, so only the bad row fails."""
        mask = torch.full((len(rids), self.words), -1, dtype=torch.int32)
        matchers, index = [], []
        for i, rid in enumerate(rids):
            row = self._rows.get(rid)
            if row is None:
                continue
            try:
                matcher = row.matcher
                if matcher is None:
                    if self._stops is not None:
                        mask[i] = ~self._stops
                elif not matcher.is_terminated():
                    matchers.append(matcher)
                    index.append(i)
            except Exception as exc:  # noqa: BLE001 - that request's error, not the step's
                logger.exception("structured_outputs: reading a row's grammar failed")
                self._failed(row, mask, i, exc)
        threads = min(self._fill_threads, len(matchers) // FILL_ROWS_PER_THREAD)
        if threads >= 2:
            # a matcher's raise inside a batch worker comes back as a Python error (xgrammar 0.2.9)
            try:
                batch = self._batches.get(threads)
                if batch is None:
                    import xgrammar as xgr

                    batch = self._batches[threads] = xgr.BatchGrammarMatcher(max_threads=threads)
                batch.batch_fill_next_token_bitmask(matchers, mask, index)
                return mask
            except Exception:  # noqa: BLE001 - the loop below finds the row
                logger.exception("structured_outputs: a batched mask fill failed, retrying row by row")
        # a fill rewrites its whole row and leaves the matcher's state, so a retry is sound
        for matcher, i in zip(matchers, index, strict=True):
            try:
                matcher.fill_next_token_bitmask(mask, i)
            except Exception as exc:  # noqa: BLE001 - that request's error, not the step's
                logger.exception("structured_outputs: filling a row's grammar mask failed")
                self._failed(self._rows[rids[i]], mask, i, exc)
        return mask

    def _failed(self, row: _Row, mask: torch.Tensor, i: int, exc: Exception) -> None:
        if self._decides and row.error is None:
            row.error = f"structured_outputs grammar failed to fill its mask: {exc}"
        if self._stops is None:
            mask[i] = -1
        else:
            mask[i] = ~self._stops if self._decides else self._stops

    def apply(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        """Set every disallowed logit to -inf, in place; capturable (no host
        reads). Columns past the vocabulary the mask covers are disallowed."""
        rows, width = logits.shape[0], logits.shape[-1]
        covered = min(width, mask.shape[-1] * 32)
        if logits.is_cuda:
            import xgrammar as xgr

            # one Triton pass, no host reads: capturable
            xgr.apply_token_bitmask_inplace(logits, mask[:rows], vocab_size=covered)
        else:
            allowed = ((mask[:rows].unsqueeze(-1) >> self._shifts) & 1).reshape(rows, -1)
            logits[:, :covered].masked_fill_(allowed[:, :covered] == 0, float("-inf"))
        if width > covered:
            logits[:, covered:] = float("-inf")
