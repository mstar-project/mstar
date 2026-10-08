Persistent sessions
===================

A **session** is a named context a client comes back to. Requests in a session
continue from the state the previous one left behind — the KV a conversation
built, the latents a denoising run ended on — instead of starting from an empty
cache. Without a session, every request opens its resource state at ingest and
frees it at teardown; a session moves that lifetime out one level, from the
request to the session.

Sessions are opt-in per model. A model that declares no
:class:`~mstar.model.sessions.SessionsConfig` refuses every session request with
a 400, and a deployment cannot turn them on for it.

What the model declares
-----------------------

``Model.get_sessions_config()`` names the resources whose state lives for the
session and the deployment-facing caps:

.. code-block:: python

   from mstar.model.sessions import SessionResourceConfig, SessionsConfig

   def get_sessions_config(self):
       return SessionsConfig(
           resources={
               # max_state is in pages, this resource's own unit
               "kv_cache": SessionResourceConfig(max_state=512),
           },
           max_concurrent_sessions=8,
           default_timeout_s=300.0,
           max_timeout_s=3600.0,
       )

The engine also holds session state for anything built *against* a named
resource — the position counters over a KV cache, for instance. Their state
addresses the state being kept, so leaving it behind would have the next
request write positions from 0 over pages the session still holds.

A hybrid model names every cache its node keeps state in. Qwen3.5's layers
carry both a KV cache and a recurrent state pool (each GDN layer's state matrix
and conv window), so its sessions config names both: the pool parks its slots
under the session as the KV manager parks its pages. Naming one alone is
refused at load, since a resumed request would continue the one and start the
other from zero. The pool's ``max_state`` is in slots, one per label.

Prefix reuse and sessions
^^^^^^^^^^^^^^^^^^^^^^^^^

A resource may do both. A session's *first* request is keyed like any other, and
what it files stays reusable by everyone. A **resumed** request steps out of the
index entirely: its keys cover the new turn alone, so page *k* of its stream is
no longer page *k* of the chain those keys describe — probing would match the
wrong span, and filing would offer the session's pages to other requests under
keys that do not describe them. Adopting a session's state therefore turns that
one request's prefix cache off, which shuts the probe, the apply, the extend and
the filing together.

Outgrowing the budget
^^^^^^^^^^^^^^^^^^^^^

``max_state`` is checked when a request hands its state back to the session, not
on every step, so an in-flight request may exceed it. A session over budget is
dropped, and its next request is refused with the reason — the client is told
rather than silently served from an empty context it believes still holds the
conversation. There is no quieter option: a session whose state vanished between
turns is not one a client can reason about.

The budget is a backstop against a session outgrowing what the deployment will
hold, not a way to trim one. **A bounded, rolling context is a different thing
and belongs to the resource**: a KV stream declares a
:class:`~mstar.engine.resources.kv.config.RetentionPolicy` on its step, and
``commit`` releases its oldest whole pages behind ``protected_prefix`` as the
stream grows — which is how sliding-window attention holds a fixed context
indefinitely. A deployment that wants a session to keep generating forever wants
that, not a budget it periodically trips.

What a request's model sees
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`~mstar.model.sessions.RequestSession` — the id, whether it continues
state already there, and whether it ends the session — reaches the model with
the request's initial forward-pass args, and rides on every
``CurrentForwardPassInfo`` after it, so a submodule reads it off the step. It is
what the server validated, and it arrives as a property of the request rather
than among the client's ``model_kwargs``.

A model that instead reads a session id out of ``model_kwargs`` (cosmos3 does
today, with a session store of its own) is not going through any of this: no
registry, no TTL, no teardown barrier. Porting such a model means reading the
``RequestSession`` above.

Submodule state
^^^^^^^^^^^^^^^

Alongside the per-request ``PerRequestState``, a submodule has a per-session
one. ``self.session_state(session_id)`` reaches it directly; a forward reads
the batch's through ``ModelInputsFromEngine.per_session_states``, keyed by
request id (``None`` for a request in no session). The engine drops it at
session teardown, so a submodule needs no cleanup code of its own.

What a deployment tunes
-----------------------

A ``sessions:`` block in the serving config layers over the model's
declaration. It may retune the caps and the per-resource budgets, and it may
turn sessions off with ``enabled: false``; it may not name a resource the model
does not hold across a session.

.. code-block:: yaml

   sessions:
     max_concurrent_sessions: 32
     default_timeout_s: 120.0
     max_timeout_s: 900.0
     ttl_mode: idle          # or: absolute
     resources:
       kv_cache:
         max_state: 1024

``ttl_mode: idle`` expires a session ``timeout_s`` after its last request
finished; ``absolute`` expires it that long after it was started, however busy
it is. A session with a request in flight is never collected.

Expiry is noticed by a sweep rather than a timer, so a session outlives its
deadline by up to one sweep. ``MSTAR_SESSION_SWEEP_INTERVAL_S`` (1.0s) sets how
often it runs: raise it on a deployment holding many sessions, lower it to make
a short TTL expire promptly in a test. See :doc:`environment_variables`.

``capacity_policy`` decides what a deployment at ``max_concurrent_sessions``
does with a new session. ``keep`` (the default) holds every session until its client ends it
or its TTL expires, and refuses a new one past the cap with a 429. ``evict``
instead tears down the least recently used **idle** session to make room. A
session with a request in flight is never evicted — it is writing its state
right now — so a deployment whose sessions are all in flight still refuses.

The HTTP API
------------

``POST /generate`` takes five session fields:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``start_session``
     - Open a session and run this request in it. With no ``session_id`` the
       server mints one.
   * - ``resume_session``
     - Continue the session named by ``session_id``.
   * - ``end_session``
     - Tear the session down once this request finishes.
   * - ``session_id``
     - The id to start with, or the one to resume.
   * - ``session_timeout_s``
     - This session's TTL. Defaults to the deployment's, and may not exceed its
       maximum.

A started session reports its id back: as the first result chunk when
streaming (modality ``"session"``, with ``session_id`` in its metadata), and as
``session_id`` in the JSON body otherwise.

``DELETE /sessions/{id}`` ends a session without a request, and
``GET /sessions/{id}`` reports one (its TTL, whether a request is in flight,
whether it is closing). ``GET /sessions`` gives counts only — live, closing,
and the cap — never ids: the server has no auth, and an id is all it takes to
resume or delete a session.

``/generate/ws`` takes the same five fields on each message, and answers a
refused session in-band with the status the form route would have given, so a
control loop can tell a 409 from a 404 without dropping its socket.

``POST /v1/chat/completions`` takes the same five fields as top-level request
keys — an mstar extension, since OpenAI has no session concept. A turn in a
session carries that turn's messages alone rather than the whole transcript, and
the server answers with ``session_id`` beside ``choices`` (or on the stream's
opening chunk). The other OpenAI routes generate one-shot speech, images and
video; they ignore the session fields rather than refusing them, since the
request models accept unknown keys.

.. code-block:: bash

   curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "bagel", "start_session": true,
     "messages": [{"role": "user", "content": "Who painted Guernica?"}]
   }'
   # -> {... "choices": [...], "session_id": "9f2c..."}

   curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "bagel", "resume_session": true, "session_id": "9f2c...",
     "messages": [{"role": "user", "content": "And when?"}]
   }'

Refusals
^^^^^^^^

- ``400`` — the model does not support sessions; ``start_session`` and
  ``resume_session`` together; a session named without either flag; a
  ``session_timeout_s`` that is not a positive number or is over the
  deployment's maximum; a ``session_id`` to start with that is not 1-128
  letters, digits, ``_`` and ``-``.
- ``404`` — resuming or deleting a session that does not exist.
- ``409`` — starting a session whose id is taken; resuming or deleting one that
  already has a request in flight, or that is being torn down.
- ``410`` — resuming a session that outgrew its state budget; its state was
  dropped when its last request finished (see `Outgrowing the budget`_).
- ``429`` — the deployment is at ``max_concurrent_sessions``.

From the SDK
------------

.. code-block:: python

   from mstar import MStarClient

   client = MStarClient()
   first = client.generate(text="Who painted Guernica?", start_session=True)
   session_id = first.session_id

   second = client.generate(
       text="And when?", resume_session=True, session_id=session_id,
   )

   client.end_session(session_id)

Streaming yields a ``SessionInfo`` event first, then the output chunks.

How it holds together
---------------------

The API server is the only thing that decides a session's fate. It tracks each
session's TTL, its one in-flight request, and the tombstone that stands while
its state is being freed; the conductor and the workers carry that out.

- **Placement is pinned.** The conductor memoizes a session's replica pick and
  reuses it for every request in the session. The state a resumed request
  continues lives on the workers that built it, so a fresh pick would route the
  request to workers holding nothing.
- **Teardown is a barrier.** Freeing a session's state — on ``DELETE``, on a
  TTL expiry, on a failed request, or riding on a request that carried
  ``end_session`` — goes to every worker the session ran on, and each ACKs once
  its state is gone. Until every ACK is in, the API server refuses the id. A
  worker that never ACKs is timed out (``MSTAR_DRAIN_TTL_S``) so one faulty
  worker cannot hold an id forever, and the API server releases a tombstone
  that stands past ``MSTAR_SESSION_TOMBSTONE_GRACE_S`` (180s) regardless.
- **A failure ends the session.** v1 keeps no rollback, so a request that
  failed or was abandoned mid-generation leaves state a later request should
  not continue from. Its session is torn down with it.

Trying it
---------

``test_text_session`` is a deployment that exists to exercise this, in two
variants picked by the config's ``model:``, both text in and text out on one GPU:

- ``test_text_session`` (``configs/test_text_session.yaml``): BAGEL's LLM with
  everything else taken away, holding its KV for the session.
- ``test_text_session_qwen3_5`` (``configs/test_text_session_qwen3_5.yaml``):
  Qwen3.5's LLM without its vision tower, holding both its KV cache and its GDN
  recurrent state. 4B by default; ``model_kwargs: {model_path_hf:
  Qwen/Qwen3.5-9B}`` picks 9B. Thinking is always off: Qwen's template drops a
  past turn's ``<think>`` block, reasoning and tags together, so a reasoning
  trace kept in the session would sit in context where a resent transcript has
  none. With it off, the one difference left is the pre-closed
  ``<think>\n\n</think>\n\n`` each turn was generated after, which the KV keeps
  and the template would not.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 bash test/text_session/launch_server.sh
   # or: CONFIG=configs/test_text_session_qwen3_5.yaml bash test/text_session/launch_server.sh
   python test/text_session/session_request.py

The client script sends a turn, resumes with a second turn whose prompt says
nothing about the first, and checks the answer could only have come from the KV
the session kept — a sessionless control run of the same question should not know
it. It then walks the teardown: delete the session, wait for the id to be
released (which happens only once the worker has confirmed its state is gone),
and check that resuming it is a 404.

``budget_probe.py`` beside it proves the same accumulation without reading a
single generated token: it sends long turns into one session until the held state
passes the configured ``max_state``, and the turn after that has to come back
with the budget error.

What a model renders for a resuming turn
---------------------------------------

``process_prompt`` receives the request's session (``session_id``, and
``started`` / ``resumed`` / ``end_session``), so a model that renders a chat
envelope can tell an opening turn from one that continues a conversation already
in the KV. A resuming turn should render as one more turn and nothing else: a
re-rendered system prompt or leading BOS lands in the middle of the
conversation, which a small model reads as being introduced to itself again.

``TextSessionModel.process_prompt`` is the worked example: each turn renders in
its own role block, and a resuming turn renders the user block and the
assistant's opener alone. A model that ignores the argument renders as it always
did.

The other end of the join is the previous reply, and what the KV holds of it
depends on how it stopped. The token that stopped the decode loop (the EOS for a
finished reply, the last token for one ``max_output_tokens`` cut) is fed back
only when a speculative step had already run past the stop, which under async
scheduling is the common case. The worker records that overshoot per node, and
when the request hands its state back to a continuing session, the engine sets
``OVERSHOT_LAST_ITER`` in that node's submodule session state (and clears it for
a turn that stopped exactly). Which token stopped the loop, and whether it was
the EOS, is the model's to record: its ``check_stop`` sees the turn's real last
token exactly once, since an overshooting step's outputs are dropped before the
stop check.

Both ``test_text_session`` variants' LLMs read both on a resumed turn's first
prefill and put back what the KV is missing (``test_text_session/turn_join.py``):

==========================  ===================  ================================
last reply ended on         its token in the KV  its token not in the KV
==========================  ===================  ================================
``<|im_end|>``              ``\n``               ``<|im_end|>\n``
another stop token          ``<|im_end|>\n``     the token, ``<|im_end|>\n``
a cut                       nothing              the token
==========================  ===================  ================================

(Qwen3.5 can also stop on ``<|endoftext|>``, which the template never closes a
turn with; BAGEL stops on ``<|im_end|>`` alone.)

A cut reply is left open rather than closed like a finished one: the KV holds
the whole reply the client saw and nothing after it, so a next turn asking to
continue reads as the reply being interrupted rather than finished. The flags
are cleared once that prefill has run, not when it is prepared, since a step
refused at admission is prepared again.

The role label is not cosmetic. Over 20 greedy two-turn chats on that model,
asked ``What is my name?`` after being told it:

==========================================  ==============  =================
resuming turn renders as                    reference kept  roles kept straight
==========================================  ==============  =================
a system block and no role labels           9/20            4/20
one role block, no system block             19/20           20/20
one role block, system block on turn one    20/20           20/20
==========================================  ==============  =================

With no label the model cannot tell its own turn from the client's, and answers
as though the client's name were its own.

Limits in this version
----------------------

- One in-flight request per session: a concurrent ``resume`` is a 409. Lifting
  it waits on bidirectional streaming, which needs the runtime to route the
  second request's inputs into the first; two requests sharing a session's
  resource state would corrupt it.
- No rollback: a failed request ends its session rather than rewinding it.
- State parked between a session's requests goes to host memory when a running
  request needs its pages, longest-idle session first and before any cached
  prefix is evicted, but only for a KV cache with ``cpu_offload_pages``; the
  next turn brings it back at admission. Without a host pool, and for a resource
  with no offload (recurrent state), parked state holds its pages or slots until
  the session ends: size ``max_concurrent_sessions`` times each resource's
  ``max_state`` against the pool, leaving room for the requests
  ``max_concurrent_requests`` allows, or a full pool will start refusing
  admission. For a recurrent pool that means ``max_slots`` covers the parked
  sessions' slots as well as one per label for every running request. (A
  session's state does follow its request through an offload while that
  request is running.)
- Sessions are reachable through the ``POST /generate`` form, the
  ``/generate/ws`` control-loop socket, ``/v1/chat/completions`` and the SDK. The
  other OpenAI-compatible routes have no use for them (one-shot speech, image
  and video generation), and the optional Rust frontend (``--rust-frontend``)
  carries no session fields: it serves its own HTTP surface, so it would need
  the five fields on its ``/generate`` and chat routes, the session chunk in its
  writers, and new bridge messages for ``GET /sessions``, ``GET /sessions/{id}`` and
  ``DELETE /sessions/{id}``, which live only in the Python server.
- Only the two ``test_text_session`` variants declare session support. Any
  other deployment, the stock Qwen3.5 and BAGEL included, refuses sessions
  until its model opts in.
