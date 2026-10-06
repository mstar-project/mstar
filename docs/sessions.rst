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

Session state need not be a resource's. ``Cosmos3Model`` declares sessions with
an empty ``resources`` and keeps a windowed rollout's world in its submodules
instead: the DiT node holds the last clean window, the streaming decoder its
decode context, so a later request resumes the same world with a new prompt.
Only the caps (concurrency, TTL) come from its ``SessionsConfig``.

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
``GET /sessions`` lists what the server holds.

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
  ``session_timeout_s`` over the deployment's maximum.
- ``404`` — resuming or deleting a session that does not exist.
- ``409`` — starting a session whose id is taken; resuming or deleting one that
  already has a request in flight, or that is being torn down.
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

``test_text_session`` is a deployment that exists to exercise this: BAGEL's LLM
with everything else taken away, text in and text out, holding its KV for the
session. One GPU.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 bash test/text_session/launch_server.sh
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
- State parked between a session's requests is not an eviction candidate: the
  worker's LRU only sees live requests. Size ``max_concurrent_sessions`` times
  each resource's ``max_state`` against the pool, leaving room for the requests
  ``max_concurrent_requests`` allows, or a full pool will start refusing
  admission. (A session's state does follow its request through an offload
  while that request is running.)
- Sessions are reachable through the ``POST /generate`` form, the
  ``/generate/ws`` control-loop socket, ``/v1/chat/completions`` and the SDK. The
  other OpenAI-compatible routes have no use for them (one-shot speech, image
  and video generation), and the optional Rust frontend (``--rust-frontend``)
  carries no session fields: it serves its own HTTP surface, so it would need
  the five fields on its ``/generate`` and chat routes, the session chunk in its
  writers, and new bridge messages for ``GET /sessions`` and
  ``DELETE /sessions/{id}``, which live only in the Python server.
- Only ``test_text_session`` declares session support. Any other deployment
  refuses sessions until its model opts in.
