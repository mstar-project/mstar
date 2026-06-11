Quickstart
==========

This page serves a model and sends a first request. If you haven't installed ``mstar``
yet, see :doc:`installation`.

1. Start a server
-----------------

The ``mstar`` CLI launches a server for a model with a sensible default config:

.. code-block:: bash

   mstar serve bagel

It listens on ``http://localhost:8000`` by default. Other models:

.. code-block:: bash

   mstar serve bagel_cfg_parallel   # BAGEL with CFG branches split across GPUs (faster image gen)
   mstar serve qwen3_omni           # omni: text/image/audio/video in, text/speech out
   mstar serve orpheus              # text-to-speech
   mstar serve pi05                 # vision-language-action (robotics)
   mstar serve vjepa2               # video world model

Defaults vary by model: most fit on a single GPU, but some ship with multi-GPU layouts —
``qwen3_omni`` uses 2 GPUs and ``bagel_cfg_parallel`` uses 3 (the main branch plus the two
classifier-free-guidance branches on their own GPUs). Choose GPUs and a port with
``--gpus`` / ``--port``:

.. code-block:: bash

   mstar serve qwen3_omni --gpus 0,1 --port 9000

For custom layouts, disaggregation, and tensor parallelism, see :doc:`serving`.

.. note::

   The **first request(s) on a fresh environment can be slow** — often tens of seconds to a
   few minutes. mstar ``torch.compile``\ s the model on first use, and that compilation
   happens lazily on the first request that exercises each path. Subsequent requests run at
   full speed, and the compiled artifacts are cached on disk, so later runs and restarts
   warm up much faster. To avoid paying it on a real request, send a throwaway warmup
   request right after the server reports ready.

2. Send a request
-----------------

**Python SDK** — works for every model and modality. Each line below targets the model
that supports it, so run the matching server first:

.. code-block:: python

   from mstar import MStarClient

   client = MStarClient("http://localhost:8000")

   print(client.chat("What is the capital of France?").text)              # text  (BAGEL / Qwen3-Omni)
   open("cat.png", "wb").write(client.generate_image("a cat in a hat"))   # image (BAGEL)
   client.tts("Hello there", voice="tara").to_wav("out.wav")             # speech (Orpheus)

Streaming yields typed chunks (``TextChunk`` / ``ImageChunk`` / ``AudioChunk``):

.. code-block:: python

   from mstar.client import TextChunk

   for event in client.chat("Tell me a short story.", stream=True):
       if isinstance(event, TextChunk):
           print(event.text, end="", flush=True)

**curl** — the native ``/generate`` endpoint works for every model:

.. code-block:: bash

   curl -s http://localhost:8000/generate -F 'text=Hello, how are you?'

**OpenAI-compatible API** — a drop-in client for ``bagel``, ``qwen3_omni``, and
``orpheus``:

.. code-block:: python

   from openai import OpenAI

   client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
   resp = client.chat.completions.create(
       model="bagel",
       messages=[{"role": "user", "content": "Give me one fun fact."}],
   )
   print(resp.choices[0].message.content)

Runnable versions of all of these live in the repo's ``examples/`` directory
(``sdk_chat.py``, ``sdk_image.py``, ``sdk_tts.py``, ``openai_chat.py``, ``openai_tts.py``,
``curl.sh``). For the full client surface, see :doc:`clients`.

3. Check health
---------------

.. code-block:: bash

   curl http://localhost:8000/health        # -> {"status": "healthy"}
