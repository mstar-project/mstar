Quickstart
==========

Start BAGEL and send it a text prompt. This guide uses the Python SDK and a server on the same machine.

Before starting, follow `Installation <installation.html>`_ for Linux with an NVIDIA GPU, Python 3.12 or later, and the ``bagel`` extra. The default BAGEL deployment puts all nodes on one GPU; that GPU must have enough memory for the model and its resources. The server downloads ``ByteDance-Seed/BAGEL-7B-MoT`` from Hugging Face when it is not cached, so it needs access to that checkpoint. For Intel GPUs, use the `Intel installation path <installation.html#using-docker-image-on-intel-gpu>`_ instead.

.. _1-start-a-server:

Start a server
--------------

In your installed M* environment, run:

.. code-block:: bash

   mstar serve bagel

Leave this terminal open. The server listens at ``http://localhost:8000``. This command selects the default single-GPU BAGEL configuration and Python HTTP frontend. See `Serving <serving.html>`_ for a different GPU layout, port, or the Rust frontend.

.. _3-check-health:

Check health
------------

In a second terminal on the same machine, check the server:

.. code-block:: bash

   curl --fail http://localhost:8000/health

A healthy response contains:

.. code-block:: json

   {"status":"healthy"}

If the connection is refused, check the first terminal for a startup error or wait for startup to finish and retry. This check confirms HTTP health; the next request checks that the model can produce text.

.. _2-send-a-request:

Send a request
--------------

In the second terminal, activate the same Python environment you used to start M*. Run:

.. code-block:: bash

   python - <<'PYTHON'
   from mstar import MStarClient

   client = MStarClient("http://localhost:8000")
   response = client.chat("What is the capital of France?")
   print(response.text)
   PYTHON

The program prints the model's answer from ``response.text``. Look for a text response to the question; the exact wording can vary. The first request may take longer while M* compiles the paths it uses. Do not use that request to judge steady-state latency.

To stream a response from the same server:

.. code-block:: python

   from mstar import MStarClient
   from mstar.client import TextChunk

   client = MStarClient("http://localhost:8000")
   for event in client.chat("Tell me a short story.", stream=True):
       if isinstance(event, TextChunk):
           print(event.text, end="", flush=True)
   print()

When finished, press Ctrl+C in the server terminal.

Try another task
----------------

- **Generate an image with BAGEL:** keep this server running and follow `the image SDK example <https://github.com/mstar-project/mstar/blob/main/examples/sdk_image.py>`_.
- **Generate speech with Orpheus:** install the ``orpheus`` extra, stop BAGEL, and start ``mstar serve orpheus``. Follow `the speech SDK example <https://github.com/mstar-project/mstar/blob/main/examples/sdk_tts.py>`_.
- **Use text, image, audio, or video inputs with Qwen3-Omni:** follow `the model guidance <models.html>`_ and `Serving <serving.html>`_. Its default configuration uses two GPUs; it is a separate server setup.
- **Explore robotics and video models:** see `supported models <models.html>`_ for Pi0.5 and V-JEPA 2 and their installation requirements.
- **Use curl or an OpenAI client:** see `the native endpoint <clients.html#native-generate>`_ or `OpenAI-compatible APIs <clients.html#openai-compatible-api>`_. Check that the chosen endpoint and modality are supported by your model.
- **Change GPU placement:** see `Serving <serving.html>`_, including the three-GPU ``bagel_cfg_parallel`` configuration.

The repository also contains `the chat SDK example <https://github.com/mstar-project/mstar/blob/main/examples/sdk_chat.py>`_, `OpenAI chat <https://github.com/mstar-project/mstar/blob/main/examples/openai_chat.py>`_, `OpenAI speech <https://github.com/mstar-project/mstar/blob/main/examples/openai_tts.py>`_, and `curl examples <https://github.com/mstar-project/mstar/blob/main/examples/curl.sh>`_.
