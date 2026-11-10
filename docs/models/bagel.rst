BAGEL
=====

Unified multimodal model (text + image understanding and generation).

Registry key: ``bagel``. Checkpoint: ``ByteDance-Seed/BAGEL-7B-MoT``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``bagel``
     - ``configs/bagel_single_gpu.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve bagel

For client requests and response formats, see :doc:`/clients`.

Guidance and numerical differences
----------------------------------

Text-to-image runs only the CFG branches a denoising step reads: two with
image guidance off, and the main branch alone outside ``cfg_interval``, about
-37% latency at 1024² on an H200. A step then computes fewer rows and returns
the main branch's velocity outside the interval where it used to recombine all
three, which is equal only up to rounding, so images move. Against the
previous output on one GPU, seeded text-to-image at the default guidance is
34 dB PSNR at worst and 45 dB at the median (58–60 dB between two boots of
one version), text-to-image without guidance 42 / 53 dB, and edits 24 / 40 dB,
about what two runs of one version's edit differ by (23 / 39 dB).

Disaggregated image generation
------------------------------

A guided request's steps inside ``cfg_interval`` run as a walk of their own,
``image_gen_guided2`` (``image_gen_guided3`` with image guidance), between
``image_gen_head`` and ``image_gen``. A config that places ``LLM`` per walk
(``configs/bagel_*_disaggregated.yaml``) lists all four on the rank that
generates images.

CFG parallel placement
----------------------

On a CFG-parallel deployment (``configs/bagel_cfg_parallel*.yaml``) those
steps run on the guidance GPUs instead, as the ``image_gen_cfg2`` walk
(``image_gen_cfg3`` with image guidance); the other steps run on the main
GPU. A config that places ``LLM_cfg_text`` or ``LLM_cfg_img`` with
``graph_walks`` names these walks: ``LLM_cfg_text`` and ``combine_cfg`` both,
``LLM_cfg_img`` only ``image_gen_cfg3``.
